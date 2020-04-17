/*
 * This file is part of DeepFactors.
 *
 * Copyright (C) 2020 Imperial College London
 * 
 * The use of the code within this file and all code within files that make up
 * the software that is DeepFactors is permitted for non-commercial purposes
 * only.  The full terms and conditions that apply to the code within this file
 * are detailed within the LICENSE file and at
 * <https://www.imperial.ac.uk/dyson-robotics-lab/projects/deepfactors/deepfactors-license>
 * unless explicitly stated. By downloading this file you agree to comply with
 * these terms.
 *
 * If you wish to use any of this code for commercial purposes then please
 * email researchcontracts.engineering@imperial.ac.uk.
 *
 */
#include "deepfactors.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/BufferPyramid.hpp>

#include "decoder_network.h"
#include "display_utils.h"
#include "cu_image_proc.h"
#include "cuda_context.h"
#include "keyframe_map.h"
#include "timing.h"
#include "logutils.h"
#include "tum_io.h"

namespace df
{

/* ************************************************************************* */
void DisplayPairs(df::Map<float>::Ptr map,
                  const std::vector<std::pair<int,int>>& pairs,
                  float avg_dpt, df::CameraPyramid<float> cam_pyr,
                  df::SE3Aligner<float>::Ptr se3_aligner,
                  int levels = 1)
{
  if (pairs.empty())
    return;

  for (int lvl = 0; lvl < levels; ++lvl)
  {
    std::vector<cv::Mat> array;
    int width = map->keyframes.Get(1)->pyr_img.GetGpuLevel(lvl).width();
    int height = map->keyframes.Get(1)->pyr_img.GetGpuLevel(lvl).height();
    vc::Image2DManaged<float, vc::TargetDeviceCUDA> img2gpu(width, height);
    vc::Image2DManaged<float, vc::TargetHost> img2cpu(width, height);

    for (auto& pair : pairs)
    {
      auto kf0 = map->keyframes.Get(pair.first);
      auto kf1 = map->keyframes.Get(pair.second);
      auto relpose = df::RelativePose(kf1->pose_wk, kf0->pose_wk);


      se3_aligner->Warp(relpose, cam_pyr[lvl], kf0->pyr_img.GetGpuLevel(lvl), kf1->pyr_img.GetGpuLevel(lvl),
                        kf0->pyr_dpt.GetGpuLevel(lvl), img2gpu);
      img2cpu.copyFrom(img2gpu);

      cv::Mat dpt0 = GetOpenCV(kf0->pyr_dpt.GetCpuLevel(lvl));
      cv::Mat img0 = GetOpenCV(kf0->pyr_img.GetCpuLevel(lvl));
      cv::Mat img2 = GetOpenCV(img2cpu);
      cv::Mat prx0 = apply_colormap(df::DepthToProx(dpt0, avg_dpt));
      cv::Mat err0 = abs(img0-img2);

      array.push_back(img0);
      array.push_back(img2.clone());
      array.push_back(prx0);
      array.push_back(err0);

//      kf0->features;
//      cv::Mat drawkp0 = kf0->color_img.clone();
//      cv::Mat drawkp1 = kf1->color_img.clone();
//      cv::drawKeypoints(drawkp0, kf0->features.keypoints, drawkp0, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//      cv::drawKeypoints(drawkp1, kf0->features.keypoints, drawkp1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//      cv::drawMatches(drawkp0, kf0->features.keypoints, drawkp1, kf1->features.keypoints, matches);

//      array.push_back()
    }
//    cv::imshow("results lvl " + std::to_string(lvl), CreateMosaic(array, pairs.size(), 4));
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
DeepFactors<Scalar,CS>::DeepFactors()
  : force_keyframe_(false),
    force_frame_(false),
    bootstrapped_(false),
    curr_kf_(0) {}

/* ************************************************************************* */
template <typename Scalar, int CS>
DeepFactors<Scalar,CS>::~DeepFactors()
{
  // pop cuda context here so that tensorflow can clean up
  cuda::PopContext();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::Init(df::PinholeCamera<Scalar>& cam, const DeepFactorsOptions& opts)
{
  opts_ = opts;
  orig_cam_ = cam;

  // first, get info from network
  netcfg_ = df::LoadJsonNetworkConfig(opts.network_path);

  CHECK_EQ(netcfg_.code_size, CS) << "Trying use network with code size different than DeepFactors was compiled for!";
  CHECK(!opts.vocabulary_path.empty()) << "Required DF option vocabulary_path not provided (path to ORB vocabulary for DBoW2)";

  // initialize the gpu
  InitGpu(opts_.gpu);

  // camera that was used in network training
  df::PinholeCamera<Scalar> netcam(netcfg_.camera.fx,
                                      netcfg_.camera.fy,
                                      netcfg_.camera.u0,
                                      netcfg_.camera.v0,
                                      netcfg_.input_width,
                                      netcfg_.input_height);

  // create a pyramid from the network camera
  camera_pyr_ = df::CameraPyramid<float>(netcam, netcfg_.pyramid_levels);

  // Create tracker
  df::CameraTracker::TrackerConfig tracker_cfg;
  tracker_cfg.pyramid_levels = netcfg_.pyramid_levels;
  tracker_cfg.iterations_per_level = opts.tracking_iterations;
  tracker_cfg.huber_delta = opts.tracking_huber_delta;
  tracker_ = std::make_shared<df::CameraTracker>(camera_pyr_, tracker_cfg);

  // Create mapper
  df::MapperOptions mapper_opts;
  mapper_opts.code_prior = opts.code_prior;
  mapper_opts.pose_prior = opts.pose_prior;
  mapper_opts.predict_code = opts.predict_code;
  /* photo */
  mapper_opts.use_photometric = opts.use_photometric;
  mapper_opts.pho_iters = opts.pho_iters;
  mapper_opts.sfm_params.huber_delta = opts.huber_delta;
  mapper_opts.sfm_params.avg_dpt = netcfg_.avg_dpt;
  mapper_opts.sfm_step_threads = opts.sfm_step_threads;
  mapper_opts.sfm_step_blocks = opts.sfm_step_blocks;
  mapper_opts.sfm_eval_threads = opts.sfm_eval_threads;
  mapper_opts.sfm_eval_blocks = opts.sfm_eval_blocks;
  /* reprojection */
  mapper_opts.use_reprojection = opts.use_reprojection;
  mapper_opts.rep_nfeatures = opts.rep_nfeatures;
  mapper_opts.rep_scale_factor = opts.rep_scale_factor;
  mapper_opts.rep_nlevels = opts.rep_nlevels;
  mapper_opts.rep_max_dist = opts.rep_max_dist;
  mapper_opts.rep_huber = opts.rep_huber;
  mapper_opts.rep_iters = opts.rep_iters;
  mapper_opts.rep_sigma = opts.rep_sigma;
  /* geometric */
  mapper_opts.use_geometric = opts_.use_geometric;
  mapper_opts.geo_npoints = opts_.geo_npoints;
  mapper_opts.geo_stochastic = opts.geo_stochastic;
  mapper_opts.geo_huber = opts.geo_huber;
  /* keyframe connections */
  mapper_opts.connection_mode = opts.connection_mode;
  mapper_opts.max_back_connections = opts.max_back_connections;
  /* ISAM2 */
  mapper_opts.isam_params.enablePartialRelinearizationCheck = opts.partial_relin_check;
  mapper_opts.isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
  mapper_opts.isam_params.findUnusedFactorSlots = true;
  mapper_opts.isam_params.relinearizeSkip = opts.relinearize_skip;
  mapper_opts.isam_params.relinearizeThreshold = opts.relinearize_threshold;
  mapper_ = std::make_unique<MapperT>(mapper_opts, camera_pyr_, network_);

  // loop detector
  LOG(INFO) << "Loading vocabulary from: " << opts.vocabulary_path;
  LoopDetectorConfig detector_cfg;
  detector_cfg.tracker_cfg = tracker_cfg;
  detector_cfg.iters = tracker_cfg.iterations_per_level;
  detector_cfg.active_window = opts.loop_active_window;
  detector_cfg.max_dist = opts.loop_max_dist;
  detector_cfg.min_similarity = opts.loop_min_similarity;
  detector_cfg.max_candidates = opts.loop_max_candidates;
  loop_detector_ = std::make_unique<LoopDetectorT>(opts.vocabulary_path, mapper_->GetMap(), detector_cfg, camera_pyr_);

  // feature detector
  feature_detector_ = std::make_unique<BriskDetector>();

  // allocate buffers for the live frame
  pyr_live_img_ = std::make_shared<ImagePyrT>(netcfg_.pyramid_levels, netcfg_.input_width, netcfg_.input_height);
  pyr_live_grad_ = std::make_shared<GradPyrT>(netcfg_.pyramid_levels, netcfg_.input_width, netcfg_.input_height);

  se3aligner_ = std::make_shared<SE3AlignerT>();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::Reset()
{
  // clear all data
  loop_detector_->Reset();
  tracker_->Reset();
  mapper_->Reset();
  NotifyMapObservers();
  bootstrapped_ = false;
  tracking_lost_ = false;
  force_keyframe_ = false;
  force_frame_ = false;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::ProcessFrame(double timestamp, const cv::Mat& frame)
{
  if (!bootstrapped_)
    throw std::runtime_error("Calling ProcessFrame before system is bootstrapped!");

  // preprocess the image and upload it to gpu buffers
  tic("preprocess and upload live frame");
  cv::Mat live_col;
  Features features;
  cv::Mat live_frame = PreprocessImage(frame, live_col, features);
  UploadLiveFrame(live_frame);
  toc("preprocess and upload live frame");

  // Track frame against curr_kf or find a keyframe to relocalize and track
  SE3T new_pose_wc = tracking_lost_ ? Relocalize() : TrackFrame();

  // if tracking is still lost then return and try with the next frame
  tracking_lost_ = CheckTrackingLost(new_pose_wc);
  if (tracking_lost_)
    return;

  // We accept the new tracking as good
  pose_wc_ = new_pose_wc;
  NotifyPoseObservers();

  // loop closure on every Nth frame
  // check for loop closure
  // detect local loops
  if (opts_.loop_closure)
  {
    auto currkf = mapper_->GetMap()->keyframes.Get(curr_kf_);
    int loop_id = loop_detector_->DetectLocalLoop(*pyr_live_img_, *pyr_live_grad_, features, currkf, pose_wc_);
    if (loop_id > 0)
    {
      if (!mapper_->GetMap()->keyframes.LinkExists(curr_kf_, loop_id))
      {
        LOG(INFO) << "Local loop detected with: " << loop_id;
        LOG(INFO) << "Adding link";
        mapper_->EnqueueLink(curr_kf_, loop_id, opts_.loop_sigma, true, false, false);
      }
    }
  }

  // detect global loop
  if (opts_.loop_closure)
  {
    auto currkf = mapper_->GetMap()->keyframes.Get(curr_kf_);
    auto loop_info = loop_detector_->DetectLoop(*pyr_live_img_, *pyr_live_grad_, features, currkf, pose_wc_);
    if (loop_info.detected)
    {
      if (!mapper_->GetMap()->keyframes.LinkExists(curr_kf_, loop_info.loop_id))
      {
        // add a relative pose constraint between keyframes
        LOG(INFO) << "Global loop detected with: " << loop_info.loop_id;
        mapper_->EnqueueLink(curr_kf_, loop_info.loop_id, opts_.loop_sigma, false, true, false);

        // add to display
        loop_links.push_back({curr_kf_, loop_info.loop_id});
      }
    }
  }

  // display loop closure links
//  DisplayPairs(mapper_->GetMap(), loop_links, netcfg_.avg_dpt, camera_pyr_, se3aligner_, 1);
//  cv::waitKey(2);

  // Check whether we should add a new keyframe
  if (NewKeyframeRequired()) //  <- check tracker inliers and/or relpose
  {
    auto kf = mapper_->EnqueueKeyframe(timestamp, live_frame, live_col, pose_wc_, features);
    LOG(INFO) << "Adding keyframe " << kf->id;

    // add keyframe features to DBoW2 database
    loop_detector_->AddKeyframe(kf);

    NotifyMapObservers();
    return; // dont do any more work at this frame because it took too long to build the kf
  }

  // Check whether we should add a new frame
  if (NewFrameRequired())
  {
    VLOG(1) << "Adding new frame to keyframe " << curr_kf_;
    mapper_->EnqueueFrame(live_frame, live_col, pose_wc_, features, curr_kf_);
  }

  // collect tracking stats
  stats_.inliers = tracker_->GetInliers();
  stats_.tracker_error = tracker_->GetError();
  stats_.residual = tracker_->GetResidualImage();
  stats_.distance = df::PoseDistance(mapper_->GetMap()->keyframes.Get(curr_kf_)->pose_wk, pose_wc_);

  // do mapping
  do
  {
    try
    {
      mapper_->MappingStep();
    }
    catch(std::exception &e)
    {
      LOG(INFO) << e.what();
      LOG(ERROR) << "Mapping failed, saving debug images";

      mapper_->PrintDebugInfo();

      // add the latest error images
      DebugImages last_imgs;
      last_imgs.reprojection_errors = mapper_->DisplayReprojectionErrors();
      last_imgs.photometric_errors = mapper_->DisplayPhotometricErrors();
      debug_buffer_.push_back(last_imgs);
      throw;
    }

    // track frame against modified keyframe
//      tracker_->TrackFrame(*pyr_live_img_, *pyr_live_grad_);
//      new_pose_wc = tracker_->GetPoseEstimate();
//      pose_wc_ = new_pose_wc;
//      NotifyPoseObservers();

    if (opts_.debug)
    {
      DebugImages dbg_imgs;

      if (opts_.loop_closure)
        mapper_->DisplayLoopClosures(loop_links);

      if (opts_.use_reprojection)
        dbg_imgs.reprojection_errors = mapper_->DisplayReprojectionErrors();

      if (opts_.use_photometric)
        dbg_imgs.photometric_errors = mapper_->DisplayPhotometricErrors();

      debug_buffer_.push_back(dbg_imgs);

      if (debug_buffer_.size() > 50)
        debug_buffer_.pop_front();
    }

    stats_.relin_info = mapper_->KeyframeRelinearization();

    NotifyStatsObservers();

    if (opts_.debug)
      cv::waitKey(2);
  } while (mapper_->HasWork() && !opts_.interleave_mapping);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::BootstrapTwoFrames(double ts1, double ts2, const cv::Mat& img0, const cv::Mat& img1)
{
  CHECK_EQ(img0.type(), CV_8UC3) << "Invalid input image to DeepFactors";
  CHECK_EQ(img1.type(), CV_8UC3) << "Invalid input image to DeepFactors";

  Reset();

  Features ft0, ft1;
  cv::Mat img0_col, img1_col;
  cv::Mat img0_proc = PreprocessImage(img0, img0_col, ft0);
  cv::Mat img1_proc = PreprocessImage(img1, img1_col, ft1);
  mapper_->InitTwoFrames(img0_proc, img1_proc, img0_col, img1_col,
                         ft0, ft1, ts1, ts2);
  bootstrapped_ = true;

  curr_kf_ = mapper_->GetMap()->keyframes.LastId();
  tracker_->SetKeyframe(mapper_->GetMap()->keyframes.Get(curr_kf_));
  tracker_->Reset();
  pose_wc_ = tracker_->GetPoseEstimate();

  // add kf descriptors for loop closure
  auto kfs = mapper_->GetMap()->keyframes;
  for (auto kf : kfs)
    loop_detector_->AddKeyframe(kf.second);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::ForceKeyframe()
{
  force_keyframe_ = true;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::ForceFrame()
{
  force_frame_ = true;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
df::PinholeCamera<Scalar> DeepFactors<Scalar,CS>::GetNetworkCam()
{
  return df::PinholeCamera<Scalar>(netcfg_.camera.fx,
                                      netcfg_.camera.fy,
                                      netcfg_.camera.u0,
                                      netcfg_.camera.v0,
                                      netcfg_.input_width,
                                      netcfg_.input_height);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::BootstrapOneFrame(double timestamp, const cv::Mat& img)
{
  CHECK_EQ(img.type(), CV_8UC3) << "Invalid input image to DeepFactors";

  Reset();

  Features ft;
  cv::Mat img_col;
  cv::Mat img_proc = PreprocessImage(img, img_col, ft);
  mapper_->InitOneFrame(timestamp, img_proc, img_col, ft);
  bootstrapped_ = true;

  curr_kf_ = mapper_->GetMap()->keyframes.LastId();
  tracker_->SetKeyframe(mapper_->GetMap()->keyframes.Get(curr_kf_));
  tracker_->Reset();
  pose_wc_ = tracker_->GetPoseEstimate();

  // add kf descriptors for loop closure
  loop_detector_->AddKeyframe(mapper_->GetMap()->keyframes.Last());
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::SetOptions(DeepFactorsOptions new_opts)
{
  if (new_opts.gpu != opts_.gpu ||
      new_opts.network_path != opts_.network_path)
    LOG(FATAL) << "Online changes to GPU or network path are not allowed";

  // save the options
  opts_ = new_opts;

  // TODO: set new tracking options
  df::CameraTracker::TrackerConfig tracker_cfg;
  tracker_cfg.pyramid_levels = netcfg_.pyramid_levels;
  tracker_cfg.iterations_per_level = opts_.tracking_iterations;
  tracker_cfg.huber_delta = opts_.tracking_huber_delta;
  tracker_->SetConfig(tracker_cfg);

  // TODO: set new mapping options
//  mapper_->SetOptions(mapper_opts);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::NotifyPoseObservers()
{
  if (pose_callback_)
    pose_callback_(pose_wc_);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::NotifyMapObservers()
{
  if (map_callback_)
    map_callback_(mapper_->GetMap());
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::NotifyStatsObservers()
{
  if (stats_callback_)
    stats_callback_(stats_);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::SavePostCrashInfo(std::string dir)
{
  // display photometric factor errors
  if (opts_.use_photometric)
  {
    std::string photo_dir = dir + "/photo";
    df::CreateDirIfNotExists(photo_dir);
    mapper_->SavePhotometricDebug(photo_dir);
  }

  // display reprojection factor errors and matches
  if (opts_.use_reprojection)
  {
    std::string rep_dir = dir + "/repr";
    df::CreateDirIfNotExists(rep_dir);
    mapper_->SaveReprojectionDebug(rep_dir);
  }

  // display geometric factor errors
  if (opts_.use_geometric)
  {
    std::string geo_dir = dir + "/geom";
    df::CreateDirIfNotExists(geo_dir);
    mapper_->SavePhotometricDebug(geo_dir);
  }

  // save all keyframes
  SaveKeyframes(dir);

  // save all debug images
  int num = 0;
  for (auto it = debug_buffer_.rbegin(); it != debug_buffer_.rend(); ++it)
  {
    auto& buf = *it;
    std::string rep_name = dir + "/reprojection_errors_" + std::to_string(num) + ".png";
    std::string pho_name = dir + "/photometric_errors_" + std::to_string(num) + ".png";
    num += 1;

    if (opts_.use_reprojection)
      cv::imwrite(rep_name, buf.reprojection_errors);

    if (opts_.use_photometric)
      cv::imwrite(pho_name, buf.photometric_errors);
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::SaveKeyframes(std::string dir)
{
  // create directory
  std::string keyframe_dir = dir + "/keyframes";
  df::CreateDirIfNotExists(keyframe_dir);

  // save keyframes
  for (auto& kf : GetMap()->keyframes)
  {
    std::string timestamp_str = std::to_string(kf.second->timestamp);
    std::string rgb_filename = timestamp_str + "_rgb.png";
    std::string dpt_filename = timestamp_str + "_dpt.png";

    auto dpt_lvl0 = kf.second->pyr_dpt.GetCpu()->operator[](0);
    vc::Image2DView<ScalarT, vc::TargetHost> dpt_view(dpt_lvl0);

    cv::Mat dpt_out;
    dpt_view.getOpenCV().convertTo(dpt_out, CV_16UC1, 5000.0f, 0.f); // convert to visible format by multiplying by 5000 and not 1000
    cv::imwrite(keyframe_dir + "/" + rgb_filename, kf.second->color_img);
    cv::imwrite(keyframe_dir + "/" + dpt_filename, dpt_out);
  }

  // save intrinsics
  auto cam = GetNetworkCam();
  {
    std::ofstream intr_file(keyframe_dir + "/intrinsics.txt");
    intr_file << cam.fx() << " " << cam.fy() << " " << cam.u0() << " "
              << cam.v0() << " " << cam.width() << " " << cam.height();
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::SaveResults(std::string dir)
{
  // save trajectory file
  {
    std::ofstream f(dir + "/trajectory.txt");
    for (auto& kf : GetMap()->keyframes)
    {
      // save keyframe pose to trajectory file
      auto pose_wk = kf.second->pose_wk;
      Eigen::Quaternionf q = pose_wk.so3().unit_quaternion();
      Eigen::Vector3f t = pose_wk.translation();
      TumPose pose{kf.second->timestamp, q, t};
      f << pose << std::endl;
    }
  }

  // save keyframes
  SaveKeyframes(dir);

  LOG(INFO) << "Saved results to " << dir;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::InitGpu(std::size_t device_id)
{
  // explicitly create our own cuda context on selected gpu
  // create a new context that we will use for our CUDA and bind to it
  // we will have to remove it for network runs
  cuda::Init();
  cuda::CreateAndBindContext(device_id);

  // initialize the network to start a tensorflow
  // session, which will create its own context
  {
    auto scoped_pop = cuda::ScopedContextPop();
    network_ = std::make_shared<df::DecoderNetwork>(netcfg_);
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DeepFactors<Scalar,CS>::UploadLiveFrame(const cv::Mat& frame)
{
  for(std::size_t i = 0 ; i < netcfg_.pyramid_levels; ++i)
  {
    if (i == 0)
    {
      vc::Image2DView<float, vc::TargetHost> tmp(frame);
      (*pyr_live_img_)[0].copyFrom(tmp);
      continue;
    }

    df::GaussianBlurDown((*pyr_live_img_)[i-1], (*pyr_live_img_)[i]);
    df::SobelGradients((*pyr_live_img_)[i], (*pyr_live_grad_)[i]);
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
cv::Mat DeepFactors<Scalar,CS>::PreprocessImage(const cv::Mat& frame, cv::Mat& out_color, Features& features)
{
  CHECK_EQ(frame.type(), CV_8UC3);

  // change camera intrinsics to image size
  orig_cam_.ResizeViewport(frame.cols, frame.rows);

  // compute maps for adjusting the image focal length in PreprocessFrame
  // TODO: precompute in Init (needs camera width and height in init though)
  const cv::Mat inK = (cv::Mat1d(3,3) << orig_cam_.fx(), 0, orig_cam_.u0(), 0, orig_cam_.fy(), orig_cam_.v0(), 0, 0, 1);
  const cv::Mat outK = (cv::Mat1d(3,3) << netcfg_.camera.fx, 0, netcfg_.camera.u0, 0, netcfg_.camera.fy, netcfg_.camera.v0, 0, 0, 1);
  cv::Size network_size(netcfg_.input_width, netcfg_.input_height);
  cv::initUndistortRectifyMap(inK, cv::Mat{}, cv::Mat{}, outK, network_size, CV_32FC1, map1_, map2_);

  // run remapping to generate an image of proper focal length
  cv::Mat outframe;
  cv::remap(frame, outframe, map1_, map2_, cv::INTER_LINEAR);
  out_color = outframe;

  // convert to grayscale
  cv::Mat outframe_gray;
  cv::cvtColor(outframe, outframe_gray, cv::COLOR_RGB2GRAY);

  // convert to floats in range 0-1
  cv::Mat outframe_float;
  outframe_gray.convertTo(outframe_float, CV_32FC1, 1/255.0);

  if (opts_.normalize_image)
  {
    cv::Scalar mean, stdev;
    cv::meanStdDev(outframe_float, mean, stdev);
    outframe_float = (outframe_float - mean[0]) / stdev[0];
  }

  // detect features
  // TODO: make sure we're passing the right image format
  features = feature_detector_->DetectAndCompute(outframe_gray);

  if (opts_.debug)
  {
    cv::Mat img = outframe_gray.clone();
    cv::drawKeypoints(img, features.keypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("features", img);
  }

  return outframe_float;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename DeepFactors<Scalar,CS>::SE3T
DeepFactors<Scalar,CS>::TrackFrame()
{
  SE3T new_pose_wc;

  // find the active keyframe to track against
  tic("Finding keyframe to track");
  auto newkfid = SelectKeyframe();
  if (newkfid != curr_kf_)
  {
    VLOG(1) << "Switching to keyframe " << newkfid;
    prev_kf_ = curr_kf_;
    curr_kf_ = newkfid;
    tracker_->SetKeyframe(mapper_->GetMap()->keyframes.Get(curr_kf_));
  }
  toc("Finding keyframe to track");

  // track live frame against kf
  tic("Tracking frame");
  tracker_->TrackFrame(*pyr_live_img_, *pyr_live_grad_);
  new_pose_wc = tracker_->GetPoseEstimate();
  toc("Tracking frame");

  return new_pose_wc;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename DeepFactors<Scalar,CS>::SE3T
DeepFactors<Scalar,CS>::Relocalize()
{
  SE3T new_pose_wc;

  auto kfmap = mapper_->GetMap();
  Scalar best_error = std::numeric_limits<Scalar>::infinity();
  KeyframeId best_id = 1;
  SE3T best_pose = kfmap->keyframes.Get(best_id)->pose_wk;
  for (const auto& id : kfmap->keyframes.Ids())
  {
    auto kf = kfmap->keyframes.Get(id);
    tracker_->SetKeyframe(kf);
    tracker_->Reset();
    tracker_->TrackFrame(*pyr_live_img_, *pyr_live_grad_);
    auto error = tracker_->GetError();
    if (error < best_error)
    {
      best_error = error;
      best_id = id;
      best_pose = tracker_->GetPoseEstimate();
    }
  }

  // reconfigure everything to be ok at next frame
  curr_kf_ = best_id;
  new_pose_wc = best_pose;
  tracker_->SetPose(best_pose);
  tracker_->SetKeyframe(kfmap->keyframes.Get(best_id));

  return new_pose_wc;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
bool DeepFactors<Scalar,CS>::NewKeyframeRequired()
{
  if (force_keyframe_)
  {
    force_keyframe_ = false;
    return true;
  }

  auto inliers = tracker_->GetInliers();
  auto curr_kf = mapper_->GetMap()->keyframes.Get(curr_kf_);
  auto distance = df::PoseDistance(curr_kf->pose_wk, pose_wc_);

  /* possibly check tracker inliers and/or relpose */
  switch(opts_.keyframe_mode)
  {
  case DeepFactorsOptions::AUTO:
  {
    bool inlier_bad = inliers < opts_.inlier_threshold;
    bool dist_far = distance > opts_.dist_threshold;

    return inlier_bad || dist_far;
  }
  case DeepFactorsOptions::AUTO_COMBINED:
  {
    bool inlier_bad = inliers < opts_.inlier_threshold;
    float rot_dist = (curr_kf->pose_wk.so3() * pose_wc_.so3().inverse()).log().norm();
    float delta = distance * 5 + rot_dist * 3;
    return delta > opts_.combined_threshold || inlier_bad;
  }
  case DeepFactorsOptions::NEVER:
    return false;
  }
  return false;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
bool DeepFactors<Scalar,CS>::NewFrameRequired()
{
  if (force_frame_)
  {
    force_frame_ = false;
    return true;
  }

  if (opts_.keyframe_mode == DeepFactorsOptions::NEVER)
    return false;

  Scalar min_frame_dist = opts_.frame_dist_threshold;
  auto curr_kf = mapper_->GetMap()->keyframes.Get(curr_kf_);
  Scalar kf_dist = df::PoseDistance(curr_kf->pose_wk, pose_wc_, 1.0f, 0.0f);
  bool far_from_keyframe = kf_dist > min_frame_dist;

  bool far_from_frames = true;
  for (auto fr : mapper_->GetMap()->frames)
  {
    Scalar fr_dist = df::PoseDistance(fr.second->pose_wk, pose_wc_, 1.0f, 0.0f);
    if (fr_dist < min_frame_dist)
      far_from_frames = false;
  }

  return far_from_keyframe && far_from_frames && !mapper_->HasWork();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename DeepFactors<Scalar,CS>::KeyframeId DeepFactors<Scalar,CS>::SelectKeyframe()
{
  if (mapper_->NumKeyframes() == 0)
    LOG(FATAL) << "KeyframeMap is empty (should not happen here)";

  KeyframeId kfid = 0;
  if (opts_.tracking_mode == DeepFactorsOptions::LAST)
  {
    kfid = mapper_->GetMap()->keyframes.LastId();
  }
  else if (opts_.tracking_mode == DeepFactorsOptions::CLOSEST)
  {
    // find closest keyframe
    auto kfmap = mapper_->GetMap();
    Scalar closest_dist = std::numeric_limits<Scalar>::infinity();
    for (auto id : kfmap->keyframes.Ids())
    {
      Scalar dist = PoseDistance(kfmap->keyframes.Get(id)->pose_wk, pose_wc_);
      if (dist < closest_dist)
      {
        closest_dist = dist;
        kfid = id;
      }
    }
  }
  else if (opts_.tracking_mode == DeepFactorsOptions::FIRST)
  {
    kfid = mapper_->GetMap()->keyframes.Ids()[0];
  }
  else
  {
    LOG(FATAL) << "Unhandled tracking mode";
  }

  return kfid;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
bool DeepFactors<Scalar,CS>::CheckTrackingLost(const SE3T& pose_wc)
{
  // tracking error check
  bool error_too_big = tracker_->GetError() > opts_.tracking_error_threshold;

  // pose jump check
  auto pose_wk = mapper_->GetMap()->keyframes.Get(curr_kf_)->pose_wk;
  Scalar distance = PoseDistance(pose_wk, pose_wc);
  bool kf_too_far = distance > opts_.tracking_dist_threshold;

  bool tracking_lost = error_too_big || kf_too_far;

  if (tracking_lost && !tracking_lost_)
  {
    LOG(INFO) << "Tracking Lost";
    LOG(INFO) << "distance from kf: " << distance;
    LOG(INFO) << "tracking error: " << tracker_->GetError();
    LOG(INFO) << "kf_too_far: " << kf_too_far;
    LOG(INFO) << "error_too_big: " << error_too_big;
    LOG(INFO) << "pose_wc: " << pose_wc.log().transpose();
  }
  else if (!tracking_lost && tracking_lost_)
  {
    LOG(INFO) << "Tracking OK";
    LOG(INFO) << "Relocalized to keyframe " << curr_kf_;
  }
  return tracking_lost;
}

/* ************************************************************************* */

// explicit instantiation
template class DeepFactors<float,DF_CODE_SIZE>;

} // namespace df
