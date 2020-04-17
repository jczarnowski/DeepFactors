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
#include <VisionCore/Image/BufferOps.hpp>
#include <opencv2/features2d.hpp>
#include <glog/logging.h>

//#define GTSAM_ENABLE_DEBUG
//#include <gtsam/base/debug.h>

#include "mapper.h"
#include "cuda_context.h"
#include "display_utils.h"
#include "cu_image_proc.h"
#include "timing.h"

#ifdef GTSAM_USE_TBB
#error "GTSAM has been compiled with TBB support, but we don't support that"
#endif

namespace df
{

/* ************************************************************************* */
void DisplayKeyframes(df::Map<float>::Ptr map,
                      const std::vector<cv::Mat>& gtdepth,
                      float avg_dpt)
{
  bool has_gtdpt = (gtdepth.size() == map->NumKeyframes());
  if (map->NumKeyframes() == 0)
    return;

//  int pyrlevels = map->keyframes.Get(1)->pyr_dpt.Levels();
  int pyrlevels = 1;
  for (int lvl = 0; lvl < pyrlevels; ++lvl)
  {
    std::vector<cv::Mat> array;
    int start = map->keyframes.Ids().size()-1;
    int end = std::max((int)map->keyframes.Ids().size()-5, 0);
    for (int i = start; i >= end; --i)
    {
      auto kf = map->keyframes.Get(map->keyframes.Ids()[i]);
      cv::Mat img = GetOpenCV(kf->pyr_img.GetCpuLevel(lvl));
      cv::Mat rec = df::DepthToProx(GetOpenCV(kf->pyr_dpt.GetCpuLevel(lvl)), avg_dpt);
      cv::Mat vld = GetOpenCV(kf->pyr_vld.GetCpuLevel(lvl));
      cv::Mat std = GetOpenCV(kf->pyr_stdev.GetCpuLevel(lvl));
      cv::Mat std_exp;
      cv::exp(std,std_exp);

      array.push_back(img);
      array.push_back(apply_colormap(rec));
      array.push_back(5*sqrt(2)*std_exp);
      array.push_back(vld);

      if (has_gtdpt)
      {
        cv::Mat gt = df::DepthToProx(gtdepth[i], avg_dpt);
        cv::Mat err = abs(gt-rec);
        array.push_back(apply_colormap(gt));
        array.push_back(err);
      }
    }

    int ncols = has_gtdpt ? 6 : 4;
    cv::imshow("keyframes lvl " + std::to_string(lvl),
               CreateMosaic(array, array.size() / ncols, ncols));
  }
}

/* ************************************************************************* */
cv::Mat DisplayPairs(df::Map<float>::Ptr map,
                     const std::vector<std::pair<std::size_t,std::size_t>>& pairs,
                     float avg_dpt, df::CameraPyramid<float> cam_pyr,
                     df::SE3Aligner<float>::Ptr se3_aligner)
{
  if (pairs.empty())
    return cv::Mat{};

//  int pyrlevels = map->keyframes.Last()->pyr_dpt.Levels();
  int pyrlevels = 1;
  cv::Mat mosaic;
  for (int lvl = 0; lvl < pyrlevels; ++lvl)
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
    }
    mosaic = CreateMosaic(array, pairs.size(), 4);
    cv::imshow("warping lvl " + std::to_string(lvl), mosaic);
  }
  return mosaic;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
Mapper<Scalar,CS>::Mapper(const MapperOptions& opts,
                          const df::CameraPyramid<Scalar>& cam_pyr,
                          df::DecoderNetwork::Ptr network)
    : network_(network), cam_pyr_(cam_pyr), opts_(opts)
{
  // create sfmaligner
  df::SfmAlignerParams alignerparams;
  alignerparams.sfmparams = opts.sfm_params;
  alignerparams.step_threads = opts.sfm_step_threads;
  alignerparams.step_blocks = opts.sfm_step_blocks;
  alignerparams.eval_threads = opts.sfm_eval_threads;
  alignerparams.eval_blocks = opts.sfm_eval_blocks;
  aligner_ = std::make_shared<df::SfmAligner<float,CS>>(alignerparams);

  // create keyframe map
  map_ = std::make_shared<MapT>();

  // initialize the isam graph
  opts_.isam_params.enableDetailedResults = true;  // for printing out relinearized vars
  opts_.isam_params.print("ISAM2 params");
  isam_graph_ = std::make_unique<gtsam::ISAM2>(opts_.isam_params);

  se3aligner_ = std::make_shared<SE3Aligner<Scalar>>();

//  SETDEBUG("ISAM2 update", true);
//  SETDEBUG("ISAM2 update verbose", true);
//  SETDEBUG("ISAM2 recalculate", true);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::InitTwoFrames(const cv::Mat& img0, const cv::Mat& img1,
                                      const cv::Mat& col0, const cv::Mat& col1,
                                      const Features& ft0, const Features& ft1,
                                      double ts1, double ts2)
{
  Reset();

  auto kf0 = BuildKeyframe(ts1, img0, col0, SE3T{}, ft0);
  auto kf1 = BuildKeyframe(ts2, img1, col1, SE3T{}, ft1);
  map_->AddKeyframe(kf0);
  map_->AddKeyframe(kf1);

  work_manager_.AddWork<work::InitVariables<Scalar,CS>>(kf0, opts_.code_prior, opts_.pose_prior);
  work_manager_.AddWork<work::InitVariables<Scalar,CS>>(kf1, opts_.code_prior);

  work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(kf0, kf1, opts_.pho_iters,
                                                        cam_pyr_, aligner_);
  work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(kf1, kf0, opts_.pho_iters,
                                                        cam_pyr_, aligner_);

  // optimize until there is no more work
  // this means either convergence or running out of iters
  while (!work_manager_.Empty())
    MappingStep();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::InitOneFrame(double timestamp, const cv::Mat& img, const cv::Mat& col,
                                     const Features& ft)
{
  Reset();

  auto kf = BuildKeyframe(timestamp, img, col, SE3T{}, ft);
  map_->AddKeyframe(kf);

  work_manager_.AddWork<work::InitVariables<Scalar,CS>>(kf, opts_.code_prior,
                                                        opts_.pose_prior);

  // do a single mapping step to initialize variables
  // warning: it might mean isam will take a step towards
  // the priors (but in my case they are the same as init)
  MappingStep();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::SetOptions(const MapperOptions& new_opts)
{
  // TODO: set options of subsystems
//  df::SfmAlignerParams alignerparams;
//  alignerparams.sfmparams = opts.sfm_params;
//  alignerparams.
}

/* ************************************************************************* */
template<typename Scalar, int CS>
std::vector<typename Mapper<Scalar,CS>::FrameId> Mapper<Scalar,CS>::NonmarginalizedFrames()
{
  std::vector<FrameId> nonmrg_frames;
  for (auto& id : map_->frames.Ids())
    if (!map_->frames.Get(id)->marginalized)
      nonmrg_frames.push_back(id);
  return nonmrg_frames;
}

/* ************************************************************************* */
template<typename Scalar, int CS>
std::map<typename Mapper<Scalar,CS>::KeyframeId, bool> Mapper<Scalar,CS>::KeyframeRelinearization()
{
  std::map<KeyframeId, bool> info;
  auto& var_status = (*isam_res_.detail).variableStatus;
  for (auto& id : map_->keyframes.Ids())
  {
    auto pkey = PoseKey(id);
    auto ckey = CodeKey(id);
    info[id] = var_status[pkey].isRelinearized || var_status[ckey].isRelinearized;
  }
  return info;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::EnqueueFrame(const cv::Mat& img, const cv::Mat& col,
                                     const SE3T& pose_init, const Features& ft,
                                     FrameId kf_id)
{
  // marginalize previous frame now
  MarginalizeFrames(NonmarginalizedFrames());

  // add new frame to the map
  auto fr = BuildFrame(img, col, pose_init, ft);
  map_->AddFrame(fr);
  auto id = fr->id;

  auto kf = map_->keyframes.Get(kf_id);

  work_manager_.AddWork<work::InitVariables<Scalar,CS>>(fr);
  work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(kf, fr, opts_.pho_iters,
                                                        cam_pyr_, aligner_);

  map_->frames.AddLink(fr->id, kf_id);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename Mapper<Scalar,CS>::KeyframePtr
Mapper<Scalar,CS>::EnqueueKeyframe(double timestamp,
                                   const cv::Mat& img, const cv::Mat& col,
                                   const SE3T& pose_init,
                                   const Features& ft)
{
  auto conns = BuildBackConnections();
  return EnqueueKeyframe(timestamp, img, col, pose_init, ft, conns);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename Mapper<Scalar,CS>::KeyframePtr
Mapper<Scalar,CS>::EnqueueKeyframe(double timestamp,
                                   const cv::Mat& img,
                                   const cv::Mat& col,
                                   const SE3T& pose_init,
                                   const Features& ft,
                                   const std::vector<FrameId>& conns)
{
  // add to map
  auto kf = BuildKeyframe(timestamp, img, col, pose_init, ft);
  map_->AddKeyframe(kf);

  // marginalize frames
  MarginalizeFrames(NonmarginalizedFrames());

  work_manager_.AddWork<work::InitVariables<Scalar,CS>>(kf, opts_.code_prior);
  for (auto& id : conns)
  {
    auto back_kf = map_->keyframes.Get(id);

    work::WorkManager::WorkPtr ptr;

    // add photometric error both ways
    if (opts_.use_photometric)
    {
      work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(kf, back_kf, opts_.pho_iters,
                                                            cam_pyr_, aligner_);
      ptr = work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(back_kf, kf, opts_.pho_iters,
                                                                  cam_pyr_, aligner_, true);
    }

    // add reprojection error both ways
    if (opts_.use_reprojection)
    {
      work_manager_.AddWork<work::OptimizeRep<Scalar,CS>>(kf, back_kf, opts_.rep_iters, cam_pyr_[0],
                                                          opts_.rep_max_dist, opts_.rep_huber,
                                                          opts_.rep_sigma, opts_.rep_ransac_maxiters,
                                                          opts_.rep_ransac_threshold);
      ptr = work_manager_.AddWork<work::OptimizeRep<Scalar,CS>>(back_kf, kf, opts_.rep_iters, cam_pyr_[0],
                                                                opts_.rep_max_dist, opts_.rep_huber,
                                                                opts_.rep_sigma, opts_.rep_ransac_maxiters,
                                                                opts_.rep_ransac_threshold);
    }

    // add geometric error to be optimized after one of the photometric errors
    if (opts_.use_geometric)
    {
      auto geo_err = std::make_shared<work::OptimizeGeo<Scalar,CS>>(kf, back_kf, opts_.geo_iters, cam_pyr_[0], opts_.geo_npoints,
                                                                    opts_.geo_huber, opts_.geo_stochastic);

      if (ptr)
        ptr->AddChild(geo_err);
      else
        work_manager_.AddWork(geo_err);
    }

    map_->keyframes.AddLink(kf->id, id);
    map_->keyframes.AddLink(id, kf->id);
  }

  return kf;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::EnqueueLink(FrameId id0, FrameId id1, Scalar rep_sigma, bool pho, bool rep, bool geo)
{
  auto kf0 = map_->keyframes.Get(id0);
  auto kf1 = map_->keyframes.Get(id1);

  work::WorkManager::WorkPtr ptr;

  // marginalize frames
  MarginalizeFrames(NonmarginalizedFrames());

  // add photometric error both ways
  if (pho)
  {
    work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(kf0, kf1, opts_.pho_iters,
                                                          cam_pyr_, aligner_);
    ptr = work_manager_.AddWork<work::OptimizePhoto<Scalar,CS>>(kf1, kf0, opts_.pho_iters,
                                                                cam_pyr_, aligner_, true);
  }

  // add reprojection error both ways
  if (rep)
  {
    ptr = work_manager_.AddWork<work::OptimizeRep<Scalar,CS>>(kf0, kf1, opts_.rep_iters, cam_pyr_[0],
                                                        opts_.rep_max_dist, opts_.rep_huber, rep_sigma,
                                                        opts_.rep_ransac_maxiters, opts_.rep_ransac_threshold);
    ptr = work_manager_.AddWork<work::OptimizeRep<Scalar,CS>>(kf1, kf0, opts_.rep_iters, cam_pyr_[0],
                                                        opts_.rep_max_dist, opts_.rep_huber, rep_sigma,
                                                        opts_.rep_ransac_maxiters, opts_.rep_ransac_threshold);
  }

  // add geometric error to be optimized after one of the photometric errors
  if (geo)
  {
    auto geo_err = std::make_shared<work::OptimizeGeo<Scalar,CS>>(kf0, kf1, opts_.geo_iters, cam_pyr_[0], opts_.geo_npoints,
                                                                  opts_.geo_huber, opts_.geo_stochastic);

    if (ptr)
      ptr->AddChild(geo_err);
    else
      work_manager_.AddWork(geo_err);
  }

  map_->keyframes.AddLink(id0, id1);
  map_->keyframes.AddLink(id1, id0);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::MarginalizeFrames(const std::vector<Mapper::FrameId> &frames)
{
//  LOG(INFO) << "Marginalizing past frames";
  if (frames.empty())
    return;

  // marginalize all frames
  gtsam::FastList<gtsam::Key> mrg_keys;
  for (const auto& id : frames)
  {
    auto fr = map_->frames.Get(id);
    if (!fr->marginalized)
    {
      mrg_keys.push_back(AuxPoseKey(id));
      fr->marginalized = true;

      // remove involved work
      auto lambda = [&] (work::Work::Ptr w) {
        auto ow = std::dynamic_pointer_cast<work::OptimizeWork<Scalar>>(w);
        if (!ow) return false;
        VLOG(1) << ow->Name();
        VLOG(1) << "Involves " << id << ": " << ow->Involves(fr);
        return ow && ow->Involves(fr);
      };

      VLOG(1) << "Removing work involved in frame " << id;
      work_manager_.Erase(lambda);
    }
  }

  if (VLOG_IS_ON(1))
  {
    VLOG(1) << "marginalizing keys: ";
    gtsam::PrintKeyList(mrg_keys);
  }

  // run marginalization
  tic("marginalize frames");
  isam_graph_->marginalizeLeaves(mrg_keys);
  toc("marginalize frames");
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                                    gtsam::FactorIndices& remove_indices,
                                    gtsam::Values& var_init)
{
  work_manager_.Bookkeeping(new_factors, remove_indices, var_init);
  work_manager_.Update();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::MappingStep()
{
  // dont do steps if there are no keyframes that were not yet optimized
  if (work_manager_.Empty())
    return;

  // display keyframes before mapping step
//  std::vector<cv::Mat> gtdpt;
//  DisplayKeyframes(map_, gtdpt, opts_.sfm_params.avg_dpt);
//  cv::waitKey(0);

  // determine new factors to add and initialization of new variables
  gtsam::Values var_init;
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::FactorIndices remove_indices;
  Bookkeeping(new_factors, remove_indices, var_init);

//  PrintFactors();

//  new_factors.print("new_factors\n");
//  var_init.print("new_vars\n");

//  std::cout << "remove indices:";
//  for (auto id : remove_indices)
//    std::cout << " " << id;
//  std::cout << std::endl;

  // do step
  gtsam::FastMap<gtsam::Key,int> constrainedKeys;

  auto it = std::find_if(map_->frames.begin(), map_->frames.end(), [] (auto& fr) -> bool { return !fr.second->marginalized; } );
  bool has_nonmarginalized_frames = it != map_->frames.end();


  int i = 0;

  for (auto& fr : map_->keyframes)
  {
    if (fr.second->id != map_->keyframes.LastId())
    {
      auto id = fr.second->id;
      constrainedKeys[PoseKey(id)] = i;
      constrainedKeys[CodeKey(id)] = i;
    }
  }
  i++;

  if (has_nonmarginalized_frames)
  {
    for (auto& fr : map_->frames)
    {
      if (fr.second->marginalized) continue;
      auto id = fr.second->id;
      constrainedKeys[AuxPoseKey(id)] = i;
    }
    i++;
  }

  auto lastid = map_->keyframes.LastId();
  constrainedKeys[PoseKey(lastid)] = i;
  constrainedKeys[CodeKey(lastid)] = i;

//  for (auto& kv : constrainedKeys)
//  {
//    LOG(INFO) << "constraint on " << gtsam::DefaultKeyFormatter(kv.first) << " to " << kv.second;
//  }

  tic("isam graph update");
  isam_res_ = isam_graph_->update(new_factors, var_init, remove_indices,
                                  constrainedKeys, boost::none, boost::none, true);
  toc("isam graph update");

  VLOG(1) << "factorsRecalculated: " << isam_res_.factorsRecalculated;
  VLOG(1) << "variablesReeliminated: " << isam_res_.variablesReeliminated;
  VLOG(1) << "variablesRelinearized: " << isam_res_.variablesRelinearized;

  // distribute new factor indices to work items that they originated from
  work_manager_.DistributeIndices(isam_res_.newFactorsIndices);

  tic("calculate estimate and map update");
  estimate_ = isam_graph_->calculateEstimate();
  UpdateMap(estimate_, isam_graph_->getDelta());
  toc("calculate estimate and map update");

  // if its not relinearizing its time to move to next level or end
  if (isam_res_.variablesRelinearized == 0)
  {
    VLOG(1) << "Not relinarizing any variables -- move to next level";
    work_manager_.SignalNoRelinearize();
  }

  // display all factors if we've added any this step
  if (VLOG_IS_ON(2) && !new_factors.empty())
  {
    SaveGraphs("isam_graph");
    PrintFactors();
  }

  if (VLOG_IS_ON(1))
    work_manager_.PrintWork();

//  DisplayPairs(*kfmap_, kfmap_->GetLinks(), 2, cam_pyr_, se3aligner_);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::Reset()
{
  map_->Clear();
  estimate_.clear();
  isam_graph_ = std::make_unique<gtsam::ISAM2>(opts_.isam_params);
  work_manager_.Clear();

  new_match_imgs_ = false;
  match_imgs_.clear();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::SaveGraphs(const std::string& prefix)
{
  // Prune the unsafe graph of null pointers to factors
  // If we dont do this, functions like NonlinearFactorGraph::SaveGraph crash
  auto graph = isam_graph_->getFactorsUnsafe().clone();
  auto it = std::remove_if(graph.begin(), graph.end(), [](auto& f) -> bool { return f == nullptr; });
  graph.erase(it, graph.end());

  // save graph
  std::string filename = prefix + "_factors.dot";
  std::ofstream file(filename);
  if (!file.is_open())
    LOG(FATAL) << "Failed to open file for writing: " << filename;
  graph.saveGraph(file);
  file.close();

  // try to save ISAM2 bayes tree
  isam_graph_->saveGraph(prefix + "_tree.dot");
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::PrintFactors()
{
  auto graph = isam_graph_->getFactorsUnsafe().clone();
  for (auto factor : graph)
  {
    if (factor)
    {
      auto ptr = factor.get();

      // print all types of factors that we use
      auto photo = dynamic_cast<PhotometricFactor<Scalar,CS>*>(ptr);
      if (photo)
        LOG(INFO) << photo->Name();
      auto geo = dynamic_cast<SparseGeometricFactor<Scalar,CS>*>(ptr);
      if (geo)
        LOG(INFO) << geo->Name();
      auto rep = dynamic_cast<ReprojectionFactor<Scalar,CS>*>(ptr);
      if (rep)
        LOG(INFO) << rep->Name();
      auto prior = dynamic_cast<gtsam::PriorFactor<gtsam::Vector>*>(ptr);
      if (prior)
        LOG(INFO) << "CodePrior on " << gtsam::DefaultKeyFormatter(prior->key());
      auto priorpose = dynamic_cast<gtsam::PriorFactor<Sophus::SE3f>*>(ptr);
      if (priorpose)
        LOG(INFO) << "PosePrior on " << gtsam::DefaultKeyFormatter(priorpose->key());
      auto lcf = dynamic_cast<gtsam::LinearContainerFactor*>(ptr);
      if (lcf)
      {
        LOG(INFO) << "LinearContainerFactor on ";
        gtsam::PrintKeyVector(lcf->keys());
      }
    }
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::PrintDebugInfo()
{
  PrintFactors();
  work_manager_.PrintWork();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::DisplayMatches()
{
  if (!match_imgs_.empty() || new_match_imgs_)
  {
    cv::imshow("matches", CreateMosaic(match_imgs_));
    new_match_imgs_ = false;
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
cv::Mat Mapper<Scalar,CS>::DisplayReprojectionErrors()
{
  int N = 3;
  auto graph = isam_graph_->getFactorsUnsafe().clone();
  auto ids = map_->keyframes.Ids();
  std::vector<cv::Mat> array;

  for (int i = ids.size()-1; i >= std::max((int)ids.size()-N,0); --i)
  {
    // for each connection find the reprojection error
    auto kf = map_->keyframes.Get(ids[i]);
    std::vector<FrameId> conns = map_->keyframes.GetConnections(kf->id, true);
    for (auto& c : conns)
    {
      if (c > kf->id) // only do back connections
        continue;

      for (auto it = std::reverse_iterator(graph.end());
          it != std::reverse_iterator(graph.begin());
          ++it)
      {
         auto factor = *it;
         if (!factor)
           continue;
         auto rep = dynamic_cast<ReprojectionFactor<Scalar,CS>*>(factor.get());
         if (!rep)
           continue;

         bool front_conn = rep->Keyframe()->id == ids[i] && rep->Frame()->id == c;
         bool back_conn = rep->Keyframe()->id == c && rep->Frame()->id == ids[i];
         if (front_conn || back_conn)
         {
           cv::Mat matches = rep->DrawMatches();
           cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols/2));
           cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols/2, matches.cols));

           cv::Mat rec = df::DepthToProx(GetOpenCV(kf->pyr_dpt.GetCpuLevel(0)), opts_.sfm_params.avg_dpt);
           array.push_back(m1);
           array.push_back(m2);
           array.push_back(apply_colormap(rec));
           array.push_back(rep->ErrorImage());
         }
       }
    }
  }

  if (array.size())
  {
    cv::Mat mosaic = CreateMosaic(array, array.size() / 4, 4);
    cv::imshow("reprojection errors", mosaic);
    return mosaic;
  }

//  LOG(INFO) << "Total reprojection error = " << total_err;
  return cv::Mat{};
}

/* ************************************************************************* */
template <typename Scalar, int CS>
cv::Mat Mapper<Scalar,CS>::DisplayPhotometricErrors()
{
  int N = 3;
  typedef typename MapT::FrameGraphT::LinkT LinkT;

  std::vector<LinkT> links;
  auto ids = map_->keyframes.Ids();
  for (int i = ids.size()-1; i >= std::max((int)ids.size()-N,0); --i)
  {
    std::vector<FrameId> conns = map_->keyframes.GetConnections(ids[i], true);
    for (auto& c : conns)
    {
      if (c < ids[i]) // show only back connections
        links.push_back({ids[i], c});
    }
  }
  return DisplayPairs(map_, links, opts_.sfm_params.avg_dpt, cam_pyr_, se3aligner_);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::DisplayLoopClosures(std::vector<std::pair<int, int>>& link)
{
  std::vector<cv::Mat> array;
  auto graph = isam_graph_->getFactorsUnsafe().clone();
  int N = 5;
  for (auto it = std::reverse_iterator(graph.end());
       it != std::reverse_iterator(graph.begin());
       ++it)
  {
    auto factor = *it;
    if (!factor)
      continue;

    auto ptr = factor.get();
    auto rep = dynamic_cast<ReprojectionFactor<Scalar,CS>*>(ptr);
    if (rep )
    {
      // check if it involves our crap
      bool is_our = false;
      int kf_id = (int)rep->Keyframe()->id;
      int fr_id = (int)rep->Frame()->id;
      for (auto& l : link)
      {
        if ((kf_id == l.first && fr_id == l.second) ||
            (kf_id == l.second && fr_id == l.first))
          is_our = true;
      }

      if (is_our)
      {
        cv::Mat matches = rep->DrawMatches();
        cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols/2));
        cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols/2, matches.cols));
        array.push_back(m1);
        array.push_back(m2);
        array.push_back(rep->ErrorImage());
        N--;
      }

      if (N < 0)
        break;
    }
  }

  if (array.size())
    cv::imshow("loop closures", CreateMosaic(array, array.size()/3, 3));
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::SavePhotometricDebug(std::string save_dir)
{
  typedef typename MapT::FrameGraphT::LinkT LinkT;
  auto ids = map_->keyframes.Ids();
  for (int i = ids.size()-1; i >= 0; --i)
  {
    std::vector<FrameId> conns = map_->keyframes.GetConnections(ids[i], true);
    for (auto& c : conns)
    {
      if (c < ids[i]) // show only back connections
      {
        std::vector<LinkT> links{{ids[i], c}};
        cv::Mat img = DisplayPairs(map_, links, opts_.sfm_params.avg_dpt, cam_pyr_, se3aligner_);
        std::string outname = save_dir + "/photo" + std::to_string(ids[i]) + "-" + std::to_string(c) + ".png";
        cv::imwrite(outname, img);
      }
    }
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::SaveReprojectionDebug(std::string rep_dir)
{
  auto graph = isam_graph_->getFactorsUnsafe().clone();
  auto ids = map_->keyframes.Ids();

  for (int i = ids.size()-1; i >= 0; --i)
  {
    // for each connection find the reprojection error
    auto kf = map_->keyframes.Get(ids[i]);
    std::vector<FrameId> conns = map_->keyframes.GetConnections(kf->id, true);
    for (auto& c : conns)
    {
      if (c > kf->id) // only do back connections
        continue;

      for (auto it = std::reverse_iterator(graph.end());
          it != std::reverse_iterator(graph.begin());
          ++it)
      {
         auto factor = *it;
         if (!factor)
           continue;
         auto rep = dynamic_cast<ReprojectionFactor<Scalar,CS>*>(factor.get());
         if (!rep)
           continue;

         bool front_conn = rep->Keyframe()->id == ids[i] && rep->Frame()->id == c;
         bool back_conn = rep->Keyframe()->id == c && rep->Frame()->id == ids[i];
         if (front_conn || back_conn)
         {
           cv::Mat matches = rep->DrawMatches();
           cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols/2));
           cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols/2, matches.cols));

           cv::Mat rec = df::DepthToProx(GetOpenCV(kf->pyr_dpt.GetCpuLevel(0)), opts_.sfm_params.avg_dpt);
           std::vector<cv::Mat> array;
           array.push_back(m1);
           array.push_back(m2);
           array.push_back(apply_colormap(rec));
           array.push_back(rep->ErrorImage());

           std::string ids_str = std::to_string(ids[i]);
           std::string c_str = std::to_string(c);
           std::string name = front_conn ? ids_str + "-" + c_str : c_str + "-" + ids_str;
           cv::Mat mosaic = CreateMosaic(array, 1, 4);
           cv::imwrite(rep_dir + "/" + name + ".png", mosaic);
         }
       }
    }
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::SaveGeometricDebug(std::string save_dir)
{

}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::UpdateMap(const gtsam::Values& vals, const gtsam::VectorValues& delta)
{
  // determine changed keyframes
  std::vector<FrameId> changed_kfs;
  for (auto& id : map_->keyframes.Ids())
  {
    gtsam::Key pkey = PoseKey(id);
    gtsam::Key ckey = CodeKey(id);

    if (delta[pkey].norm() > 1e-5 || delta[ckey].norm() > 1e-5)
      changed_kfs.push_back(id);
  }

  // modify each changed keyframe
  for (const auto& id : changed_kfs)
  {
    auto kf = map_->keyframes.Get(id);
    kf->code = vals.at(CodeKey(id)).template cast<gtsam::Vector>().template cast<float>();
    kf->pose_wk = vals.at(PoseKey(id)).template cast<Sophus::SE3f>();

    Eigen::Matrix<float,CS,1> cde = kf->code;
    for (uint i = 0; i < kf->pyr_img.Levels(); ++i)
    {
      df::UpdateDepth(cde, kf->pyr_prx_orig.GetGpuLevel(i),
                         kf->pyr_jac.GetGpuLevel(i),
                         (float)opts_.sfm_params.avg_dpt,
                         kf->pyr_dpt.GetGpuLevel(i));
    }
  }

  // modify all non-marginalized frames because it will never be many
  for (const auto& id : map_->frames.Ids())
  {
    auto fr = map_->frames.Get(id);
    if (!fr->marginalized)
      fr->pose_wk = vals.at(AuxPoseKey(id)).template cast<Sophus::SE3f>();
  }

  NotifyMapObservers();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename Mapper<Scalar,CS>::FramePtr
Mapper<Scalar,CS>::BuildFrame(const cv::Mat& img, const cv::Mat& col, const SE3T& pose_init,
                              const Features& ft)
{
  // create a new empty frame
  auto fr = std::make_shared<Frame>(cam_pyr_.Levels(), cam_pyr_[0].width(),
                                    cam_pyr_[0].height());
  fr->pose_wk = pose_init;
  fr->color_img = col.clone();
  fr->FillPyramids(img, cam_pyr_.Levels());
  fr->features = ft;
  fr->has_keypoints = true;
  return fr;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename Mapper<Scalar,CS>::KeyframePtr
Mapper<Scalar,CS>::BuildKeyframe(double timestamp, const cv::Mat& img, const cv::Mat& col,
                                 const SE3T& pose_init, const Features& ft)
{
  // create a new empty kf
  auto kf = std::make_shared<df::Keyframe<float>>(cam_pyr_.Levels(),
                                                     cam_pyr_[0].width(),
                                                     cam_pyr_[0].height(), CS);
  // fill code and pose
//  kf->code.setZero();
  kf->pose_wk = pose_init;
  kf->color_img = col.clone();
  kf->timestamp = timestamp;

  // fill pyramids
  for(uint i = 0; i < cam_pyr_.Levels(); ++i)
  {
    vc::image::fillBuffer(kf->pyr_vld.GetGpuLevel(i), 1.0f);

    if (i == 0)
    {
      vc::Image2DView<float, vc::TargetHost> tmp1(img);
      kf->pyr_img.GetGpuLevel(0).copyFrom(tmp1);
      df::SobelGradients(kf->pyr_img.GetGpuLevel(0), kf->pyr_grad.GetGpuLevel(0));
      continue;
    }

    df::GaussianBlurDown(kf->pyr_img.GetGpuLevel(i-1), kf->pyr_img.GetGpuLevel(i));
    df::SobelGradients(kf->pyr_img.GetGpuLevel(i), kf->pyr_grad.GetGpuLevel(i));
  }

  // decode zero code
  // preprocess color_img to network input format
  // color image is already of the network size and focal length
  // need to convert it to gray float 0-1
  cv::Mat netimg;
  cv::cvtColor(kf->color_img, netimg, cv::COLOR_RGB2GRAY);
  netimg.convertTo(netimg, CV_32FC1, 1/255.0);
  vc::Image2DView<float, vc::TargetHost> netimg_view(netimg);

  auto prx_orig_ptr = kf->pyr_prx_orig.GetCpuMutable();
  auto jac_ptr = kf->pyr_jac.GetCpuMutable();
  auto stdev_ptr = kf->pyr_stdev.GetCpuMutable();
  tic("running the network");
  {
    cuda::ScopedContextPop pop;
    const Eigen::VectorXf zero_code = Eigen::VectorXf::Zero(CS);

    if (opts_.predict_code)
    {
      Eigen::MatrixXf pred_code(CS,1);
      network_->PredictAndDecode(netimg_view, zero_code, &pred_code, prx_orig_ptr.get(), stdev_ptr.get(), jac_ptr.get());
      kf->code = pred_code;
      LOG(INFO) << "Predicted code: " << kf->code.transpose();
    }
    else
    {
      network_->Decode(netimg_view, zero_code, prx_orig_ptr.get(), stdev_ptr.get(), jac_ptr.get());
      kf->code = zero_code;
    }
  }
  toc("running the network");

  // fill depth based on code
  for (uint i = 0; i < cam_pyr_.Levels(); ++i)
  {
    df::UpdateDepth((Eigen::Matrix<Scalar,CS,1>)kf->code,
                       kf->pyr_prx_orig.GetGpuLevel(i),
                       kf->pyr_jac.GetGpuLevel(i),
                       opts_.sfm_params.avg_dpt,
                       kf->pyr_dpt.GetGpuLevel(i));
  }

  if (opts_.use_geometric)
  {
    // calculate dpt grad on level 0
    int w = cam_pyr_[0].width();
    int h = cam_pyr_[0].height();
    vc::Buffer2DManaged<typename Keyframe::GradT, vc::TargetDeviceCUDA> dpt_grad(w,h);
    df::SobelGradients(kf->pyr_dpt.GetGpuLevel(0), dpt_grad);
    kf->dpt_grad.copyFrom(dpt_grad);
  }

  kf->features = ft;
  kf->has_keypoints = true;

  return kf;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
std::vector<typename Mapper<Scalar,CS>::FrameId>
Mapper<Scalar,CS>::BuildBackConnections()
{
  // add photometric errors to previous frames
  int start = map_->keyframes.LastId();
  int stop = 1;
  switch (opts_.connection_mode)
  {
  case MapperOptions::LASTN:
    stop = std::max(1, start-opts_.max_back_connections+1);
    break;
  case MapperOptions::FULL:
    stop = 1;
    break;
  case MapperOptions::FIRST:
    stop = start = 1;
    break;
  case MapperOptions::LAST:
    start = stop = map_->keyframes.LastId();
    break;
  }

  std::vector<FrameId> conns;
  for (int i = start; i >= stop; i--)
    conns.push_back(i);
  return conns;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void Mapper<Scalar,CS>::NotifyMapObservers()
{
  if (map_callback_)
    map_callback_(map_);
}

/* ************************************************************************* */
// explicit instantiation
template class Mapper<float, DF_CODE_SIZE>;

} // namespace df
