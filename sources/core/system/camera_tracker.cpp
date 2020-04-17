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
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <VisionCore/Buffers/Image2D.hpp>

#include "camera_tracker.h"
#include "cu_image_proc.h"

namespace df
{

CameraTracker::CameraTracker(const df::CameraPyramid<float>& camera_pyr, const TrackerConfig& config)
    : config_(config), camera_pyr_(camera_pyr)
{
  // check the config
  if(config_.iterations_per_level.size() != config_.pyramid_levels)
    LOG(FATAL) << "CameraTracker config error: iterations_per_level size not equal pyramid_levels";
  se3aligner_.SetHuberDelta(config_.huber_delta);
}

CameraTracker::~CameraTracker()
{

}

void CameraTracker::TrackFrame(const ImageBufferPyramid& pyr_img1, const GradBufferPyramid& pyr_grad1)
{
  if (!kf_)
    throw std::runtime_error("Calling CameraTracker::TrackFrame before a keyframe was set");

  // run iterations
  for (int level = config_.pyramid_levels-1; level >= 0; --level)
  {
    for (int iter = 0; iter < config_.iterations_per_level[level]; ++iter)
    {
      // run se3aligners
      auto result = se3aligner_.RunStep(pose_ck_, camera_pyr_[level],
                                        kf_->pyr_img.GetGpuLevel(level),
                                        pyr_img1[level],
                                        kf_->pyr_dpt.GetGpuLevel(level),
                                        pyr_grad1[level]);
      // update estimate
      Eigen::Matrix<float,6,1> update = -result.JtJ.toDenseMatrix().ldlt().solve(result.Jtr);
      Eigen::Matrix<float,3,1> trs_update = update.head<3>();
      Eigen::Matrix<float,3,1> rot_update = update.tail<3>();
      pose_ck_.translation() += trs_update;
      pose_ck_.so3() = Sophus::SO3f::exp(rot_update) * pose_ck_.so3();

      if (level == 0 && iter == config_.iterations_per_level[level]-1)
      {
        inliers_ = result.inliers / (float) pyr_img1[level].area();
        error_ = result.inliers != 0 ? result.residual / result.inliers : std::numeric_limits<float>::infinity();
      }
    }
  }

  // get a residual image for display
  int level = 0;
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> warped(pyr_img1[level].width(), pyr_img1[level].height());
  vc::Image2DManaged<float, vc::TargetHost> warped_host(warped.width(), warped.height());
  vc::Image2DManaged<float, vc::TargetHost> kfimg_host(warped.width(), warped.height());
  se3aligner_.Warp(pose_ck_, camera_pyr_[level], kf_->pyr_img.GetGpuLevel(level), pyr_img1[level], kf_->pyr_dpt.GetGpuLevel(level), warped);
  warped_host.copyFrom(warped);
  kfimg_host.copyFrom(kf_->pyr_img.GetGpuLevel(level));
  cv::Mat absdiff = cv::abs(warped_host.getOpenCV()-kfimg_host.getOpenCV());
  residual_ = absdiff.clone();

  // DEBUG: outlier image
//  cv::Mat res_color;
//  cv::cvtColor(residual_ , res_color, CV_GRAY2RGB);
//  res_color.setTo(cv::Vec3f(1.0,0,0), cv::abs(residual_) > config_.huber_delta);
////  cv::imshow("tracking residual", res);
//  cv::imshow("outliers", res_color);
//  cv::waitKey(1);
}

void CameraTracker::Reset()
{
  pose_ck_ = Sophus::SE3f();
}

Sophus::SE3f CameraTracker::GetPoseEstimate()
{
  // convert calculate pose_wc using pose_wk and pose_ck
  // wc = wk * ck.inverse()
  return kf_->pose_wk * pose_ck_.inverse();
}

void CameraTracker::SetKeyframe(std::shared_ptr<const KeyframeT> kf)
{
  // change the current pose estimate to proper camera frame
  if (kf_)
  {
    // transform pose ck to wc using old keyframe
    auto wc = kf_->pose_wk * pose_ck_.inverse();

    // calculate pose w.r.t new keyframe
    pose_ck_ = wc.inverse() * kf->pose_wk;
//    pose_ck_ = Sophus::SE3f{};
  }

  // set the pointer
  kf_ = kf;
}

void CameraTracker::SetPose(const Sophus::SE3f& pose_wc)
{
  // calculate pose w.r.t new keyframe
  pose_ck_ = pose_wc.inverse() * kf_->pose_wk;
}

void CameraTracker::SetConfig(const TrackerConfig& new_cfg)
{
  // check options that can't be changed online
  if(new_cfg.pyramid_levels != config_.pyramid_levels)
    LOG(FATAL) << "CameraTracker config error: online changes to pyramid_levels are not allowed";

  // validate parameters
  if (new_cfg.iterations_per_level.size() != config_.pyramid_levels)
    LOG(FATAL) << "CameraTracker config error: iterations_per_level size not equal pyramid_levels";

  config_ = new_cfg;
}

} // namespace df
