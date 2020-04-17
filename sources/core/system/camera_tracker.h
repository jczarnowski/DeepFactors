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
#ifndef DF_CAMERA_TRACKER_H_
#define DF_CAMERA_TRACKER_H_

#include <cstddef>
#include <vector>

#include <sophus/se3.hpp>
#include <VisionCore/Buffers/BufferPyramid.hpp>

#include "cu_se3aligner.h"
#include "keyframe.h"
#include "camera_pyramid.h"

// forward declarations
namespace cv { class Mat; }

namespace df
{

/**
 * Tracks against a specified keyframe
 * Implements gaussian pyramid scheme
 * Uses SO3Aligner and SE3Aligner
 */
class CameraTracker
{
public:
  typedef Eigen::Matrix<float,1,2> GradType;
  typedef df::Keyframe<float> KeyframeT;
  typedef vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> ImageBufferPyramid;
  typedef vc::RuntimeBufferPyramidManaged<GradType, vc::TargetDeviceCUDA> GradBufferPyramid;

  struct TrackerConfig
  {
    std::size_t               pyramid_levels;
    std::vector<int>          iterations_per_level = {10, 5, 4};
    double                    huber_delta;
  };

  CameraTracker() = delete;
  CameraTracker(const df::CameraPyramid<float>& camera_pyr, const TrackerConfig& config);
  virtual ~CameraTracker();

  void TrackFrame(const ImageBufferPyramid& img1, const GradBufferPyramid& grad1);
  void Reset();

  void SetKeyframe(std::shared_ptr<const KeyframeT> kf);
  void SetPose(const Sophus::SE3f& pose_wc);
  void SetConfig(const TrackerConfig& new_cfg);

  Sophus::SE3f GetPoseEstimate();
  float GetInliers() { return inliers_; }
  float GetError() { return error_; }
  cv::Mat GetResidualImage() { return residual_; }

private:
  float inliers_;
  float error_;
  Sophus::SE3f pose_ck_;
  TrackerConfig config_;
  CameraPyramid<float> camera_pyr_;
  std::shared_ptr<const KeyframeT> kf_;
  cv::Mat residual_;

  df::SE3Aligner<float> se3aligner_;
//   df::SO3Aligner so3aligner_;
};

} // namespace df

#endif // DF_CAMERA_TRACKER_H_
