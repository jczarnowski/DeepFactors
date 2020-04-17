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
#ifndef DF_LOOP_DETECTOR_H_
#define DF_LOOP_DETECTOR_H_

#include <vector>
#include <memory>

#include <VisionCore/Buffers/BufferPyramid.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <DBoW2.h>

#include "fbrisk.h"
#include "keyframe.h"
#include "keyframe_map.h"
#include "feature_detection.h"
#include "camera_tracker.h"
#include "camera_pyramid.h"

namespace df
{

typedef DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>
  BriskVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>
  BriskDatabase;

struct LoopDetectorConfig
{
  CameraTracker::TrackerConfig tracker_cfg;
  std::vector<int> iters;
  float min_similarity = 0.35f;
  float max_error = 0.5f;
  float max_dist = 0.1f;
  int max_candidates = 3;
  int active_window = 5;
};

template <typename Scalar>
class LoopDetector
{
public:
  typedef Keyframe<Scalar> KeyframeT;
  typedef typename KeyframeT::Ptr KeyframePtr;
  typedef typename Map<Scalar>::Ptr MapPtr;
  typedef std::shared_ptr<CameraTracker> TrackerPtr;
  typedef CameraTracker::TrackerConfig TrackerConfig;
  typedef CameraPyramid<Scalar> CameraPyr;
  typedef typename KeyframeT::IdType IdType;
  typedef std::map<IdType, DBoW2::BowVector> DescriptionMap;
  typedef Eigen::Matrix<Scalar,1,2> ImageGrad;
  typedef vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> ImagePyramid;
  typedef vc::RuntimeBufferPyramidManaged<ImageGrad, vc::TargetDeviceCUDA> GradPyramid;
  typedef Sophus::SE3<Scalar> SE3T;

  struct LoopInfo
  {
    int loop_id = 1;
    SE3T pose_wc;
    bool detected = false;
  };

  LoopDetector(std::string voc_path, MapPtr map, LoopDetectorConfig cfg, CameraPyr cam);

  void AddKeyframe(KeyframePtr kf);
  int DetectLoop(KeyframePtr kf);
  LoopInfo DetectLoop(const ImagePyramid& pyr_img, const GradPyramid& pyr_grad,
                      const Features& ft, KeyframePtr curr_kf, SE3T pose_cam);
  int DetectLocalLoop(const ImagePyramid& pyr_img, const GradPyramid& pyr_grad,
                      const Features& ft, KeyframePtr curr_kf, SE3T pose_cam);

  void Reset();

private:
  std::vector<std::vector<uchar>> ChangeStructure(cv::Mat mat);

  DescriptionMap dmap_;
  BriskVocabulary voc_;
  BriskDatabase db_;
  MapPtr map_;
  TrackerPtr tracker_;
  LoopDetectorConfig cfg_;
};

} // namespace df

#endif // DF_LOOP_DETECTOR_H_
