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
#ifndef DF_DEEPFACTORS_H_
#define DF_DEEPFACTORS_H_

#include <cstddef>
#include <memory>
#include <vector>
#include <Eigen/Core>

#include "mapper.h"
#include "camera_tracker.h"
#include "pinhole_camera.h"
#include "decoder_network.h"
#include "cuda_context.h"
#include "keyframe_map.h"
#include "deepfactors_options.h"
#include "loop_detector.h"
#include "feature_detection.h"

// debug temporary
#include "cu_se3aligner.h"

namespace cv { class Mat; }

namespace df
{

struct DeepFactorsStatistics
{
  float inliers;
  float distance;
  float tracker_error;
  cv::Mat residual;
  std::map<std::size_t, bool> relin_info;
};

template <typename Scalar, int CS>
class DeepFactors
{
public:
  typedef Scalar ScalarT;
  typedef Sophus::SE3<Scalar> SE3T;
  typedef df::Keyframe<Scalar> KeyframeT;
  typedef typename df::Keyframe<Scalar>::IdType KeyframeId;
  typedef df::Map<Scalar> MapT;
  typedef typename MapT::Ptr MapPtr;
  typedef df::Mapper<Scalar,CS> MapperT;
  typedef LoopDetector<Scalar> LoopDetectorT;
  typedef vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> ImagePyrT;
  typedef vc::RuntimeBufferPyramidManaged<Eigen::Matrix<Scalar,1,2>, vc::TargetDeviceCUDA> GradPyrT;

  // callback types
  typedef std::function<void (MapPtr)> MapCallbackT;
  typedef std::function<void (const SE3T&)> PoseCallbackT;
  typedef std::function<void (const DeepFactorsStatistics&)> StatsCallbackT;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
  DeepFactors();
  virtual ~DeepFactors();

  /* Do not create copies of this object */
  DeepFactors(const DeepFactors& other) = delete;

  /*
   * Logic
   */
  void Init(df::PinholeCamera<Scalar>& cam, const DeepFactorsOptions& opts);
  void Reset();
  void ProcessFrame(double timestamp, const cv::Mat& frame);

  /*
   * Initialize the system with two images and optimize them
   */
  void BootstrapTwoFrames(double ts1, double ts2, const cv::Mat& img0, const cv::Mat& img1);

  /*
   * Initialize the system with a single image (decodes zero-code only)
   */
  void BootstrapOneFrame(double timestamp, const cv::Mat& img);

  void ForceKeyframe();
  void ForceFrame();

  /*
   * Getters
   */
  MapPtr GetMap() { return mapper_->GetMap(); }
  SE3T GetCameraPose() { return pose_wc_; }
  df::DecoderNetwork::NetworkConfig GetNetworkConfig() { return netcfg_; }
  DeepFactorsStatistics GetStatistics() { return stats_; }
  df::PinholeCamera<Scalar> GetNetworkCam();

  /*
   * Setters
   */
  void SetMapCallback(MapCallbackT cb) { map_callback_ = cb; mapper_->SetMapCallback(map_callback_); }
  void SetPoseCallback(PoseCallbackT cb) { pose_callback_ = cb; }
  void SetStatsCallback(StatsCallbackT cb) { stats_callback_ = cb; }
  void SetOptions(DeepFactorsOptions opts);

  /*
   * Notifications
   */
  void NotifyPoseObservers();
  void NotifyMapObservers();
  void NotifyStatsObservers();

  /*
   * Debugging
   */
  void SavePostCrashInfo(std::string dir);
  void SaveResults(std::string dir);
  void SaveKeyframes(std::string dir);

private:
  void InitGpu(std::size_t device_id);
  void UploadLiveFrame(const cv::Mat& frame);
  cv::Mat PreprocessImage(const cv::Mat& frame, cv::Mat& out_color,
                          Features& features);
  SE3T TrackFrame();
  SE3T Relocalize();

  bool NewKeyframeRequired();
  bool NewFrameRequired();
  KeyframeId SelectKeyframe();
  bool CheckTrackingLost(const SE3T& pose_wc);

private:
  bool force_keyframe_;
  bool force_frame_;
  bool bootstrapped_;
  bool tracking_lost_;
  KeyframeId curr_kf_;
  KeyframeId prev_kf_;
  SE3T pose_wc_;
  DeepFactorsOptions opts_;
  DeepFactorsStatistics stats_;

  std::shared_ptr<DecoderNetwork> network_;
  std::shared_ptr<CameraTracker> tracker_;
  std::unique_ptr<MapperT> mapper_;
  std::unique_ptr<LoopDetectorT> loop_detector_;
  std::unique_ptr<FeatureDetector> feature_detector_;

  df::CameraPyramid<Scalar> camera_pyr_;
  df::PinholeCamera<Scalar> orig_cam_;
  df::DecoderNetwork::NetworkConfig netcfg_;
  cv::Mat map1_, map2_; // maps from initUndistorRectifyMap

  // buffers for the live frame
  std::shared_ptr<ImagePyrT> pyr_live_img_;
  std::shared_ptr<GradPyrT> pyr_live_grad_;

  MapCallbackT map_callback_;
  PoseCallbackT pose_callback_;
  StatsCallbackT stats_callback_;

  // debug stuff
  std::vector<std::pair<int,int>> loop_links;

  struct DebugImages
  {
    cv::Mat reprojection_errors;
    cv::Mat photometric_errors;
  };
  std::list<DebugImages> debug_buffer_;

  typedef df::SE3Aligner<Scalar> SE3AlignerT;
  typename SE3AlignerT::Ptr se3aligner_;
};

} // namespace df

#endif // DF_H_
