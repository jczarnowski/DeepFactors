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
#ifndef DF_ICL_INTERFACE_H
#define DF_ICL_INTERFACE_H

#include <sophus/se3.hpp>

#include "tum_io.h"
#include "dataset_interface.h"
#include "camera_interface_factory.h"

namespace df
{
namespace drivers
{

struct FrameInfo
{
  double img_timestamp;
  double dpt_timestamp;
  std::string img_path;
  std::string dpt_path;
  Sophus::SE3f pose_wf;
};

class IclInterface : public DatasetInterface
{
public:
  IclInterface(std::string data_path);
  virtual ~IclInterface();

  virtual void GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) override;
  virtual std::vector<DatasetFrame> GetAll() override;
  virtual std::vector<Sophus::SE3f> GetPoses() override { return poses_; }

  virtual bool SupportsDepth() override { return true; }
  virtual bool HasIntrinsics() override { return true; }
  virtual bool HasPoses() override { return true; }
  virtual bool HasMore() override { return curr_frame_ < (int)frames_.size()-1; }

  virtual df::PinholeCamera<float> GetIntrinsics() override;

private:
  void LoadFrame(int idx, double& timestamp, cv::Mat* img, cv::Mat* dpt);

  std::vector<TumPose> LoadPoses(std::string data_path);
  std::vector<FrameInfo> ParseFrames(std::string data_path);
  void AssignPoses(std::vector<FrameInfo>& frames, const std::vector<TumPose>& rawposes);

  std::vector<TumPose> raw_poses_;
  std::vector<FrameInfo> frames_;
  std::vector<Sophus::SE3f> poses_; // caching these to for GetPoses
  std::string data_path_;
  int curr_frame_;
};

class IclInterfaceFactory : public SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) override;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) override;
  virtual std::string GetPrefix() override;

private:
  const std::string url_prefix = "icl";
  const std::string url_params = "sequence_path";
};

} // namespace drivers
} // namespace df

#endif // DF_ICL_INTERFACE_H
