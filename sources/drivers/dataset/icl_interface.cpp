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
#include <fstream>
#include <glog/logging.h>

#include "icl_interface.h"

namespace df
{
namespace drivers
{

static InterfaceRegistrar<IclInterfaceFactory> automatic;

IclInterface::IclInterface(std::string data_path)
    : data_path_(data_path), curr_frame_(-1)
{
  // load raw ground truth poses
  raw_poses_ = LoadPoses(data_path);

  // load list of image filenames with timestamps
  frames_ = ParseFrames(data_path);

  // skip the last frame in associations because
  // its pose is always missing
  frames_.pop_back();

  // match ground truth poses to images
  if (raw_poses_.size())
    AssignPoses(frames_, raw_poses_);
}

IclInterface::~IclInterface() {}

void IclInterface::GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt)
{
  if (HasMore())
    curr_frame_++;
  LoadFrame(curr_frame_, timestamp, img, dpt);
}

std::vector<DatasetFrame> IclInterface::GetAll()
{
  std::vector<DatasetFrame> frames;
  for (uint i = 0; i < frames_.size(); ++i)
  {
    DatasetFrame fr;
    LoadFrame(i, fr.timestamp, &fr.img, &fr.dpt);
    fr.pose_wf = frames_[i].pose_wf;
    frames.push_back(fr);
  }
  return frames;
}

df::PinholeCamera<float> IclInterface::GetIntrinsics()
{
  return df::PinholeCamera<float>(481.2,480.0,319.5,239.5,640,480);
}

void IclInterface::LoadFrame(int idx, double& timestamp, cv::Mat* img, cv::Mat* dpt)
{
  auto finfo = frames_[idx];

  if (img)
  {
    std::string path = data_path_ + "/" + finfo.img_path;
    *img = cv::imread(path);
    CHECK(!img->empty()) << "Failed to load file: " << path;
  }

  if (dpt)
  {
    std::string path = data_path_ + "/" + finfo.dpt_path;
    *dpt = cv::imread(path, cv::IMREAD_ANYDEPTH);
    dpt->convertTo(*dpt, CV_32FC1, 1.0 / 5000.0);
    CHECK(!img->empty()) << "Failed to load file: " << path;
  }

  // We take the rgb image timestamp to be the frame timestamp
  timestamp = finfo.img_timestamp;
}

std::vector<TumPose> IclInterface::LoadPoses(std::string data_path)
{
  std::string file_path = data_path + "/groundtruth.txt";
  std::ifstream f(file_path);

  if (!f.is_open())
  {
    LOG(INFO) << "groundtruth.txt not present, not loading the poses";
    return {};
  }

  std::vector<TumPose> poses;
  TumPose pose;
  std::string line;
  while (std::getline(f, line)) {
    if (line.find("#") != std::string::npos) // skip comments
      continue;

    std::stringstream ss(line);
    ss >> pose;
    pose.rot.normalize();
    poses.push_back(pose);
  }
  return poses;
}

std::vector<FrameInfo> IclInterface::ParseFrames(std::string data_path)
{
  std::ifstream f(data_path + "/associate.txt");
  CHECK(f) << "associate.txt not found! You need to download associate.py from the TUM "
           << "website and run 'python associate.py rgb.txt depth.txt > associate.txt'";

  std::string line;
  std::vector<FrameInfo> frames;
  while (std::getline(f, line))
  {
    if (line.find("#") != std::string::npos) // skip comment lines
      continue;

    FrameInfo finfo;
    std::stringstream ss(line);
    ss >> finfo.dpt_timestamp >> finfo.dpt_path >> finfo.img_timestamp >> finfo.img_path;
    frames.push_back(finfo);
  }
  return frames;
}

void IclInterface::AssignPoses(std::vector<FrameInfo>& frames,
                               const std::vector<TumPose>& rawposes)
{
  for (uint i = 0; i < frames.size(); ++i)
  {
    TumPose absolutePose = rawposes[i];
    TumPose firstPose = rawposes[0];

    // START: COPIED FROM TRISTAN
    // load relative pose
    Eigen::Quaternionf q_AB = firstPose.rot.inverse() * absolutePose.rot;
    q_AB.normalize();
    Eigen::Matrix3f C_WA = firstPose.rot.toRotationMatrix();
    Eigen::Matrix3f C_AW = C_WA.transpose();
    Eigen::Vector3f diff = absolutePose.trs - firstPose.trs;
    Eigen::Vector3f A_t_AB = C_AW * diff;

    // convert to matrix
    Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
    relPose.topLeftCorner<3,3>() = q_AB.toRotationMatrix();
    relPose.topRightCorner<3,1>() = A_t_AB;

    // fix ICL-NUIM pose issues
    relPose = relPose.inverse().eval();
    Eigen::Matrix4f Sy = Eigen::Matrix4f::Identity();
    Sy(1,1) = -1;
    relPose = (Sy * relPose * Sy).eval();
    relPose = relPose.inverse().eval();
    // END: COPIED FROM TRISTAN

    // convert 4x4 mat to sophus
    Eigen::Quaternionf rot(relPose.topLeftCorner<3,3>());
    frames[i].pose_wf.setQuaternion(rot);
    frames[i].pose_wf.translation() = relPose.topRightCorner<3,1>();

//    // make the pose relative to the first frame
//    if (i > 0)
//      frames[i].pose_wf = frames[0].pose_wf.inverse() * frames[i].pose_wf;

    // make the timestamp relative to the first frame
//    frames[i].img_timestamp -= frames[0].img_timestamp;

    poses_.push_back(frames[i].pose_wf);
  }

//  frames[0].pose_wf = Sophus::SE3f{};
//  poses_[0] = Sophus::SE3f{};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Factory class for this interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<CameraInterface> IclInterfaceFactory::FromUrlParams(const std::string& url_params)
{
  return std::make_unique<IclInterface>(url_params);
}

std::string IclInterfaceFactory::GetUrlPattern(const std::string& prefix_tag)
{
  return url_prefix + prefix_tag + url_params;
}

std::string IclInterfaceFactory::GetPrefix()
{
  return url_prefix;
}

} // namespace drivers
} // namespace df

