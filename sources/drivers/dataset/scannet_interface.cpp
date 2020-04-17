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
#include "scannet_interface.h"

#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

namespace df
{
namespace drivers
{

std::vector<std::string> split(const std::string &s, char delim)
{
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

static InterfaceRegistrar<ScanNetInterfaceFactory> automatic;

ScanNetInterface::ScanNetInterface(std::string seq_path)
    : seq_path_(seq_path), current_frame_(-1)
{
  /* ScanNet structure
   * <scene_id>
   *    color
   *       *.jpg
   *    intrinsic
   *       extrinsic_color.txt
   *       extrinsic_depth.txt  
   *       intrinsic_color.txt  
   *       intrinsic_depth.txt
   *    pose
   *       *.txt
   */

  // check if the dataset has depth
  has_depth_ = boost::filesystem::is_directory(seq_path + "/depth");
  
  // generate frame files
  std::size_t num_frames = ProbeNumberOfFrames(seq_path);
  for (std::size_t i = 0; i < num_frames; ++i)
  {
    std::string frame_str = std::to_string(i);
    img_files_.push_back("color/" + frame_str + ".jpg");
    dpt_files_.push_back("depth/" + frame_str + ".png");
    pose_files_.push_back("pose/" + frame_str + ".txt");
  }

  LOG(INFO) << "Parsed " << num_frames << " frames from: " << seq_path;
  if (!has_depth_)
    LOG(INFO) << "Depth images not found";

  // load poses
  poses_ = LoadPoses();
  
  // probe the first file for image size
  auto first_img = cv::imread(seq_path + "/" + img_files_[0]);

  // load intrinsics
  cam_ = LoadIntrinsics(seq_path_, first_img.cols, first_img.rows);
  cam_.ResizeViewport(640,480);
}

ScanNetInterface::~ScanNetInterface() {}

void ScanNetInterface::GrabFrames(double &timestamp, cv::Mat* img, cv::Mat* dpt)
{
  if (HasMore())
    current_frame_++;
  Sophus::SE3f dummy;
  LoadFrame(current_frame_, timestamp, img, dpt, dummy);
}

std::vector<DatasetFrame> ScanNetInterface::GetAll()
{
  std::vector<DatasetFrame> frames;
  frames.reserve(img_files_.size());
  for (uint i = 0; i < img_files_.size(); ++i)
  {
    DatasetFrame fr;
    LoadFrame(i, fr.timestamp, &fr.img, &fr.dpt, fr.pose_wf);
    fr.timestamp = i;
    frames.push_back(fr);
  }
  return frames;
}

df::PinholeCamera<float> ScanNetInterface::GetIntrinsics()
{
  return cam_;
}

void ScanNetInterface::LoadFrame(int i, double& timestamp, cv::Mat* img, cv::Mat* dpt, Sophus::SE3f& pose)
{
  if (dpt && !has_depth_)
    LOG(FATAL) << "Requesting to load depth when the dataset does not support it";

  if (img)
  {
    std::string img_path = seq_path_ + "/" + img_files_[i];
    *img = cv::imread(img_path);
    CHECK(!img->empty()) << "Failed to load image: " << img_path;
    cv::resize(*img, *img, cv::Size(640, 480));
    timestamp = static_cast<double>(i);
  }

  if (dpt)
  {
    std::string dpt_path = seq_path_ + "/" + dpt_files_[i];
    *dpt = cv::imread(dpt_path, cv::IMREAD_ANYDEPTH);
    dpt->convertTo(*dpt, CV_32FC1, 0.001);
    CHECK(!dpt->empty()) << "Failed to load depth: " << dpt_path;
  }

  pose = poses_[i];
}

df::PinholeCamera<float> 
ScanNetInterface::LoadIntrinsics(std::string seq_dir, int width, int height)
{
  std::string intr_file = seq_dir + "/intrinsic/intrinsic_color.txt";
  std::ifstream f(intr_file);
  CHECK(f.is_open()) << "Failed to open file: " << intr_file;

  // load intrinsic matrix
  Eigen::Matrix4f K;
  std::string elem;
  for (int x = 0; x < 4; ++x)
  {
    for (int y = 0; y < 4; ++y)
    {
      f >> elem;
      K(x,y) = std::stof(elem);
    }
  }

  return df::PinholeCamera<float>(K(0,0), K(1,1), K(0,2), K(1,2), width, height);
}

std::vector<Sophus::SE3f> ScanNetInterface::LoadPoses()
{
  std::vector<Sophus::SE3f> poses;
  Sophus::SE3f first_pose;
  for (uint i = 0; i < pose_files_.size(); ++i)
  {
    std::string pose_file = seq_path_ + "/" + pose_files_[i];
    std::ifstream f(pose_file);
    CHECK(f.is_open()) << "Failed to open file: " << pose_file;

    Eigen::Matrix4f T;
    std::string elem;
    for (int x = 0; x < 4; ++x)
    {
      for (int y = 0; y < 4; ++y)
      {
        f >> elem;
        T(x,y) = std::stof(elem);
      }
    }

    if (T.allFinite())
    {
      auto pose = Sophus::SE3f(T);

      if (i == 0)
        first_pose = pose;
      poses.push_back(first_pose.inverse() * pose);
    }
  }
  return poses;
}

std::size_t ScanNetInterface::ProbeNumberOfFrames(std::string dir)
{
  using namespace boost::filesystem;
  return std::count_if(
      directory_iterator(path(dir + "/color")),
      directory_iterator(),
      [] (const path& p) { return is_regular_file(p); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Factory class for this interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<CameraInterface> ScanNetInterfaceFactory::FromUrlParams(const std::string& url_params)
{
  return std::make_unique<ScanNetInterface>(url_params);
}

std::string ScanNetInterfaceFactory::GetUrlPattern(const std::string& prefix_tag)
{
  return url_prefix + prefix_tag + url_params;
}

std::string ScanNetInterfaceFactory::GetPrefix()
{
  return url_prefix;
}

} // namespace drivers
} // namespace df
