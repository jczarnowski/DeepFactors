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
#include "file_interface.h"

#include <memory>
#include <fstream>
#include <glob.h>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace df
{
namespace drivers
{

static InterfaceRegistrar<FileInterfaceFactory> automatic;

inline std::vector<std::string> glob(const std::string& pat) {
  using namespace std;
  glob_t glob_result;
  glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
  vector<string> ret;
  for(unsigned int i=0;i<glob_result.gl_pathc;++i){
    ret.push_back(string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  return ret;
}

FileInterface::FileInterface(const std::string& glob_path) : count_(0)
{
  files_ = glob(glob_path + "/*.png");

  if (files_.empty())
    throw std::runtime_error("No matches found for glob: " + glob_path);

  LOG(INFO) << "Parsed " << files_.size() << " image files";

  std::ifstream file(glob_path + "/cam.txt");

  float fx, fy, u0, v0, w, h;
  file >> fx >> fy >> u0 >> v0 >> w >> h;
  file.close();

  cam_ = df::PinholeCamera<float>(fx, fy, u0, v0, w, h);
}

FileInterface::~FileInterface()
{
}

void FileInterface::GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt)
{
  if (dpt != nullptr)
    throw UnsupportedFeatureException("Depth not supported by this camera");

  auto filename = files_[count_];
  timestamp = static_cast<double>(count_);
  if (count_ < (int)files_.size()-1)
    count_++;
  *img = cv::imread(filename, cv::IMREAD_COLOR);
  if (img->empty())
    throw std::runtime_error("Failed to load image " + std::to_string(count_-1) + ": " + filename);
}

bool FileInterface::HasMore()
{
  return count_ < (int)files_.size()-2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Factory class for this interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<CameraInterface> FileInterfaceFactory::FromUrlParams(const std::string& url_params)
{
  // easy enough: we expect url_params to be a file glob so pass it directly to the class
  return std::make_unique<FileInterface>(url_params);
}

std::string FileInterfaceFactory::GetUrlPattern(const std::string& prefix_tag)
{
  return url_prefix + prefix_tag + url_params;
}

std::string FileInterfaceFactory::GetPrefix()
{
  return url_prefix;
}

} // namespace drivers
} // namespace df
