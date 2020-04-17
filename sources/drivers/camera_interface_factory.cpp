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
#include <glog/logging.h>

#include "camera_interface_factory.h"
#include "camera_interface.h"

namespace df
{
namespace drivers
{

// out of line declaration for static members
std::shared_ptr<CameraInterfaceFactory> CameraInterfaceFactory::ptr_;

std::shared_ptr<CameraInterfaceFactory> CameraInterfaceFactory::Get()
{
  if (!ptr_)
    ptr_ = std::make_shared<CameraInterfaceFactory>();
  return ptr_;
}

std::unique_ptr<CameraInterface> CameraInterfaceFactory::GetInterfaceFromUrl(const std::string& url)
{
  // Find the URL type prefix "://"
  auto url_parts = PartitionUrl(url);
  std::string prefix = url_parts[0];
  std::string remainder = url_parts[1];

  LOG(INFO) << "prefix: " << url_parts[0] << ", remainder: " << url_parts[1];

  if (factory_map_.find(prefix) == factory_map_.end())
  {
    std::stringstream ss;
    ss << "Interface for prefix " << prefix << " not registered" << std::endl;
    ss << "Supported interfaces: ";
    for (auto& pref : supported_interfaces_)
      ss << pref << ", ";
    throw MalformedUrlException(url, ss.str());
  }

  return factory_map_[prefix]->FromUrlParams(remainder);
}

std::string CameraInterfaceFactory::GetUrlHelp()
{
  std::stringstream ss;
  ss << "Supported URLs:" << std::endl;
  for (auto& url_form : url_forms_)
    ss << url_form << std::endl;
  return ss.str();
}

std::vector<std::string> CameraInterfaceFactory::PartitionUrl(const std::string& url)
{
  auto pos = url.find(prefix_tag_);
  if (pos > url.length())
    throw MalformedUrlException(url, "Missing tag: " + prefix_tag_);

  std::string prefix = url.substr(0, pos);
  std::string remainder = url.substr(pos + prefix_tag_.length());
  return {prefix, remainder};
}

} // namespace drivers
} //namespace df
