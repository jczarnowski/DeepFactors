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
#ifndef DF_CAMERA_INTERFACE_FACTORY_H_
#define DF_CAMERA_INTERFACE_FACTORY_H_

#include <functional>
#include <memory>
#include <vector>
#include <map>

#include "camera_interface.h"

namespace df
{
namespace drivers
{

/**
 * Exception to indicate that user has passed
 * an URL that does not match accepted pattern URL_PATTERN
 */
class MalformedUrlException : public std::runtime_error
{
public:
  MalformedUrlException(const std::string& pattern, const std::string& reason)
  : std::runtime_error("Invalid source URL " + pattern + ": " + reason) {}
};

/**
 * Base class for a specific interface factory like pointgrey, files, etc
 */
class SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) = 0;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) = 0;
  virtual std::string GetPrefix() = 0;
};

/**
 * Singleton class that registers our supported supported interfaces
 * and produces CameraInterfaces based on URL
 */
class CameraInterfaceFactory
{
public:
  typedef std::map<std::string, std::shared_ptr<SpecificInterfaceFactory>> FactoryMapT;

  std::unique_ptr<CameraInterface> GetInterfaceFromUrl(const std::string& url);

  template <typename T>
  void RegisterInterface()
  {
    auto factory_obj = std::make_shared<T>();
    typename FactoryMapT::value_type pair(factory_obj->GetPrefix(), factory_obj);
    factory_map_.insert(pair);
    url_forms_.push_back(factory_obj->GetUrlPattern(prefix_tag_));
    supported_interfaces_.push_back(factory_obj->GetPrefix());
  }

  std::string GetUrlHelp();

  static std::shared_ptr<CameraInterfaceFactory> Get();

private:
  std::vector<std::string> PartitionUrl(const std::string& url);

  FactoryMapT factory_map_;
  std::vector<std::string> supported_interfaces_;
  std::vector<std::string> url_forms_;

  const std::string prefix_tag_ = "://";
  static std::shared_ptr<CameraInterfaceFactory> ptr_;
};

/**
 * Helper class to register new camera interfaces
 * Declare as static variable
 */
template <typename T>
struct InterfaceRegistrar {
  InterfaceRegistrar () {
    CameraInterfaceFactory::Get()->RegisterInterface<T>();
  }
};

} // namespace drivers
} //namespace df

#endif // DF_CAMERA_INTERFACE_FACTORY_H_
