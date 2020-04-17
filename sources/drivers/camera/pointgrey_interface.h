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
#ifndef DF_POINTGREY_INTERFACE_H_
#define DF_POINTGREY_INTERFACE_H_

#include <memory>
#include "live_interface.h"
#include "camera_interface_factory.h"

namespace drivers { namespace camera { class PointGrey; } }

namespace df
{
namespace drivers
{

class PointGreyInterface : public LiveInterface
{
public:
  PointGreyInterface(uint camera_id);
  virtual ~PointGreyInterface() override;

  virtual void SetGain(float gain) override;
  virtual void SetGainAuto(bool on) override;
  virtual float GetGain() override;
  virtual void SetShutter(float shutter) override;
  virtual void SetShutterAuto(bool on) override;
  virtual float GetShutter() override;

  virtual void GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) override;

private:
  std::unique_ptr<::drivers::camera::PointGrey> driver;
};

class PointGreyInterfaceFactory : public SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) override;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) override;
  virtual std::string GetPrefix() override;

private:
  const std::string url_prefix = "flycap";
  const std::string url_params = "[cameraId]";
};

} // namespace drivers
} // namespace df

#endif // DF_POINTGREY_INTERFACE_H_
