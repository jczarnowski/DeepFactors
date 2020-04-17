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
#include "openni_interface.h"

#include <glog/logging.h>
#include <OpenNIDriver.hpp>
#include <opencv2/opencv.hpp>

namespace df
{
namespace drivers
{

static InterfaceRegistrar<OpenNIInterfaceFactory> automatic;

OpenNIInterface::OpenNIInterface(std::size_t camera_id)
{
  driver = std::make_unique<::drivers::camera::OpenNI>();

  try
  {
    driver->open(camera_id);
    driver->setDepthMode(640, 480, 30, false, true);
    driver->setRGBMode(640, 480, 30, ::drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8);
    driver->setDepthMirroring(false);
    driver->setRGBMirroring(false);
    driver->setSynchronization(true);

    driver->setFeatureAuto(::drivers::camera::EFeature::WHITE_BALANCE, false);
    driver->setFeatureAuto(::drivers::camera::EFeature::EXPOSURE, true);

    driver->setFeatureValueAbs(::drivers::camera::EFeature::SHUTTER, 1000);
//    driver->setFeatureValueAbs(::drivers::camera::EFeature::GAIN, 50);

    float fx, fy, u0, v0;
    driver->getRGBIntrinsics(fx,fy,u0,v0);
    LOG(INFO) << "OpenNI Calibration: " << fx << ", " << fy << ", " << u0 << ", " << v0;

    driver->start(true, true);
  }
  catch (std::exception& e)
  {
    throw std::runtime_error("Failed to open and configure OpenNI camera: " + std::string(e.what()));
  }
}

OpenNIInterface::~OpenNIInterface()
{
  driver->stop();
  driver->close();
}

void OpenNIInterface::SetGain(float gain)
{
  driver->setFeatureValueAbs(::drivers::camera::EFeature::GAIN, gain);
}

void OpenNIInterface::SetGainAuto(bool on)
{
  // instead of setting gain, this controls white balance and exposure
  // its meant to be used to switch on/off all the color changes
  driver->setFeatureAuto(::drivers::camera::EFeature::WHITE_BALANCE, on);
  driver->setFeatureAuto(::drivers::camera::EFeature::EXPOSURE, on);
}

float OpenNIInterface::GetGain()
{
  return driver->getFeatureValueAbs(::drivers::camera::EFeature::GAIN);
}

void OpenNIInterface::SetShutter(float shutter)
{
  driver->setFeatureValueAbs(::drivers::camera::EFeature::SHUTTER, shutter);
}

void OpenNIInterface::SetShutterAuto(bool on)
{
  // unsupported
}

float OpenNIInterface::GetShutter()
{
  return driver->getFeatureValueAbs(::drivers::camera::EFeature::SHUTTER);
}

void OpenNIInterface::GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt)
{
  ::drivers::camera::FrameBuffer fb_dpt, fb_rgb;

  if (!driver->captureFrame(fb_dpt, fb_rgb))
    throw std::runtime_error("Error grabbing frame");

  if (img != nullptr)
  {
    cv::Mat ptr_mat(fb_rgb.getHeight(), fb_rgb.getWidth(), CV_8UC3, fb_rgb.getData());
    ptr_mat.copyTo(*img);
    cv::cvtColor(*img, *img, cv::COLOR_RGB2BGR);
    timestamp = static_cast<double>(fb_rgb.getTimeStamp());
  }

  if (dpt != nullptr)
  {
    cv::Mat ptr_mat(fb_dpt.getHeight(), fb_dpt.getWidth(), CV_16UC1, fb_dpt.getData());
    ptr_mat.convertTo(*dpt, CV_32FC1, 1.0/1000.0);
  }
}

df::PinholeCamera<float> OpenNIInterface::GetIntrinsics()
{
  return df::PinholeCamera<float>(570.342, 570.342, 320, 240, 640, 480);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Factory class for this interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<CameraInterface> OpenNIInterfaceFactory::FromUrlParams(const std::string& url_params)
{
  // parse cameraid from params
  std::size_t camera_id = 0;
  if (url_params.length())
    camera_id = std::stoul(url_params);

  return std::make_unique<OpenNIInterface>(camera_id);
}

std::string OpenNIInterfaceFactory::GetUrlPattern(const std::string& prefix_tag)
{
  return url_prefix + prefix_tag + url_params;
}

std::string OpenNIInterfaceFactory::GetPrefix()
{
  return url_prefix;
}

} // namespace drivers
} // namespace df
