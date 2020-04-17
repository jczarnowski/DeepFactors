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
#ifndef DF_CAMERA_PYRAMID_H_
#define DF_CAMERA_PYRAMID_H_

#include <vector>
#include "pinhole_camera.h"

namespace df
{

template <typename T>
class CameraPyramid
{
  typedef df::PinholeCamera<T> CameraT;

public:
  CameraPyramid() {}

  CameraPyramid(const CameraT& cam, std::size_t levels) : levels_(levels)
  {
    for (std::size_t i = 0; i < levels; ++i)
    {
      cameras_.emplace_back(cam);

      if (i != 0)
      {
        std::size_t new_width = cameras_[i-1].width() / 2;
        std::size_t new_height = cameras_[i-1].height() / 2;
        cameras_[i].ResizeViewport(new_width, new_height);
      }
    }
  }

  const CameraT& operator[](int i) const
  {
    return cameras_[i];
  }

  CameraT& operator[](int i)
  {
    return cameras_[i];
  }

  std::size_t Levels() const { return levels_; }

private:
  std::vector<CameraT> cameras_;
  std::size_t levels_;
};

} // namespace df

template <typename T>
std::ostream& operator<<(std::ostream& os, const df::CameraPyramid<T>& pyr)
{
  os << "CameraPyramid:" << std::endl;
  for (std::size_t i = 0; i < pyr.Levels(); ++i)
    os << "Level " << i << " " << pyr[i] << std::endl;
  return os;
}

#endif // DF_CAMERA_PYRAMID_H_
