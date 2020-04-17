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
#ifndef DF_INTRINSICS_H_
#define DF_INTRINSICS_H_

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "pinhole_camera.h"

namespace df 
{

inline
cv::Mat ChangeIntrinsics(const cv::Mat& img,
                         const df::PinholeCamera<float>& cam_in,
                         const df::PinholeCamera<float>& cam_out)
{
  cv::Mat inK, outK;
  cv::eigen2cv(cam_in.Matrix<float>(), inK);
  cv::eigen2cv(cam_out.Matrix<float>(), outK);

  cv::Mat map1, map2;
  cv::Size out_size(cam_out.width(), cam_out.height());
  cv::initUndistortRectifyMap(inK, cv::Mat{}, cv::Mat{}, outK, out_size, CV_32FC1, map1, map2);

  cv::Mat remapped;
  cv::remap(img, remapped, map1, map2, cv::INTER_LINEAR);
  return remapped;
}

} // namespace df

#endif // DF_INTRINSICS_H_
