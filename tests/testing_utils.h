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
#ifndef DF_TESTING_UTILS_H_
#define DF_TESTING_UTILS_H_

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include "pinhole_camera.h"
#include "intrinsics.h"

namespace df
{

template <typename T>
df::PinholeCamera<T> GetSceneNetCam(std::size_t w, std::size_t h)
{
  // camera information
  float fx = w / 2 / 0.5773502691896257;
  float fy = h / 2 / 0.41421356237309503;
  return df::PinholeCamera<T>(fx, fy, w/2, h/2, w, h);
}

inline
cv::Mat LoadPreprocessImage(std::string path,
                            const df::PinholeCamera<float>& in_cam,
                            const df::PinholeCamera<float>& out_cam)
{
  CHECK(in_cam.width() != 0 && in_cam.height() != 0) << "Invalid input camera provided";
  CHECK(out_cam.width() != 0 && out_cam.height() != 0) << "Invalid input camera provided";

  // load image
  cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
  CHECK(!img.empty()) << "Failed loading test image " << path;
  CHECK_EQ(img.type(), CV_8UC1);

  // make sure the intrinsics are scaled to input image size
  df::PinholeCamera<float> in_cam_resized(in_cam);
  in_cam_resized.ResizeViewport(img.cols, img.rows);

  // preprocess and return
  img.convertTo(img, CV_32FC1, 1/255.0);
  return df::ChangeIntrinsics(img, in_cam_resized, out_cam);
}

template <typename Derived, typename Scalar>
void CompareWithTol(const Eigen::DenseBase<Derived>& m1, const Eigen::DenseBase<Derived>& m2, Scalar tol)
{
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      EXPECT_NEAR(m1(i,j), m2(i,j), tol);
}

template <typename T>
Sophus::SE3<T> GetPerturbedPose(const Sophus::SE3<T>& se3, std::size_t idx, T eps)
{
  assert(idx >= 0 && idx < 6 && "invalid perturbation index");
  Sophus::SE3<T> pose_perturbed(se3);
  if (idx < 3)
  {
    pose_perturbed.translation()[idx] += eps;
  }
  else
  {
    Eigen::Matrix<T, 3, 1> omega = Eigen::Matrix<T, 3, 1>::Zero();
    omega(idx-3) = eps;
    pose_perturbed.so3() = Sophus::SE3<T>::SO3Type::exp(omega) * pose_perturbed.so3();
  }
  return pose_perturbed;
}

} // namespace df

#endif // DF_TESTING_UTILS_H_
