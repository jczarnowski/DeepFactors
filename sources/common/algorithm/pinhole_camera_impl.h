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
#ifndef DF_PINHOLE_CAMERA_IMPL_H_
#define DF_PINHOLE_CAMERA_IMPL_H_

#include <opencv2/opencv.hpp>

namespace df
{

template <typename Scalar>
PinholeCamera<Scalar>::PinholeCamera() : PinholeCamera(0,0,0,0,0,0) {}

template <typename Scalar>
PinholeCamera<Scalar>::PinholeCamera(Scalar fx, Scalar fy, Scalar u0, Scalar v0, Scalar width, Scalar height)
    : fx_(fx), fy_(fy), u0_(u0), v0_(v0), width_(width), height_(height) {}

template <typename Scalar>
PinholeCamera<Scalar>::~PinholeCamera() {}

/**
 * Project point onto the camera
 */
template <typename Scalar>
inline
typename PinholeCamera<Scalar>::PixelT PinholeCamera<Scalar>::Project(const PointTRef& point) const
{
  return PixelT(fx_ * point[0] / point[2] + u0_,
                fy_ * point[1] / point[2] + v0_);
}

/**
  * reproject pixel to a certain depth
  */
template <typename Scalar>
inline
EIGEN_DEVICE_FUNC typename PinholeCamera<Scalar>::PointT PinholeCamera<Scalar>::Reproject(const PixelTRef& pixel, const Scalar& depth) const
{
  PointT point((pixel[0] - u0_) / fx_, (pixel[1] - v0_) / fy_, 1);
  return point * depth;
}

/**
  * Jacobian of reproject w.r.t pixel
  */
template <typename Scalar>
inline
Eigen::Matrix<Scalar, 3, 2> PinholeCamera<Scalar>::ReprojectPixelJacobian(const PixelTRef& pixel, Scalar depth) const
{
  Eigen::Matrix<Scalar, 3, 2> jac;
  jac << depth / fx_,           0,
                   0, depth / fy_,
                   0,           0;
  return jac;
}

/**
  * Jacobian of reproject w.r.t pixel
  */
template <typename Scalar>
inline
Eigen::Matrix<Scalar, 3, 1> PinholeCamera<Scalar>::ReprojectDepthJacobian(const PixelTRef& pixel, Scalar depth) const
{
  Eigen::Matrix<Scalar, 3, 1> jac;
  jac << (pixel(0) - u0_) / fx_,
         (pixel(1) - v0_) / fy_,
         1;
  return jac;
}

/**
  * Jacobian of project w.r.t point
  */
template <typename Scalar>
inline
Eigen::Matrix<Scalar, 2, 3> PinholeCamera<Scalar>::ProjectPointJacobian(const PointTRef& point) const
{
  Eigen::Matrix<Scalar,2,3> jac;
  jac << fx_ / point[2],               0,   -(fx_ * point[0]) / point[2] / point[2],
                      0,  fy_ / point[2],   -(fy_ * point[1]) / point[2] / point[2];
  return jac;
}

/**
  * Check if pixel is inside camera
  */
template <typename Scalar>
template <typename T>
inline
bool PinholeCamera<Scalar>::PixelValid(T x, T y, std::size_t border) const
{
  return x >= border && x < width_ - border && y >= border && y < height_ - border;
}

/**
  * Check if pixel is inside camera
  */
template <typename Scalar>
inline
bool PinholeCamera<Scalar>::PixelValid(const PixelTRef& pixel, std::size_t border) const
{
  return PixelValid(pixel[0], pixel[1], border);
}

/**
  * Resize the camera
  */
template <typename Scalar>
template <typename T>
inline
void PinholeCamera<Scalar>::ResizeViewport(const T& new_width, const T& new_height)
{
  const Scalar x_ratio = new_width / width();
  const Scalar y_ratio = new_height / height();
  fx_ *= x_ratio;
  fy_ *= y_ratio;
  u0_ *= x_ratio;
  v0_ *= y_ratio;
  width_ = new_width;
  height_ = new_height;
}

template <typename Scalar>
PinholeCamera<Scalar> PinholeCamera<Scalar>::FromFile(const std::string& path)
{
  cv::FileStorage fs(path, cv::FileStorage::READ);

  if (!fs.isOpened())
    throw std::runtime_error("Failed to open file: " + path);

  int width, height;
  fs["image_width"] >> width;
  fs["image_height"] >> height;

  cv::Mat K;
  fs["camera_matrix"] >> K;

  PinholeCamera<Scalar> cam(K.at<double>(0,0), K.at<double>(1,1), K.at<double>(0,2), K.at<double>(1,2), width, height);
  return cam;
}

template <typename Scalar>
template <typename T>
inline Eigen::Matrix<T,3,3> PinholeCamera<Scalar>::Matrix() const
{
  Eigen::Matrix<T,3,3> K = Eigen::Matrix<T,3,3>::Identity();
  K(0,0) = fx();
  K(1,1) = fy();
  K(0,2) = u0();
  K(1,2) = v0();
  return K;
}

template <typename Scalar>
template <typename T>
inline Eigen::Matrix<T,3,3> PinholeCamera<Scalar>::InverseMatrix() const
{
  Eigen::Matrix<T,3,3> Kinv = Eigen::Matrix<T,3,3>::Identity();
  Kinv(0,0) = 1.0 / fx();
  Kinv(1,1) = 1.0 / fy();
  Kinv(0,2) = -u0() / fx();
  Kinv(1,2) = -v0() / fy();
  return Kinv;
}

template <typename Scalar>
template <typename T>
inline PinholeCamera<T> PinholeCamera<Scalar>::Cast() const
{
  return df::PinholeCamera<T>(T(fx_), T(fy_), T(u0_), T(v0_), width_, height_);
}

} // namespace df

#endif // DF_PINHOLE_CAMERA_IMPL_H_
