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
#ifndef DF_PINHOLE_CAMERA_H_
#define DF_PINHOLE_CAMERA_H_

#include <Eigen/Core>

namespace df
{

/**
 * Class representing a standard pinhole camera
 * Does all camera operations/jacobians
 * Works for float and double
 *
 * Reproject: Pixel, Depth -> Point
 * ReprojectPixelJacobian: Pixel, Depth -> Jacobian w.r.t pixel
 * ReprojectDepthJacobian: Pixel, Depth, -> Jacobian w.r.t depth
 *
 * Project: Point -> Pixel
 * ProjectPointJacobian: Point -> Jacobian w.r.t point
 *
 * PixelValid: Is pixel within the camera
 * ResizeViewport: Resize camera to given width, height
 *
 */
template <typename Scalar = float>
class PinholeCamera
{
public:
  using PointT = Eigen::Matrix<Scalar, 3, 1>;
  using PixelT = Eigen::Matrix<Scalar, 2, 1>;
  using PointTRef = Eigen::Ref<const PointT>;
  using PixelTRef = Eigen::Ref<const PixelT>;
  using Type = Scalar;

  EIGEN_DEVICE_FUNC PinholeCamera();
  EIGEN_DEVICE_FUNC PinholeCamera(Scalar fx, Scalar fy, Scalar u0, Scalar v0, Scalar width, Scalar height);
  EIGEN_DEVICE_FUNC ~PinholeCamera();

  /**
   * Project point onto the camera
   */
  inline EIGEN_DEVICE_FUNC PixelT Project(const PointTRef& point) const;

  /**
   * Jacobian of project w.r.t point
   */
  inline EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 2, 3> ProjectPointJacobian(const PointTRef& point) const;

  /**
   * reproject pixel to a certain depth
   */
  inline EIGEN_DEVICE_FUNC PointT Reproject(const PixelTRef& pixel, const Scalar& depth) const;

  /**
   * Jacobian of reproject w.r.t pixel
   */
  inline EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 3, 2> ReprojectPixelJacobian(const PixelTRef& pixel, Scalar depth) const;

  /**
   * Jacobian of reproject w.r.t pixel
   */
  inline EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 3, 1> ReprojectDepthJacobian(const PixelTRef& pixel, Scalar depth) const;


  /**
   * Check if pixel is inside camera
   */
  template <typename T = Scalar>
  inline EIGEN_DEVICE_FUNC bool PixelValid(T x, T y, std::size_t border = 0) const;

  /**
   * Check if pixel is inside camera
   */
  inline EIGEN_DEVICE_FUNC bool PixelValid(const PixelTRef& pixel, std::size_t border = 0) const;

  /**
   * Resize the camera
   */
  template <typename T = Scalar>
  inline EIGEN_DEVICE_FUNC void ResizeViewport(const T& new_width, const T& new_height);

  /**
   * Const accessors
   */
  inline EIGEN_DEVICE_FUNC const Scalar& fx() const { return fx_; }
  inline EIGEN_DEVICE_FUNC const Scalar& fy() const { return fy_; }
  inline EIGEN_DEVICE_FUNC const Scalar& u0() const { return u0_; }
  inline EIGEN_DEVICE_FUNC const Scalar& v0() const { return v0_; }
  inline EIGEN_DEVICE_FUNC const Scalar& width() const { return width_; }
  inline EIGEN_DEVICE_FUNC const Scalar& height() const { return height_; }

  template <typename T>
  inline Eigen::Matrix<T,3,3> Matrix() const;

  template <typename T>
  inline Eigen::Matrix<T,3,3> InverseMatrix() const;

  template <typename T>
  inline PinholeCamera<T> Cast() const;

  static PinholeCamera<Scalar> FromFile(const std::string& path);

private:
  Scalar fx_;
  Scalar fy_;
  Scalar u0_;
  Scalar v0_;
  Scalar width_;
  Scalar height_;
};

} // namespace df

template <typename T>
std::ostream& operator<<(std::ostream& os, const df::PinholeCamera<T>& cam)
{
  os << "PinholeCamera(" << "fx=" << cam.fx()
     << ", fy=" << cam.fy() << ", u0=" << cam.u0()
     << ", v0=" << cam.v0() << ", w=" << cam.width()
     << ", h=" << cam.height() << ")";
  return os;
}

#include "pinhole_camera_impl.h"

#endif // DF_PINHOLE_CAMERA_H_
