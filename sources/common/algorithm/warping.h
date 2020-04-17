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
#ifndef DF_WARPING_H_
#define DF_WARPING_H_

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

#include "pinhole_camera.h"

namespace df
{

template <typename T, typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
T ProxToDepth(T prx, Scalar avg_dpt)
{
  return avg_dpt / prx - avg_dpt;
}

template <typename T, typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
T DepthToProx(T dpt, Scalar avg_dpt)
{
  return avg_dpt / (avg_dpt + dpt);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Scalar DepthJacobianPrx(Scalar dpt, Scalar avg_dpt)
{
  Scalar prx = avg_dpt / (avg_dpt + dpt);
  return -avg_dpt / (prx * prx);
}

template <typename CodeT, typename JacT, typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
T ProxFromCode(const Eigen::MatrixBase<CodeT>& code,
               const Eigen::MatrixBase<JacT>& prx_J_cde,
               T prx_0code)
{
  return prx_0code + (prx_J_cde * code)(0);
}

template <typename CodeT, typename JacT, typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
T DepthFromCode(const Eigen::MatrixBase<CodeT>& code,
                const Eigen::MatrixBase<JacT>& prx_J_cde,
                T prx_0code, T avg_dpt)
{
  T prx = ProxFromCode(code, prx_J_cde, prx_0code);
  return ProxToDepth(prx, avg_dpt);
}

template <typename T, typename CodeT, typename ImageBuffer=vc::Image2DManaged<T, vc::TargetHost>>
ImageBuffer RenderDpt(int width, int height, T avg_dpt,
                      const Eigen::MatrixBase<CodeT>& code,
                      const typename ImageBuffer::ViewT& code_jac,
                      const typename ImageBuffer::ViewT& prx_0code)
{
  int cs = code.rows();

  // render new depth
  ImageBuffer dpt(width, height);
  for (int x = 0; x < width; ++x)
  {
    for (int y = 0; y < height; ++y)
    {
      Eigen::Map<const Eigen::Matrix<T,-1,-1>> prx_J_cde(&code_jac(x*cs, y), 1, cs);
      dpt(x,y) = DepthFromCode(code, prx_J_cde, prx_0code(x,y), avg_dpt);
    }
  }

  return dpt;
}

/**
 * Takes two poses in the same reference frame and
 * returns pose1 expressed in frame pose0
 * pose_ab = (pose_a)^-1 * pose_b
 */
template <typename T, typename JacT=Eigen::Matrix<T,6,6>>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Sophus::SE3<T> RelativePose(const Sophus::SE3<T>& pose_a, const Sophus::SE3<T>& pose_b)
{
  return pose_a.inverse() * pose_b;
}

template <typename T, typename JacT=Eigen::Matrix<T,6,6>>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Sophus::SE3<T> RelativePose(const Sophus::SE3<T>& pose_a, const Sophus::SE3<T>& pose_b,
                            JacT& jac_a, JacT& jac_b)
{
  typedef Eigen::Matrix<T,3,3> Mat33;
  typedef Eigen::Matrix<T,3,1> Vec3;
  typedef typename Sophus::SE3<T> SE3T;

  SE3T pose_ab = RelativePose(pose_a, pose_b);

  Mat33 rot_a = pose_a.so3().matrix();
  Vec3 trs_a = pose_a.translation();
  Vec3 trs_b = pose_b.translation();

  // dRelPose / dtrs_a
  jac_a.template block<3,3>(0,0) = -rot_a.transpose();
  jac_a.template block<3,3>(3,0).setZero();

  // dRelpose / drot_a
  jac_a.template block<3,3>(0,3) = -SE3T::SO3Type::hat(rot_a.transpose() * (trs_a - trs_b)) * rot_a.transpose();
  jac_a.template block<3,3>(3,3) = -rot_a.transpose();

  // dRelPose / dtrs_b
  jac_b.template block<3,3>(0,0) = rot_a.transpose();
  jac_b.template block<3,3>(3,0).setZero();

  // dRelpose / drot_b
  jac_b.template block<3,3>(0,3).setZero();
  jac_b.template block<3,3>(3,3) = rot_a.transpose();

  return pose_ab;
}

template <typename T>
T PoseDistance(const Sophus::SE3<T>& pose_a, const Sophus::SE3<T>& pose_b,
               T trs_wgt=8.0, T rot_wgt=3.0)
{
  auto relpose = RelativePose(pose_a, pose_b);
  T drot = relpose.so3().log().head(2).norm(); // ignore the roll
  T dtrs = relpose.translation().norm();
  return dtrs * trs_wgt + drot * rot_wgt;
}

/**
 * transform jacobian w.r.t. the transform
 * T(x) = X = Rx + t
 * we use R3 x SO3 instead of SE3 here to decouple (t, R)
 * [dX/dt, dX/dR] = (3 x 6)
 * [I, -(Rx)^]
 */
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Eigen::Matrix<T,3,6> TransformJacobianPose(const Eigen::Matrix<T,3,1>& pt, const Sophus::SE3<T>& pose)
{
  Eigen::Matrix<T,3,6> dXdT;
  dXdT.template block<3,3>(0,0) = Eigen::Matrix<T,3,3>::Identity();
  dXdT.template block<3,3>(0,3) = -Sophus::SO3<T>::hat(pose.so3() * pt);
  return dXdT;
}

/**
 * transform jacobian w.r.t. input point
 * T(x) = X = Rx + t
 * dX/dx = (3 x 3)
 * [R]
 */
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Eigen::Matrix<T,3,3> TransformJacobianPoint(const Eigen::Matrix<T,3,1>& pt, const Sophus::SE3<T>& pose)
{
  return pose.so3().matrix();
}

/**
 * Structure containing information about the found correspondence,
 * as well as some byproducts. Members:
 *  pix0: source pixel
 *  pt: reprojected pixel to a point in space
 *  tpt: above point transformed into the second
 *  pix1: corresponding pixel in second camera
 *  valid: whether a correspondence has been found
 */
template <typename T>
struct Correspondence
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<T,2,1> pix0;
  Eigen::Matrix<T,3,1> pt;
  Eigen::Matrix<T,3,1> tpt;
  Eigen::Matrix<T,2,1> pix1;
  bool valid = false;
};

/**
 * Find correspondence between the current pixel (x,y)
 * and a camera at pose se3 using depth dpt
 */
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Correspondence<Scalar> FindCorrespondence(std::size_t x, std::size_t y, Scalar dpt,
                                          const df::PinholeCamera<Scalar>& cam,
                                          const Sophus::SE3<Scalar>& se3,
                                          int border = 1, Scalar min_dpt = 0, bool check_bounds = true)
{
  typedef Eigen::Matrix<Scalar,2,1> PixelT;
  typedef Eigen::Matrix<Scalar,3,1> PointT;

  const PixelT pix0(x,y);

  Correspondence<Scalar> corresp;
  corresp.valid = false;

  // reproject and transform point
  const PointT pt = cam.Reproject(pix0, dpt);
  const PointT tpt = se3 * pt;

  // check if depth is valid
  const Scalar depth = tpt[2];
  if (depth > min_dpt)
  {
    // project onto the second camera
    const Eigen::Matrix<Scalar,2,1> pix1 = cam.Project(tpt);

    // check if its inside camera
    if (cam.PixelValid(pix1, border) || !check_bounds)
    {
      corresp.pix0 = pix0;
      corresp.pix1 = pix1;
      corresp.pt = pt;
      corresp.tpt = tpt;
      corresp.valid = true;
    }
  }
  return corresp;
}

/**
 * Jacobian of the warping required to find correspondence
 * w.r.t the pose parameters (x,y,z,alpha,beta,gamma)
 */
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Eigen::Matrix<T,2,6> FindCorrespondenceJacobianPose(const Correspondence<T>& corresp,
                                                    const T dpt,
                                                    const df::PinholeCamera<T>& cam,
                                                    const Sophus::SE3<T>& se3)
{
  const Eigen::Matrix<T,3,6> dXdT = TransformJacobianPose<T>(corresp.pt, se3);
  const Eigen::Matrix<T,2,3> dCam = cam.ProjectPointJacobian(corresp.tpt);
  return dCam * dXdT;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
void FindCorrespondenceJacobianDepth(const Correspondence<T>& corresp,
                                     const T dpt,
                                     const df::PinholeCamera<T>& cam,
                                     const Sophus::SE3<T>& se3,
                                     Eigen::Matrix<T,2,1>& jac)
{
  // calculate jacobians
  const Eigen::Matrix<T,2,3> pix1_J_tpt = cam.ProjectPointJacobian(corresp.tpt);
  const Eigen::Matrix<T,3,3> tpt_J_pt = TransformJacobianPoint(corresp.pt, se3);
  const Eigen::Matrix<T,3,1> pt_J_dpt = cam.ReprojectDepthJacobian(corresp.pix0, dpt);
  jac.noalias() = pix1_J_tpt * tpt_J_pt * pt_J_dpt;
}

// calculate prx jacobian
template <typename T, typename CorrespJacT>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
void FindCorrespondenceJacobianPrx(const df::Correspondence<T>& corresp,
                                    const T& dpt,
                                    const df::PinholeCamera<T>& cam,
                                    const Sophus::SE3<T>& pose,
                                    const T& avg_dpt,
                                    Eigen::MatrixBase<CorrespJacT>& jac)
{
  EIGEN_STATIC_ASSERT_FIXED_SIZE(CorrespJacT);
  EIGEN_STATIC_ASSERT(CorrespJacT::RowsAtCompileTime == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  Eigen::Matrix<T,2,1> pix1_J_dpt;
  df::FindCorrespondenceJacobianDepth(corresp, dpt, cam, pose, pix1_J_dpt);
  const T dpt_J_prx = df::DepthJacobianPrx(dpt, avg_dpt);
  jac.noalias() = pix1_J_dpt * dpt_J_prx;
}

// calculate prx jacobian
template <typename T, typename PrxJacT, typename CorrespJacT>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
void FindCorrespondenceJacobianCode(const df::Correspondence<T>& corresp,
                                    const T& dpt,
                                    const df::PinholeCamera<T>& cam,
                                    const Sophus::SE3<T>& pose,
                                    const Eigen::MatrixBase<PrxJacT>& prx_J_cde,
                                    const T& avg_dpt,
                                    Eigen::MatrixBase<CorrespJacT>& jac)
{
  EIGEN_STATIC_ASSERT_FIXED_SIZE(PrxJacT);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(CorrespJacT);
  EIGEN_STATIC_ASSERT(PrxJacT::RowsAtCompileTime == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
  EIGEN_STATIC_ASSERT(CorrespJacT::RowsAtCompileTime == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  Eigen::Matrix<T,2,1> pix1_J_dpt;
  df::FindCorrespondenceJacobianDepth(corresp, dpt, cam, pose, pix1_J_dpt);
  const T dpt_J_prx = df::DepthJacobianPrx(dpt, avg_dpt);
  jac.noalias() = pix1_J_dpt * dpt_J_prx * prx_J_cde;
}

} // namespace df

#endif // DF_WARPING_H_
