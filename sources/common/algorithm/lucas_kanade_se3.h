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
#ifndef DF_JACOBIANS_H_
#define DF_JACOBIANS_H_

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

#include "reduction_items.h"
#include "warping.h"
#include "m_estimators.h"

namespace df
{

/**
 * Full, per-pixel SE3 LucasKanade algorithm
 */
template <typename Scalar, typename Device,
          typename SE3T=Sophus::SE3<Scalar>,
          typename CamT=df::PinholeCamera<Scalar>,
          typename ReductionItem=df::JTJJrReductionItem<Scalar,6>,
          typename GradT=Eigen::Matrix<Scalar,1,2>>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
ReductionItem LucasKanadeSE3(std::size_t x, std::size_t y,
                             const SE3T& se3,
                             const CamT& cam,
                             const vc::Image2DView<Scalar, Device>& img0,
                             const vc::Image2DView<Scalar, Device>& img1,
                             const vc::Image2DView<Scalar, Device>& dpt0,
                             const vc::Image2DView<GradT, Device>& grad1,
                             const Scalar huber_delta)
{
  ReductionItem result;
  const Correspondence<Scalar> corresp = FindCorrespondence(x, y, dpt0(x,y), cam, se3);

  if (corresp.valid)
  {
    // calculate jacobians
    const Eigen::Matrix<Scalar,2,6> corresp_jac = FindCorrespondenceJacobianPose(corresp, dpt0(x,y), cam, se3);
    const Eigen::Matrix<Scalar,1,2> grad = grad1.template getBilinear<Eigen::Matrix<Scalar,1,2>>(corresp.pix1);
    Eigen::Matrix<Scalar,1,6> J = -grad * corresp_jac; // (1x6) = (1x2)x(2x6)

    // get the photometric error
    Scalar diff = static_cast<Scalar>(img0(x,y)) - img1.template getBilinear<Scalar>(corresp.pix1);

    // apply weighting
    const Scalar hbr_wgt = HuberWeight(diff, huber_delta);
    diff *= hbr_wgt;
    J *= hbr_wgt;
//    if (abs(diff) > huber_delta)
//      return result;

    // fill the ReductionItem
    result.inliers = 1;
    result.residual = diff * diff;
    result.Jtr = J.transpose() * diff;
    result.JtJ = ReductionItem::HessianType(J.transpose());
  }
  return result;
}

/*
 * Solves the normal equations system described by JtJ and Jtr
 * and updates the current pose estimate with the solution.
 * We split the translation and rotation for the update here.
 */
template <typename T>
void SE3SolveAndUpdate(const Eigen::Matrix<T,6,6>& JtJ, const Eigen::Matrix<T,6,1>& Jtr, Sophus::SE3<T>& curr_est)
{
  // solve for the update
  Eigen::Matrix<T,6,1> update = -JtJ.ldlt().solve(Jtr);

  // split and apply the update
  Eigen::Matrix<T,3,1> trs_update = update.template head<3>();
  Eigen::Matrix<T,3,1> rot_update = update.template tail<3>();
  curr_est.translation() += trs_update;
  curr_est.so3() = Sophus::SO3<T>::exp(rot_update) * curr_est.so3();
}

} // namespace df

#endif // DF_JACOBIANS_H_
