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
#ifndef DF_DENSE_SFM_H_
#define DF_DENSE_SFM_H_

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Math/LossFunctions.hpp>

#include "warping.h"
#include "reduction_items.h"
#include "kernel_utils.h"
#include "m_estimators.h"

#include <typeinfo>

namespace df
{

struct DenseSfmParams
{
  float huber_delta = 0.1f;
  float ocl_th = 1000; // disabled by default
  float avg_dpt = 2.0f;
  float min_dpt = 0.0f;
  int valid_border = 2;
};

template <typename Scalar>
EIGEN_DEVICE_FUNC
Scalar DenseSfm_RobustLoss(Scalar x, Scalar delta)
{
  return HuberWeight(x, delta);
}

/*
 * Return a weight that has to be applied to both the jacobian and residual
 * and that is calculated from the network parametrized uncertainty
 */
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Scalar DenseSfm_UncertaintyWeight(Scalar logb, Scalar err_J_prx)
{
  // calculate uncertainty of the photo error
  const Scalar sigma_prx = sqrtf(2) * expf(logb);
  const Scalar sigma_err = fabsf(err_J_prx) * sigma_prx;
  const Scalar sigma_err_norm = fmaxf(sigma_err * Scalar(1000), Scalar(1));

//   printf("sigma_err = %f\n", sigma_err);
  return Scalar(1.0); //sigma_err_norm;
}

/*
 * Full SE3 LucasKanade algorithm with code depth
 */
template <typename Scalar, int CS,
          typename Device=vc::TargetDeviceCUDA,
          typename SE3T=Sophus::SE3<Scalar>,
          typename ImageGrad=Eigen::Matrix<Scalar,1,2>,
          typename GradBuffer=vc::Image2DView<ImageGrad,Device>,
          typename ReductionItem=df::CorrespondenceReductionItem<Scalar>>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
void DenseSfm_EvaluateError(std::size_t x, std::size_t y,
                            const SE3T& pose_10,
                            const df::PinholeCamera<Scalar>& cam,
                            const vc::Image2DView<Scalar,Device>& img0,
                            const vc::Image2DView<Scalar,Device>& img1,
                            const vc::Image2DView<Scalar,Device>& dpt0,
                            const vc::Image2DView<Scalar,Device>& std0,
                            const GradBuffer& grad1,
                            const DenseSfmParams& params,
                            ReductionItem& result)
{
  // find the correspondence
  const Correspondence<Scalar> corresp = FindCorrespondence(x, y, dpt0(x,y), cam, pose_10);
  if (corresp.valid)
  {
    // get the photometric error
    Scalar diff = static_cast<Scalar>(img0(x,y)) - img1.template getBilinear<Scalar>(corresp.pix1);

    // huber weighting
    const Scalar hbr_wgt = DenseSfm_RobustLoss(diff, params.huber_delta);

    // calculate err_J_prx to propagate uncertainty
    Eigen::Matrix<Scalar,2,1> pix1_J_prx;
    FindCorrespondenceJacobianPrx(corresp, dpt0(x,y), cam, pose_10, params.avg_dpt, pix1_J_prx);
    const ImageGrad grad = grad1.template getBilinear<ImageGrad>(corresp.pix1);
    const Scalar err_J_prx = -(grad * pix1_J_prx)(0,0);

    // calculate uncertainty of the photo error
    const Scalar sigma_wgt = DenseSfm_UncertaintyWeight(std0(x,y), err_J_prx);

    // get the weighting
    const Scalar total_wgt = hbr_wgt * sigma_wgt;

    // weight stuff
    diff *= total_wgt;

    // fill the ReductionItem
    result.inliers += 1;
    result.residual += diff * diff;
  }
}

 /*
  * Full SE3 LucasKanade algorithm with code depth
  */
template <typename Scalar, int CS,
          typename Device=vc::TargetDeviceCUDA,
          typename SE3T=Sophus::SE3<Scalar>,
          typename RelposeJac=Eigen::Matrix<Scalar,SE3T::DoF,SE3T::DoF>,
          typename ImageBuffer=vc::Image2DView<Scalar,Device>,
          typename ImageGrad=Eigen::Matrix<Scalar,1,2>,
          typename GradBuffer=vc::Image2DView<ImageGrad,Device>,
          typename ReductionItem=df::JTJJrReductionItem<Scalar, 2*SE3T::DoF+CS>>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
void DenseSfm(std::size_t x, std::size_t y,
              const SE3T& pose_10,
              const RelposeJac& pose10_J_pose0,
              const RelposeJac& pose10_J_pose1,
              const Eigen::Matrix<Scalar,CS,1>& code,
              const df::PinholeCamera<Scalar>& cam,
              const vc::Image2DView<Scalar,Device>& img0,
              const vc::Image2DView<Scalar,Device>& img1,
              const vc::Image2DView<Scalar,Device>& dpt0,
              const vc::Image2DView<Scalar,Device>& std0,
              vc::Image2DView<Scalar,Device>& valid0,
              const vc::Image2DView<Scalar,Device>& prx0_jac,
              const GradBuffer& grad1,
              const DenseSfmParams& params,
              ReductionItem& result)
{
  // get code jacobian at pixel x,y
  Eigen::Map<const Eigen::Matrix<Scalar,1,CS>> tmp(&prx0_jac(x*CS,y));
  const Eigen::Matrix<Scalar,1,CS> prx_J_cde(tmp); // force copy

  // find the correspondence
  const Correspondence<Scalar> corresp = FindCorrespondence(x, y, dpt0(x,y), cam, pose_10,
                                                            params.valid_border, params.min_dpt);
  const Eigen::Matrix<Scalar,2,6> corresp_J_pose10 =
      FindCorrespondenceJacobianPose(corresp, dpt0(x,y), cam, pose_10);

  if (corresp.valid)
  {
    valid0(x,y) = 1.0f;

    // [dErr/dPose0, dErr/dPose1, dErr/dCode0] = 1x(6+6+CS)
    Eigen::Matrix<Scalar,1,12+CS> J;

    // calculate pose jacobian
    const ImageGrad grad = grad1.template getBilinear<ImageGrad>(corresp.pix1);
    J.template block<1,6>(0,0) = -grad * corresp_J_pose10 * pose10_J_pose0; // (1x6) = (1x2)x(2x6)x(6x6)
    J.template block<1,6>(0,6) = -grad * corresp_J_pose10 * pose10_J_pose1; // (1x6) = (1x2)x(2x6)x(6x6)

    // calculate prx jacobian
    Eigen::Matrix<Scalar,2,1> pix1_J_prx;
    FindCorrespondenceJacobianPrx(corresp, dpt0(x,y), cam, pose_10, params.avg_dpt, pix1_J_prx);
    const Scalar err_J_prx = -(grad * pix1_J_prx)(0,0);

    // calculate cde jacobian
    J.template block<1,CS>(0,12) = err_J_prx * prx_J_cde;

    // get the photometric error
    Scalar diff = static_cast<Scalar>(img0(x,y)) - img1.template getBilinear<Scalar>(corresp.pix1);

    // huber weighting
    const Scalar hbr_wgt = DenseSfm_RobustLoss(diff, params.huber_delta);

    // calculate uncertainty of the photo error
    const Scalar sigma_wgt = DenseSfm_UncertaintyWeight(std0(x,y), err_J_prx);

    // get the weighting
    const Scalar total_wgt = hbr_wgt * sigma_wgt;

    // weight stuff
    J *= total_wgt;
    diff *= total_wgt;

    // fill the ReductionItem
    result.inliers += 1;
    result.residual +=  diff * diff;
    result.Jtr += J.transpose() * diff;
    result.JtJ += typename ReductionItem::HessianType(J.transpose());
  }
}

} // namespace df

#endif  // DF_DENSE_SFM_H_
