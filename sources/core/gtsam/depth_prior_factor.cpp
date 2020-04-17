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
#include <gtsam/linear/HessianFactor.h>

#include "depth_prior_factor.h"
#include "cu_depthaligner.h"
#include "cu_image_proc.h"
#include "warping.h"

namespace df
{

template <typename Scalar, int CS>
DepthPriorFactor<Scalar,CS>::DepthPriorFactor(const cv::Mat& dpt,
                                              const KeyframePtr& kf,
                                              const gtsam::Key& code_key,
                                              Scalar sigma,
                                              std::size_t pyrlevels,
                                              Scalar avgdpt,
                                              AlignerPtr aligner)
    : Base(gtsam::cref_list_of<1>(code_key)),
      code_key_(code_key),
      avgdpt_(avgdpt),
      sigma_(sigma),
      pyrlevels_(pyrlevels),
      kf_(kf),
      aligner_(aligner),
      pyr_tgtdpt_(pyrlevels, dpt.cols, dpt.rows)
{
  // upload dpt image to GPU
  vc::Image2DView<Scalar, vc::TargetHost> dpt_buf(dpt);
  pyr_tgtdpt_[0].copyFrom(dpt_buf);

  // create a pyramid
  for (std::size_t i = 1; i < pyrlevels_; ++i)
  {
    df::GaussianBlurDown(pyr_tgtdpt_[i-1], pyr_tgtdpt_[i]);
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
DepthPriorFactor<Scalar,CS>::~DepthPriorFactor() {}

/* ************************************************************************* */
template <typename Scalar, int CS>
double DepthPriorFactor<Scalar,CS>::error(const gtsam::Values& c) const
{
  if (this->active(c))
  {
    // get our values
    CodeT code = c.at<CodeT>(code_key_);

    // RunWarping to get the error
    auto res = RunAlignment(code);

    return 0.5 * res.residual / sigma_ / sigma_;
  }
  else
  {
    return 0.0;
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
boost::shared_ptr<gtsam::GaussianFactor>
DepthPriorFactor<Scalar,CS>::linearize(const gtsam::Values& c) const
{
  // Only linearize if the factor is active
  if (!this->active(c))
    return boost::shared_ptr<gtsam::HessianFactor>();

  // recover our values
  CodeT code = c.at<CodeT>(code_key_);

  // get the jacobians
  auto res = RunAlignment(code);

  // create and return a HessianFactor
  gtsam::Matrix JtJ = res.JtJ.toDenseMatrix().template cast<double>();
  gtsam::Vector Jtr = res.Jtr.template cast<double>();

  // integrate the uncertainty here
  JtJ /= sigma_ * sigma_;
  Jtr /= sigma_ * sigma_;
  return boost::make_shared<gtsam::HessianFactor>(code_key_, JtJ, -Jtr, (double)res.residual);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename DepthPriorFactor<Scalar,CS>::StepResult
DepthPriorFactor<Scalar,CS>::RunAlignment(const CodeT& code) const
{
  UpdateKfDepth(code);

  StepResult result;
  const Eigen::Matrix<Scalar,CS,1> cde = code.template cast<Scalar>();
  for (std::size_t i = 0; i < pyrlevels_; ++i)
  {
    // run step
    result += aligner_->RunStep(cde,
                                pyr_tgtdpt_[i],
                                kf_->pyr_prx_orig.GetGpuLevel(i),
                                kf_->pyr_jac.GetGpuLevel(i));
  }
  return result;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void DepthPriorFactor<Scalar,CS>::UpdateKfDepth(const CodeT& code) const
{
  const Eigen::Matrix<Scalar,CS,1> cde = code.template cast<Scalar>();
  for (std::size_t i = 0; i < pyrlevels_; ++i)
  {
    df::UpdateDepth(cde,
                       kf_->pyr_prx_orig.GetGpuLevel(i),
                       kf_->pyr_jac.GetGpuLevel(i),
                       avgdpt_,
                       kf_->pyr_dpt.GetGpuLevel(i));
  }
}

/* ************************************************************************* */
// explicit instantiation
template class DepthPriorFactor<float,DF_CODE_SIZE>;

} // namespace df
