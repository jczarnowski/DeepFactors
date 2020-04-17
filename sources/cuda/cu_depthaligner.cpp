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
#include <glog/logging.h>

#include "cu_depthaligner.h"
#include "launch_utils.h"
#include "kernel_utils.h"
#include "warping.h"

namespace df
{

////////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int CS, typename BaseT=DepthAligner<Scalar,CS>>
__global__ /*__launch_bounds__(64, 6)*/
void kernel_depthaligner_run_step(const typename BaseT::CodeT code,
                                  const typename BaseT::ImageBuffer tgt_dpt,
                                  const typename BaseT::ImageBuffer prx_orig,
                                  const typename BaseT::ImageBuffer prx_jac,
                                  vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
{
  typedef typename DepthAligner<Scalar, CS>::ReductionItem Item;
  Item sum;

  // TODO: cba to parametrize the 2.0
  Scalar avg_dpt = Scalar(2);

  vc::runReductions(tgt_dpt.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / tgt_dpt.width();
    const unsigned int x = i - (y * tgt_dpt.width());

    // calculate current depth with code and dpt0_orig and jacobian
    // prx0 = prx0_orig + prx0_jac * code
    // and convert prx0 to dpt0
    Eigen::Map<const Eigen::Matrix<Scalar,1,CS>> tmp(&prx_jac(x*CS,y));
    const Eigen::Matrix<Scalar,1,CS> prx_J_cde(tmp); // force copy
    Scalar dpt = df::DepthFromCode(code, prx_J_cde, prx_orig(x,y), avg_dpt);

    // calculate error
    Scalar diff = tgt_dpt(x,y) - dpt;

    // calculate jacobians
    Eigen::Matrix<Scalar,1,CS> J = -2 * abs(diff) * df::DepthJacobianPrx(dpt, avg_dpt) * prx_J_cde;

    sum.inliers += 1;
    sum.residual += diff * diff;
    sum.Jtr += J.transpose() * diff;
    sum.JtJ += Item::HessianType(J.transpose());
  });

  vc::finalizeReduction(bscratch.ptr(), &sum, Item::WarpReduceSum, Item());
}

////////////////////////////////////////////////////////////////////////////////////
// DepthAligner class definition
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int CS>
DepthAligner<Scalar, CS>::DepthAligner() : bscratch_(1024) {}

template <typename Scalar, int CS>
DepthAligner<Scalar, CS>::~DepthAligner() {}

template <typename Scalar, int CS>
typename DepthAligner<Scalar, CS>::ReductionItem
DepthAligner<Scalar, CS>::RunStep(const CodeT& code,
                                  const ImageBuffer& target_dpt,
                                  const ImageBuffer& prx_orig,
                                  const ImageBuffer& prx_jac)
{
  int codesize = prx_jac.width() / target_dpt.width();
  CHECK_EQ(codesize, CS) << "DepthAligner used with a different code size than it was compiled for";

  // These I have tuned manually for now
  // NOTE: threads must be a multiple of 32!
  int threads = 32;
  int blocks = 40;

  // impose the limit on blocks
  // max blocks has been carefully selected for the warp shuffle
  // TODO: move max blocks somewhere up in the class
  const int max_blocks = 1024;
  blocks = min(blocks, max_blocks);
  const int smemSize = (threads / 32) * sizeof(ReductionItem);

  kernel_depthaligner_run_step<Scalar, CS><<<blocks, threads, smemSize>>>(code, target_dpt, prx_orig, prx_jac, bscratch_);
  kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch_, bscratch_, blocks);
  CudaCheckLastError("[DepthAligner::RunStep] kernel launch failed");

  ReductionItem result;
  auto tmpbuf = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
  tmpbuf.copyFrom(bscratch_);
  return result;
}

////////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation
////////////////////////////////////////////////////////////////////////////////////
template class DepthAligner<float, DF_CODE_SIZE>;

} // namespace df


