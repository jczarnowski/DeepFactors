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

#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/Buffers/Reductions.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

#include "cu_sfmaligner.h"
#include "pinhole_camera.h"
#include "launch_utils.h"
#include "kernel_utils.h"
#include "dense_sfm.h"
#include "cuda_context.h"

namespace df
{

__constant__ DenseSfmParams sfm_params;

////////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int CS, typename BaseT=SfmAligner<Scalar,CS>>
__global__ /*__launch_bounds__(64, 6)*/
void kernel_step_calculate(
    const typename BaseT::SE3T pose_10,
    const typename BaseT::RelposeJac pose10_J_pose0,
    const typename BaseT::RelposeJac pose10_J_pose1,
    const typename BaseT::CodeT code,
    const df::PinholeCamera<Scalar> cam,
    const typename BaseT::ImageBuffer img0,
    const typename BaseT::ImageBuffer img1,
    const typename BaseT::ImageBuffer dpt0,
    const typename BaseT::ImageBuffer std0,
    typename BaseT::ImageBuffer valid0,
    const typename BaseT::ImageBuffer prx0_jac,
    const typename BaseT::GradBuffer grad1,
    vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
{
  typedef typename SfmAligner<Scalar, CS>::ReductionItem Item;
  Item sum;

  vc::runReductions(img0.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / img0.width();
    const unsigned int x = i - (y * img0.width());
    DenseSfm<Scalar,CS>(x, y, pose_10, pose10_J_pose0, pose10_J_pose1,
                        code, cam, img0, img1, dpt0, std0, valid0,
                        prx0_jac, grad1, sfm_params, sum);
  });

  vc::finalizeReduction(bscratch.ptr(), &sum, Item::WarpReduceSum, Item());
}

template <typename Scalar, int CS, typename BaseT=SfmAligner<Scalar,CS>>
__global__
void kernel_evaluate_error(
    const typename BaseT::SE3T pose_10,
    const df::PinholeCamera<Scalar> cam,
    const typename BaseT::ImageBuffer img0,
    const typename BaseT::ImageBuffer img1,
    const typename BaseT::ImageBuffer dpt0,
    const typename BaseT::ImageBuffer std0,
    const typename BaseT::GradBuffer grad1,
    vc::Buffer1DView<typename BaseT::ErrorReductionItem, vc::TargetDeviceCUDA> bscratch)
{
  typedef typename SfmAligner<Scalar, CS>::ErrorReductionItem Item;
  Item sum;

  vc::runReductions(img0.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / img0.width();
    const unsigned int x = i - (y * img0.width());

    DenseSfm_EvaluateError<Scalar,CS>(x, y, pose_10, cam, img0, img1,
                                      dpt0, std0, grad1, sfm_params, sum);
  });

  vc::finalizeReduction(bscratch.ptr(), &sum, Item::WarpReduceSum, Item());
}

////////////////////////////////////////////////////////////////////////////////////
// SfmAligner class definition
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int CS>
SfmAligner<Scalar, CS>::SfmAligner(SfmAlignerParams params)
    : params_(params), bscratch_(1024), bscratch2_(1024)
{
  SetEvalThreadsBlocks(params.eval_threads, params.eval_blocks);
  SetStepThreadsBlocks(params.step_threads, params.step_blocks);

  // upload parameters as a constant
  cudaMemcpyToSymbol(sfm_params, &params.sfmparams, sizeof(DenseSfmParams));
  CudaCheckLastError("Copying sfm parameters to gpu failed");

  devinfo_ = cuda::GetCurrentDeviceInfo();
}

template <typename Scalar, int CS>
SfmAligner<Scalar, CS>::~SfmAligner() {}

template <typename Scalar, int CS>
typename SfmAligner<Scalar, CS>::ErrorReductionItem
SfmAligner<Scalar, CS>::EvaluateError(const Sophus::SE3<Scalar>& pose0,
                                      const Sophus::SE3<Scalar>& pose1,
                                      const df::PinholeCamera<Scalar>& cam,
                                      const ImageBuffer& img0,
                                      const ImageBuffer& img1,
                                      const ImageBuffer& dpt0,
                                      const ImageBuffer& std0,
                                      const GradBuffer& grad1)
{
  // calculate relative pose and its jacobian
  const SE3T pose_10 = RelativePose(pose1, pose0);

  int threads = params_.eval_threads;
  int blocks = min(params_.eval_blocks, max_blocks);
  const int smemSize = (threads / devinfo_.WarpSize) * sizeof(ErrorReductionItem);

  kernel_evaluate_error<Scalar, CS><<<blocks, threads, smemSize>>>(pose_10, cam, img0, img1,
                                                                   dpt0, std0, grad1, bscratch2_);
  kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch2_, bscratch2_, blocks);
  CudaCheckLastError("[SfmAligner::EvaluateError] kernel launch failed");

  ErrorReductionItem result;
  auto tmpbuf = vc::Buffer1DView<ErrorReductionItem, vc::TargetHost>(&result, 1);
  tmpbuf.copyFrom(bscratch2_);
  return result;
}

template <typename Scalar, int CS>
typename SfmAligner<Scalar, CS>::ReductionItem
SfmAligner<Scalar, CS>::RunStep(const Sophus::SE3<Scalar>& pose0,
                                const Sophus::SE3<Scalar>& pose1,
                                const CodeT& code0,
                                const df::PinholeCamera<Scalar>& cam,
                                const ImageBuffer& img0,
                                const ImageBuffer& img1,
                                const ImageBuffer& dpt0,
                                const ImageBuffer& std0,
                                ImageBuffer& valid0,
                                const ImageBuffer& prx0_jac,
                                const GradBuffer& grad1)
{
  // calculate relative pose and its jacobian
  RelposeJac pose10_J_pose0;
  RelposeJac pose10_J_pose1;
  const SE3T pose_10 = RelativePose(pose1, pose0, pose10_J_pose1, pose10_J_pose0);

  int threads = params_.step_threads;
  int blocks = min(params_.step_blocks, max_blocks);
  const int smemSize = (threads / 32) * sizeof(ReductionItem);
  if (smemSize > devinfo_.SharedMemPerBlock)
    throw std::runtime_error("Too much shared memory per block requested (" + std::to_string(smemSize)
                             + " vs " + std::to_string(devinfo_.SharedMemPerBlock) + ")");

  kernel_step_calculate<Scalar, CS><<<blocks, threads, smemSize>>>(pose_10, pose10_J_pose0, pose10_J_pose1,
                                                                   code0, cam, img0, img1, dpt0, std0,
                                                                   valid0, prx0_jac, grad1, bscratch_);
  kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch_, bscratch_, blocks);
  CudaCheckLastError("[SfmAligner::RunStep] kernel launch failed");

  ReductionItem result;
  auto tmpbuf = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
  tmpbuf.copyFrom(bscratch_);
  return result;
}

template <typename Scalar, int CS>
void SfmAligner<Scalar, CS>::SetEvalThreadsBlocks(int threads, int blocks)
{
  CHECK_EQ(threads % 32,0) << "threads must be a multiple of 32!";
  CHECK_LE(blocks, max_blocks) << "blocks must be less than " << max_blocks;
  params_.eval_threads = threads;
  params_.eval_blocks = blocks;
}

template <typename Scalar, int CS>
void SfmAligner<Scalar, CS>::SetStepThreadsBlocks(int threads, int blocks)
{
  CHECK_EQ(threads % 32,0) << "threads must be a multiple of 32!";
  CHECK_LE(blocks, max_blocks) << "blocks must be less than " << max_blocks;
  params_.step_threads = threads;
  params_.step_blocks = blocks;
}

// explicitly instantiate for float/double and various code sizes
#define SPECIALIZE_SFMALIGNER_FOR(type, cs) \
  template class SfmAligner<type, cs>;

SPECIALIZE_SFMALIGNER_FOR(float, 32)
//SPECIALIZE_SFMALIGNER_FOR(float, 64)
//SPECIALIZE_SFMALIGNER_FOR(float, 128)

} // namespace df
