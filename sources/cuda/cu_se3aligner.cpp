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

#include "cu_se3aligner.h"
#include "pinhole_camera.h"
#include "lucas_kanade_se3.h"
#include "launch_utils.h"
#include "kernel_utils.h"

namespace df
{

////////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, typename BaseT=SE3Aligner<Scalar>>
__global__ void kernel_step_calculate(const Sophus::SE3<Scalar> se3,
                                      const df::PinholeCamera<Scalar> cam,
                                      const typename BaseT::ImageBuffer img0,
                                      const typename BaseT::ImageBuffer img1,
                                      const typename BaseT::ImageBuffer dpt0,
                                      const typename BaseT::GradBuffer grad1,
                                      const Scalar huber_delta,
                                      vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
{
  typedef typename SE3Aligner<Scalar>::ReductionItem Item;
  Item sum;

  vc::runReductions(img0.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / img0.width();
    const unsigned int x = i - (y * img0.width());

    sum += LucasKanadeSE3(x, y, se3, cam, img0, img1, dpt0, grad1, huber_delta);
  });

  vc::finalizeReduction(bscratch.ptr(), &sum, Item::WarpReduceSum, Item());
}

template <typename Scalar, typename BaseT=SE3Aligner<Scalar>>
__global__ void kernel_warp_calculate(const Sophus::SE3<Scalar> se3,
                                      const df::PinholeCamera<Scalar> cam,
                                      const typename BaseT::ImageBuffer img0,
                                      const typename BaseT::ImageBuffer img1,
                                      const typename BaseT::ImageBuffer dpt0,
                                      typename BaseT::ImageBuffer img2,
                                      vc::Buffer1DView<typename BaseT::CorrespondenceItem, vc::TargetDeviceCUDA> bscratch)
{
  typedef typename BaseT::CorrespondenceItem Item;
  typedef typename df::PinholeCamera<Scalar>::PixelT PixelT;
  typedef typename df::PinholeCamera<Scalar>::PointT PointT;

  Item sum;

  // Produce data for each pixel
  // this uses grid-stride loop running your lambda
  vc::runReductions(img0.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / img0.width();
    const unsigned int x = i - (y * img0.width());

    img2(x,y) = 0;

    // reproject and transform point
    const PixelT pix0(x, y);
    const PointT pt = cam.Reproject(pix0, dpt0(x,y));
    const PointT tpt = se3 * pt;

    // check if depth is valid
    const Scalar depth = tpt[2];
    if (depth <= 0)
      return;

    // project onto the second camera
    const PixelT pix1 = cam.Project(tpt);
    if (cam.PixelValid(pix1, 1))
    {
      const Scalar sampled = img1.template getBilinear<Scalar>(pix1);

      // render img2
      img2(x,y) = sampled;

      // fill reduction item
      sum.inliers += 1;
      sum.residual += static_cast<Scalar>(img0(x,y)) - sampled;
    }
  });

  // This reduces the blocks using kepler warp shuffle
  // and stores per-block results in the scratchpad buffer
  vc::finalizeReduction(bscratch.ptr(), &sum, Item::WarpReduceSum, Item());
}

////////////////////////////////////////////////////////////////////////////////////
// SE3Aligner class definition
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
SE3Aligner<Scalar>::SE3Aligner() : max_blocks(1024), bscratch1_(max_blocks), bscratch2_(max_blocks) {}

template <typename Scalar>
SE3Aligner<Scalar>::~SE3Aligner() {}

template <typename Scalar>
typename SE3Aligner<Scalar>::CorrespondenceItem
SE3Aligner<Scalar>::Warp(const Sophus::SE3<Scalar>& se3,
                         const df::PinholeCamera<Scalar>& cam,
                         const ImageBuffer& img0,
                         const ImageBuffer& img1,
                         const ImageBuffer& dpt0,
                         ImageBuffer& img2)
{
  CorrespondenceItem result;

  // TODO: change to something standarized
  int threads = 256;
  int blocks = std::min(static_cast<int>((img0.area() + threads - 1) / threads), max_blocks);
  const int smemSize = (threads / 32 + 1) * sizeof(SE3Aligner::CorrespondenceItem);

  kernel_warp_calculate<<<blocks, threads, smemSize>>>(se3, cam, img0, img1, dpt0, img2, bscratch1_);
  CudaCheckLastError("[SE3Aligner::Warp] Kernel launch failed (kernel_warp_calculate)");

  kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch1_, bscratch1_, blocks);
  CudaCheckLastError("[SE3Aligner::Warp] Kernel launch failed (kernel_finalize_reduction)");

  // copy result back
  auto tmpbuf = vc::Buffer1DView<CorrespondenceItem, vc::TargetHost>(&result, 1);
  tmpbuf.copyFrom(bscratch1_);
  return result;
}

template <typename Scalar>
typename SE3Aligner<Scalar>::ReductionItem
SE3Aligner<Scalar>::RunStep(const Sophus::SE3<Scalar>& se3,
                            const df::PinholeCamera<Scalar>& cam,
                            const ImageBuffer& img0,
                            const ImageBuffer& img1,
                            const ImageBuffer& dpt0,
                            const GradBuffer& grad1)
{
  ReductionItem result;

  // TODO: change to something standarized
  int threads = 256;
  int blocks = std::min(static_cast<int>((img0.area() + threads - 1) / threads), max_blocks);
  const int smemSize = (threads / 32 + 1) * sizeof(SE3Aligner::ReductionItem); // need one element in smem per warp

  kernel_step_calculate<<<blocks, threads, smemSize>>>(se3, cam, img0, img1, dpt0, grad1, huber_delta_, bscratch2_);
  kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch2_, bscratch2_, blocks);
  CudaCheckLastError("[SE3Aligner::RunStep] Kernel launch failed");

  auto tmpbuf = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
  tmpbuf.copyFrom(bscratch2_);
  return result;
}

// explicitly instantiate for float and double
template class SE3Aligner<float>;

} // namespace df
