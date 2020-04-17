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
#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/Buffers/Reductions.hpp>
#include <Eigen/Core>

#include "cu_image_proc.h"
#include "launch_utils.h"
#include "kernel_utils.h"
#include "warping.h"

namespace df
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel gradients
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
static inline void SetSobelCoefficients(Eigen::DenseBase<Derived>& mx,
                                        Eigen::DenseBase<Derived>& my)
{
  typedef typename Eigen::DenseBase<Derived>::Scalar Scalar;

  // NOTE: Canonical Sobel kernels
  mx << Scalar(-1.0), Scalar(0.0), Scalar(1.0),
        Scalar(-2.0), Scalar(0.0), Scalar(2.0),
        Scalar(-1.0), Scalar(0.0), Scalar(1.0);

  my << Scalar(-1.0), Scalar(-2.0), Scalar(-1.0),
        Scalar(0.0),  Scalar(0.0),  Scalar(0.0),
        Scalar(1.0),  Scalar(2.0),  Scalar(1.0);
}

struct SobelCoeffs
{
  float X[9];
  float Y[9];
};
__constant__ SobelCoeffs SC;

template<typename PixelT, typename TG>
__global__ void kernel_sobel_gradients(const vc::Buffer2DView<PixelT,vc::TargetDeviceCUDA> img,
                                       vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA> grad)
{
  typedef Eigen::Matrix<TG,1,2> DerivT;

  // current point
  const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

  Eigen::Map<const Eigen::Matrix<float,3,3>> mx(SC.X);
  Eigen::Map<const Eigen::Matrix<float,3,3>> my(SC.Y);

  if(img.inBounds(x,y) && grad.inBounds(x,y))
  {
    DerivT& out = grad(x,y);

    TG sum_dx = 0;
    TG sum_dy = 0;

    for(int py = -1 ; py <= 1 ; ++py)
    {
      for(int px = -1 ; px <= 1 ; ++px)
      {
        const PixelT pix = img.getWithClampedRange((int)x + px,(int)y + py);
        const float& kv_dx = mx(1 + py, 1 + px);
        const float& kv_dy = my(1 + py, 1 + px);
        sum_dx += (pix * kv_dx);
        sum_dy += (pix * kv_dy);
      }
    }

    out << sum_dx , sum_dy;
    out /= 8;
  }
}

template<typename T, typename TG>
void SobelGradients(const vc::Buffer2DView<T,vc::TargetDeviceCUDA>& img,
                    vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA>& grad)
{
  // get a gaussian kernel in the gpu
  SobelCoeffs coeffs;
  Eigen::Map<Eigen::Matrix<float,3,3>> mx(coeffs.X);
  Eigen::Map<Eigen::Matrix<float,3,3>> my(coeffs.Y);
  SetSobelCoefficients(mx, my);
  cudaMemcpyToSymbol(SC, &coeffs, sizeof(SobelCoeffs));
  CudaCheckLastError("cudaMemcpyToSymbol failed");

  // calculate blocks and threads
  dim3 threads, blocks;
  vc::InitDimFromBufferOver(threads, blocks, grad);

  // run kernel
  kernel_sobel_gradients<<<blocks,threads>>>(img, grad);
  CudaCheckLastError("Kernel launch failed (kernel_sobel_gradients)");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Blur Downsampling
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
static inline void SetGaussCoefficients(Eigen::DenseBase<Derived>& gaussKernel)
{
  typedef typename Eigen::DenseBase<Derived>::Scalar Scalar;

  gaussKernel << Scalar(1.0), Scalar(4.0),  Scalar(6.0),  Scalar(4.0),  Scalar(1.0),
                 Scalar(4.0), Scalar(16.0), Scalar(24.0), Scalar(16.0), Scalar(4.0),
                 Scalar(6.0), Scalar(24.0), Scalar(36.0), Scalar(24.0), Scalar(6.0),
                 Scalar(4.0), Scalar(16.0), Scalar(24.0), Scalar(16.0), Scalar(4.0),
                 Scalar(1.0), Scalar(4.0),  Scalar(6.0),  Scalar(4.0),  Scalar(1.0);
}


__constant__ float gauss_coeffs[25];

template<typename Scalar>
__global__ void kernel_gaussian_blur_down(const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> in,
                                          vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> out)
{
  // current point
  const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

  Eigen::Map<const Eigen::Matrix<float,5,5>> kernel(gauss_coeffs);
  static constexpr int kdim = 5;

  if(out.inBounds(x,y))
  {
    Scalar sum = 0.0;
    Scalar wall = 0.0;

    for(int py = 0; py < kdim; ++py)
    {
      for(int px = 0; px < kdim; ++px)
      {
        const int nx = clamp((int) (2 * x + px - kdim / 2), 0, (int)in.width() - 1);
        const int ny = clamp((int) (2 * y + py - kdim / 2), 0, (int)in.height() - 1);
        const Scalar pix = in(nx,ny);
        sum += (pix * kernel(px,py));
        wall += kernel(px,py);
      }
    }

    out(x,y) = sum / wall;
  }
}

template <typename T>
void GaussianBlurDown(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& in,
                      vc::Buffer2DView<T, vc::TargetDeviceCUDA>& out)
{
  // get a gaussian kernel in the gpu
  float coeffs[25];
  Eigen::Map<Eigen::Matrix<float,5,5>> gauss_kernel(coeffs);
  SetGaussCoefficients(gauss_kernel);
  cudaMemcpyToSymbol(gauss_coeffs, &coeffs, sizeof(coeffs));
  CudaCheckLastError("cudaMemcpyToSymbol failed");

  // calculate blocks and threads
  dim3 threads, blocks;
  vc::InitDimFromBufferOver(threads, blocks, out);

  // run kernel
  kernel_gaussian_blur_down<<<blocks,threads>>>(in, out);
  CudaCheckLastError("Kernel launch failed (kernel_gaussian_blur_down)");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Squared error
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
__global__ void kernel_squared_error(const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> buf1,
                                     const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> buf2,
                                     vc::Buffer1DView<Scalar, vc::TargetDeviceCUDA> bscratch)
{

  Scalar sum = 0;
  vc::runReductions(buf1.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / buf1.width();
    const unsigned int x = i - (y * buf1.width());
    const Scalar diff = buf1(x,y) - buf2(x,y);
    sum += diff * diff;
  });

  vc::finalizeReduction(bscratch.ptr(), &sum, &reduction_traits<Scalar>::WarpReduceSum, Scalar(0));
}

template <typename T>
T SquaredError(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf1,
               const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf2,
               vc::Buffer1DView<T, vc::TargetDeviceCUDA>& bscratch)
{
  // These I have tuned manually for now
  // NOTE: threads must be a multiple of 32!
  int threads = 32;
  int blocks = 40;

  // impose the limit on blocks
  // max blocks has been carefully selected for the warp shuffle
  // TODO: move max blocks somewhere up in the class
  const int max_blocks = 1024;
  blocks = min(blocks, max_blocks);

  const int smemSize = (threads / 32 + 1) * sizeof(T);

  kernel_squared_error<<<blocks, threads, smemSize>>>(buf1, buf2, bscratch);
  kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch, bscratch, blocks);
  CudaCheckLastError("[SquaredError] kernel launch failed");

  T result;
  auto tmpbuf = vc::Buffer1DView<T, vc::TargetHost>(&result, 1);
  tmpbuf.copyFrom(bscratch);
  return result;
}

template <typename T>
T SquaredError(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf1,
               const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf2)
{
  vc::Buffer1DManaged<T, vc::TargetDeviceCUDA> bscratch(1024);
  return SquaredError(buf1, buf2, bscratch);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Update depth
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int CS>
__global__ void kernel_update_depth(const Eigen::Matrix<Scalar,CS,1> code,
                                    const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> prx_orig,
                                    const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> prx_jac,
                                    const Scalar avg_dpt,
                                    vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> dpt_out)
{
  vc::runReductions(dpt_out.area(), [&] __device__ (unsigned int i)
  {
    const unsigned int y = i / dpt_out.width();
    const unsigned int x = i - (y * dpt_out.width());

    Eigen::Map<const Eigen::Matrix<Scalar,1,CS>> tmp(&prx_jac(x*CS,y));
    const Eigen::Matrix<Scalar,1,CS> prx_J_cde(tmp); // force copy
    dpt_out(x,y) = df::DepthFromCode(code, prx_J_cde, prx_orig(x,y), avg_dpt);
  });
}

template <typename T, int CS, typename ImageBuf>
void UpdateDepth(const Eigen::Matrix<T,CS,1>& code,
                 const ImageBuf& prx_orig,
                 const ImageBuf& prx_jac,
                 T avg_dpt,
                 ImageBuf& dpt_out)
{
  int threads = 32;
  int blocks = 40;
  kernel_update_depth<<<blocks, threads>>>(code, prx_orig, prx_jac, avg_dpt, dpt_out);
  CudaCheckLastError("[UpdateDepth] kernel launch failed");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Explicit instantiations
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template void GaussianBlurDown(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& in,
                               vc::Buffer2DView<float, vc::TargetDeviceCUDA>& out);
//template void GaussianBlurDown(const vc::Buffer2DView<double, vc::TargetDeviceCUDA>& in,
//                               vc::Buffer2DView<double, vc::TargetDeviceCUDA>& out);

template void SobelGradients(const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& img,
                             vc::Buffer2DView<Eigen::Matrix<float,1,2>,vc::TargetDeviceCUDA>& grad);
//template void SobelGradients(const vc::Buffer2DView<double,vc::TargetDeviceCUDA>& img,
//                             vc::Buffer2DView<Eigen::Matrix<double,1,2>,vc::TargetDeviceCUDA>& grad);

template float SquaredError(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf1,
                            const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf2);
//template double SquaredError(const vc::Buffer2DView<double, vc::TargetDeviceCUDA>& buf1,
//                             const vc::Buffer2DView<double, vc::TargetDeviceCUDA>& buf2);

template void UpdateDepth(const Eigen::Matrix<float,32,1>& code,
                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_orig,
                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_jac,
                          float avg_dpt,
                          vc::Buffer2DView<float,vc::TargetDeviceCUDA>& dpt_out);
//template void UpdateDepth(const Eigen::Matrix<float,64,1>& code,
//                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_orig,
//                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_jac,
//                          float avg_dpt,
//                          vc::Buffer2DView<float,vc::TargetDeviceCUDA>& dpt_out);
//template void UpdateDepth(const Eigen::Matrix<float,128,1>& code,
//                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_orig,
//                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_jac,
//                          float avg_dpt,
//                          vc::Buffer2DView<float,vc::TargetDeviceCUDA>& dpt_out);


} // namespace df
