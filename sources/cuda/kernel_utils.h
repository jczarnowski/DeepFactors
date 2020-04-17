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
#ifndef DF_KERNEL_UTILS_H_
#define DF_KERNEL_UTILS_H_

#include <VisionCore/Platform.hpp>
#include <VisionCore/Buffers/Reductions.hpp>
#include <VisionCore/CUDAGenerics.hpp>
#include <Eigen/Core>

namespace df
{

#ifdef __CUDACC__

/*
 * Traits needed to use kernel_finalize_reduction
 */
template <typename T>
struct reduction_traits
{
  static inline EIGEN_PURE_DEVICE_FUNC void WarpReduceSum(T& item)
  {
    for(unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
      item += vc::shfl_down(item, offset);
    }
  }

  static inline EIGEN_DEVICE_FUNC T zero()
  {
    return T(0);
  }
};

template <typename T>
__global__ void kernel_finalize_reduction(const vc::Buffer1DView<T, vc::TargetDeviceCUDA> in,
                                          vc::Buffer1DView<T, vc::TargetDeviceCUDA> out,
                                          const int nblocks)
{
  T sum;

  // this uses grid-stride loop running your lambda
  // each thread sums up multiples of data
  vc::runReductions(nblocks, [&] __device__ (unsigned int i)
  {
    sum += in[i];
  });

  // reduce blocks and store to memory
  // we assume here this kernel is always run with one block
  // so this will be the final sum
  vc::finalizeReduction(out.ptr(), &sum, &reduction_traits<T>::WarpReduceSum, reduction_traits<T>::zero());
}

#endif

} //namespace df


template <typename Derived>
EIGEN_DEVICE_FUNC void print_mat(const Eigen::MatrixBase<Derived>& mat)
{
  for (int i = 0; i < Derived::RowsAtCompileTime; ++i)
  {
    for (int j = 0; j < Derived::ColsAtCompileTime; ++j)
      printf("%f ", mat(i,j));
    printf("\n");
  }
}

#endif // DF_KERNEL_UTILS_H_
