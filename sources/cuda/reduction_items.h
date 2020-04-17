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
#ifndef DF_REDUCTION_ITEMS_H_
#define DF_REDUCTION_ITEMS_H_

#include <Eigen/Core>
#include <VisionCore/Platform.hpp>
#include <VisionCore/CUDAGenerics.hpp>
#include <VisionCore/Types/SquareUpperTriangularMatrix.hpp>

#include "kernel_utils.h"

namespace df
{

////////////////////////////////////////////////////////////////////////////////////
// CorrespondenceReductionItem
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
struct CorrespondenceReductionItem
{
  EIGEN_DEVICE_FUNC CorrespondenceReductionItem() : residual(0), inliers(0) {}

  #ifdef __CUDACC__
  /**
   * Kepler shuffle summation for CorrespondenceReductionItem.
   */
  EIGEN_PURE_DEVICE_FUNC static inline void WarpReduceSum(CorrespondenceReductionItem<Scalar>& val)
  {
    #pragma unroll
    for(unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
      val.residual += vc::shfl_down(val.residual, offset);
      val.inliers += vc::shfl_down(val.inliers, offset);
    }
  }
  #endif

  EIGEN_DEVICE_FUNC inline CorrespondenceReductionItem<Scalar> operator+(const CorrespondenceReductionItem<Scalar>& rhs) const
  {
    CorrespondenceReductionItem<Scalar> result;
    result += rhs;
    return result;
  }

  EIGEN_DEVICE_FUNC inline CorrespondenceReductionItem<Scalar>& operator+=(const CorrespondenceReductionItem<Scalar>& rhs)
  {
    residual += rhs.residual;
    inliers += rhs.inliers;
    return *this;
  }

  Scalar residual;
  std::size_t inliers;
};

////////////////////////////////////////////////////////////////////////////////////
// JTJJrReductionItem
////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int NP>
struct JTJJrReductionItem
{
  typedef vc::types::SquareUpperTriangularMatrix<Scalar, NP> HessianType;  // Robert's square upper triangular matrix
  typedef Eigen::Matrix<Scalar, NP, 1> JacobianType;

  EIGEN_DEVICE_FUNC JTJJrReductionItem()
  {
    JtJ = HessianType::Zero();
    Jtr = JacobianType::Zero();
    residual = Scalar(0);
    inliers = 0;
  }

  #ifdef __CUDACC__
  /**
   * Kepler shuffle summation for JTJJrReductionItem.
   */
  EIGEN_PURE_DEVICE_FUNC static inline void WarpReduceSum(JTJJrReductionItem<Scalar, NP>& val)
  {
    /**
     * uncommenting all the unrolls will kill the compiler
     * cannot unroll the hessian -- too big
     * does unrolling outer loops make sense performance-wise?
     */
//     #pragma unroll
    for(unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
      val.residual += vc::shfl_down(val.residual, offset);
      val.inliers += vc::shfl_down(val.inliers, offset);

      #pragma unroll
      for(std::size_t i = 0 ; i < JacobianType::RowsAtCompileTime; ++i)
      {
        val.Jtr(i) += vc::shfl_down(val.Jtr(i), offset);
      }

//       #pragma unroll
      for(std::size_t i = 0 ; i < HessianType::CoeffType::RowsAtCompileTime ; ++i)
      {
        val.JtJ.coeff()(i) += vc::shfl_down(val.JtJ.coeff()(i), offset);
      }
    }
  }
  #endif

  EIGEN_DEVICE_FUNC inline JTJJrReductionItem<Scalar, NP> operator+(const JTJJrReductionItem<Scalar, NP>& rhs) const
  {
    JTJJrReductionItem<Scalar, NP> result;
    result += rhs;
    return result;
  }

  EIGEN_DEVICE_FUNC inline JTJJrReductionItem<Scalar, NP>& operator+=(const JTJJrReductionItem<Scalar, NP>& rhs)
  {
    JtJ += rhs.JtJ;
    Jtr += rhs.Jtr;
    residual += rhs.residual;
    inliers += rhs.inliers;
    return *this;
  }

  HessianType JtJ;
  JacobianType Jtr;
  Scalar residual;
  std::size_t inliers;
};

} // namespace df

////////////////////////////////////////////////////////////////////////////////////
// Overloads for vc::SharedMemory<T> for our types
////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

namespace vc
{

#define SPECIALIZE_JTJJrReductionItem(TYPE, CS) \
  template<> \
  struct SharedMemory<df::JTJJrReductionItem<TYPE, CS>> { \
    typedef df::JTJJrReductionItem<TYPE, CS> ReductionItem; \
    \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr() const { \
      extern __shared__ ReductionItem smem_JTJJrReductionItem_##TYPE_##CS[]; \
      return smem_JTJJrReductionItem_##TYPE_##CS; \
    } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr(std::size_t idx) { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem* ptr(std::size_t idx) const { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator()(std::size_t s) { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator()(std::size_t s) const { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator[](std::size_t ix) { return operator()(ix); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator[](std::size_t ix) const { return operator()(ix); } \
  };

#define SPECIALIZE_CorrespondenceReductionItem(TYPE) \
  template<> \
  struct SharedMemory<df::CorrespondenceReductionItem<TYPE>> \
  { \
    typedef df::CorrespondenceReductionItem<TYPE> ReductionItem; \
    \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr() const { \
      extern __shared__ ReductionItem smem_CorrespondenceReductionItem_##TYPE[]; \
      return smem_CorrespondenceReductionItem_##TYPE; \
    } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr(std::size_t idx) { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem* ptr(std::size_t idx) const { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator()(std::size_t s) { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator()(std::size_t s) const { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator[](std::size_t ix) { return operator()(ix); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator[](std::size_t ix) const { return operator()(ix); } \
  };

SPECIALIZE_JTJJrReductionItem(float, 6)    // 6DoF tracking
SPECIALIZE_JTJJrReductionItem(float, 32)   // Depth alignment (against gt)
SPECIALIZE_JTJJrReductionItem(float, 44)   // SfM alignment 2*6DoF + 32 code
SPECIALIZE_CorrespondenceReductionItem(float)

// SPECIALIZE_JTJJrReductionItem(double)
// SPECIALIZE_CorrespondenceReductionItem(double)

#undef SPECIALIZE_JTJJrReductionItem
#undef SPECIALIZE_CorrespondenceReductionItem

} // namespace vc

////////////////////////////////////////////////////////////////////////////////////
// Partial specialization for reduction items so that they work with
// kernel_finalize_reduction
////////////////////////////////////////////////////////////////////////////////////

namespace df
{

/*
  * CorrespondenceReductionItem
  */
template <typename Scalar>
struct reduction_traits<df::CorrespondenceReductionItem<Scalar>>
{
  typedef df::CorrespondenceReductionItem<Scalar> OurT;
  static inline EIGEN_PURE_DEVICE_FUNC void WarpReduceSum(OurT& item)
  {
    OurT::WarpReduceSum(item);
  }

  static inline EIGEN_DEVICE_FUNC OurT zero()
  {
    return OurT();
  }
};

/*
  * JTJJrReductionItem
  */
template <typename Scalar, int NP>
struct reduction_traits<df::JTJJrReductionItem<Scalar,NP>>
{
  typedef df::JTJJrReductionItem<Scalar,NP> OurT;
  static inline EIGEN_PURE_DEVICE_FUNC void WarpReduceSum(OurT& item)
  {
    OurT::WarpReduceSum(item);
  }

  static inline EIGEN_DEVICE_FUNC OurT zero()
  {
    return OurT();
  }
};

} // namespace type

#endif // __CUDACC__

#endif // DF_REDUCTION_ITEMS_H_
