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
#ifndef DF_CU_SFMALIGNER_H_
#define DF_CU_SFMALIGNER_H_

#include <cstddef>
#include <memory>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/CUDAGenerics.hpp>

#include "reduction_items.h"
#include "device_info.h"
#include "dense_sfm.h"

namespace df
{

// forward declarations
template <typename Scalar>
class PinholeCamera;

struct SfmAlignerParams
{
  DenseSfmParams sfmparams;
  int step_threads = 32;  // NOTE: threads must be a multiple of 32!
  int step_blocks = 11;
  int eval_threads = 224;
  int eval_blocks = 66;
};

template <typename Scalar, int CS>
class SfmAligner
{
public:
  typedef std::shared_ptr<SfmAligner<Scalar, CS>> Ptr;
  typedef Eigen::Matrix<Scalar,CS,1> CodeT;
  typedef Eigen::Matrix<Scalar,1,2> ImageGrad;
  typedef Sophus::SE3<Scalar> SE3T;
  typedef vc::Image2DView<Scalar,vc::TargetDeviceCUDA> ImageBuffer;
  typedef vc::Image2DView<ImageGrad,vc::TargetDeviceCUDA> GradBuffer;
  typedef JTJJrReductionItem<Scalar,2*SE3T::DoF+CS> ReductionItem;
  typedef CorrespondenceReductionItem<Scalar> ErrorReductionItem;
  typedef Eigen::Matrix<Scalar,SE3T::DoF,SE3T::DoF> RelposeJac;

  SfmAligner(SfmAlignerParams params = SfmAlignerParams());
  virtual ~SfmAligner();

  ErrorReductionItem EvaluateError(const Sophus::SE3<Scalar>& pose0,
                                   const Sophus::SE3<Scalar>& pose1,
                                   const df::PinholeCamera<Scalar>& cam,
                                   const ImageBuffer& img0,
                                   const ImageBuffer& img1,
                                   const ImageBuffer& dpt0,
                                   const ImageBuffer& std0,
                                   const GradBuffer& grad1);

  ReductionItem RunStep(const SE3T& pose0,
                        const SE3T& pose1,
                        const CodeT& code0,
                        const df::PinholeCamera<Scalar>& cam,
                        const ImageBuffer& img0,
                        const ImageBuffer& img1,
                        const ImageBuffer& dpt0,
                        const ImageBuffer& std0,
                        ImageBuffer& valid0,
                        const ImageBuffer& prx0_jac,
                        const GradBuffer& grad1);

  void SetEvalThreadsBlocks(int threads, int blocks);
  void SetStepThreadsBlocks(int threads, int blocks);

private:
  static const int max_blocks = 1024;
  SfmAlignerParams params_;
  cuda::DeviceInfo devinfo_;
  vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> bscratch_;
  vc::Buffer1DManaged<ErrorReductionItem, vc::TargetDeviceCUDA> bscratch2_;
};

} // namespace df

#endif // DF_CU_SFMALIGNER_H_
