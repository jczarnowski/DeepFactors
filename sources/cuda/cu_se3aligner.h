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
#ifndef DF_CU_SE3ALIGNER_H_
#define DF_CU_SE3ALIGNER_H_

#include <cstddef>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/CUDAGenerics.hpp>

#include "reduction_items.h"

namespace df
{

// forward declarations
template <typename Scalar>
class PinholeCamera;

template <typename Scalar>
class SE3Aligner
{
public:
  typedef std::shared_ptr<SE3Aligner<Scalar>> Ptr;
  typedef Sophus::SE3<Scalar> UpdateType;
  typedef Eigen::Matrix<Scalar,1,2> ImageGrad;
  typedef JTJJrReductionItem<Scalar,6> ReductionItem;
  typedef CorrespondenceReductionItem<Scalar> CorrespondenceItem;
  typedef vc::Image2DView<Scalar, vc::TargetDeviceCUDA> ImageBuffer;
  typedef vc::Image2DView<ImageGrad, vc::TargetDeviceCUDA> GradBuffer;
  typedef vc::Buffer1DManaged<CorrespondenceItem, vc::TargetDeviceCUDA> CorrespondenceReductionBuffer;
  typedef vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> StepReductionBuffer;

  SE3Aligner();
  virtual ~SE3Aligner();

  /**
   * Renders img0 from image img1 at pose T_01
   */
  CorrespondenceItem Warp(const Sophus::SE3<Scalar>& se3,
                          const df::PinholeCamera<Scalar>& cam,
                          const ImageBuffer& img0,
                          const ImageBuffer& img1,
                          const ImageBuffer& dpt0,
                          ImageBuffer& img2);    // img1 warped into img0

  ReductionItem RunStep(const Sophus::SE3<Scalar>& se3,
                        const df::PinholeCamera<Scalar>& cam,
                        const ImageBuffer& img0,
                        const ImageBuffer& img1,
                        const ImageBuffer& dpt0,
                        const GradBuffer& grad1);

  void SetHuberDelta(float val) { huber_delta_ = val; }

private:
  /*
   * Maximum number of blocks we launch the kernel with.
   * This should be max 1024 as we can finalize the reduction
   * in a single block of 1024 threads (max number of threads per block)
   */
  int max_blocks = 1024;

  // buffers for per-block reduced items
  CorrespondenceReductionBuffer bscratch1_;
  StepReductionBuffer bscratch2_;
  float huber_delta_ = 0.1f;
};

} // namespace df

#endif // DF_CU_SE3ALIGNER_H_
