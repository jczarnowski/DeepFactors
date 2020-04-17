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
#ifndef DF_DEPTH_PRIOR_FACTOR_H_
#define DF_DEPTH_PRIOR_FACTOR_H_

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <VisionCore/Buffers/BufferPyramid.hpp>
#include <opencv2/opencv.hpp>

#include "keyframe.h"
#include "cu_depthaligner.h"

namespace df
{

template <typename Scalar, int CS>
class DepthPriorFactor : public gtsam::NonlinearFactor
{
  typedef gtsam::Vector CodeT;
  typedef gtsam::NonlinearFactor Base;
  typedef df::Keyframe<Scalar> KeyframeT;
  typedef typename KeyframeT::Ptr KeyframePtr;
  typedef vc::Buffer2DManaged<Scalar, vc::TargetDeviceCUDA> DepthBufferDevice;
  typedef vc::Buffer2DManaged<Scalar, vc::TargetHost> DepthBufferHost;
  typedef df::DepthAligner<Scalar,CS> AlignerT;
  typedef typename AlignerT::Ptr AlignerPtr;
  typedef typename AlignerT::ReductionItem StepResult;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DepthPriorFactor(const cv::Mat& dpt,
                   const KeyframePtr& kf,
                   const gtsam::Key& code_key,
                   Scalar sigma,
                   std::size_t pyrlevels,
                   Scalar avgdpt,
                   AlignerPtr aligner);
  virtual ~DepthPriorFactor();

  /**
   * Calculate the error of the factor
   * This is equal to the log of gaussian likelihood, e.g. \f$ 0.5(h(x)-z)^2/sigma^2 \f$
   */
  double error(const gtsam::Values& c) const override;

  /*
   * Linearize the factor around values c. Returns a HessianFactor
   */
  boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

  /**
   * Get the dimension of the factor (number of rows on linearization)
   */
  size_t dim() const override { return CS; };

private:
  /**
   * Run the alignment on all pyramid levels
   */
  StepResult RunAlignment(const CodeT& code) const;

  /**
   * Update the keyframe depth to match a given code
   */
  void UpdateKfDepth(const CodeT& code) const;

private:
  gtsam::Key code_key_;
  Scalar avgdpt_;
  Scalar sigma_;
  std::size_t pyrlevels_;
  KeyframePtr kf_;
  AlignerPtr aligner_;
  vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> pyr_tgtdpt_;
};

} // namespace df

#endif // DF_DEPTH_PRIOR_FACTOR_H_
