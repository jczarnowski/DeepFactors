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
#ifndef DF_PHOTOMETRIC_FACTOR_H_
#define DF_PHOTOMETRIC_FACTOR_H_

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Dense>

#include "cu_sfmaligner.h"
#include "pinhole_camera.h"
#include "keyframe.h"

namespace df
{

template <typename Scalar, int CS>
class PhotometricFactor : public gtsam::NonlinearFactor
{
  typedef PhotometricFactor<Scalar,CS> This;
  typedef gtsam::NonlinearFactor Base;
  typedef Sophus::SE3<Scalar> PoseT;
  typedef gtsam::Vector CodeT;
  typedef df::Keyframe<Scalar> KeyframeT;
  typedef df::Frame<Scalar> FrameT;
  typedef typename KeyframeT::Ptr KeyframePtr;
  typedef typename FrameT::Ptr FramePtr;
  typedef df::SfmAligner<Scalar,CS> AlignerT;
  typedef typename AlignerT::Ptr AlignerPtr;
  typedef typename AlignerT::ReductionItem JacobianResult;
  typedef typename AlignerT::ErrorReductionItem ErrorResult;

  static constexpr int NP = 2 * PoseT::DoF + CS;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PhotometricFactor(const df::PinholeCamera<Scalar> cam,
                    const KeyframePtr& kf,
                    const FramePtr& fr,
                    const gtsam::Key& pose0_key,
                    const gtsam::Key& pose1_key,
                    const gtsam::Key& code0_key,
                    int pyrlevel,
                    AlignerPtr aligner,
                    bool update_valid = true);

  virtual ~PhotometricFactor();

  /**
   * Calculate the error of the factor
   * This is equal to the log of gaussian likelihood, e.g. \f$ 0.5(h(x)-z)^2/sigma^2 \f$
   */
  double error(const gtsam::Values& c) const override;

  /*
   * Linearize the factor around values c. Returns a HessianFactor
   * e = x^Gx - x^g + f
   */
  boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

  /**
   * Get the dimension of the factor (number of rows on linearization)
   */
  size_t dim() const override { return NP; }

  virtual shared_ptr clone() const {
    return shared_ptr(new This(cam_, kf_, fr_, pose0_key_, pose1_key_, code0_key_, pyrlevel_, aligner_));
  }

  /**
   * Return a string describing this factor
   */
  std::string Name() const;

  /**
   * Pyramid level this factor calculates error/jacobians on
   */
  int PyramidLevel() const { return pyrlevel_; }

private:
  ErrorResult RunWarping(const PoseT& pose0, const PoseT& pose1, const CodeT& code0) const;
  JacobianResult RunAlignmentStep(const PoseT& pose0, const PoseT& pose1,
                                  const CodeT& code0) const;
  JacobianResult GetJacobiansIfNeeded(const PoseT& pose0, const PoseT& pose1,
                                      const CodeT& code0) const;

  /**
   * @brief UpdateDepthMaps
   * Calculate keyframe depth maps that correspond to their assigned code
   */
  void UpdateDepthMaps(const CodeT& code0) const;

private:
  /* variables we tie with this factor */
  gtsam::Key pose0_key_;
  gtsam::Key pose1_key_;
  gtsam::Key code0_key_;

  /* data required to do the warping */
  int pyrlevel_;
  AlignerPtr aligner_;
  df::PinholeCamera<Scalar> cam_;

  /* we do modify the keyframes in the factor */
  mutable KeyframePtr kf_;
  FramePtr fr_;

  /* system at the linearization point */
  mutable JacobianResult lin_system_;
  mutable CodeT lin_code0_;
  mutable PoseT lin_pose0_;
  mutable PoseT lin_pose1_;

  mutable bool first_;
  bool update_valid_;
};

} // namespace df

#endif // DF_PHOTOMETRIC_FACTOR_H_
