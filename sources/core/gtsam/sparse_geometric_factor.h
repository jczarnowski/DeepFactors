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
#ifndef DF_SPARSE_GEOMETRIC_FACTOR_H_
#define DF_SPARSE_GEOMETRIC_FACTOR_H_

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>

#include "uniform_sampler.h"
#include "pinhole_camera.h"
#include "keyframe.h"

namespace df
{

template <typename Scalar, int CS>
class SparseGeometricFactor : public gtsam::NonlinearFactor
{
  typedef SparseGeometricFactor<Scalar,CS> This;
  typedef gtsam::NonlinearFactor Base;
  typedef Sophus::SE3<Scalar> PoseT;
  typedef gtsam::Vector CodeT;
  typedef df::Keyframe<Scalar> KeyframeT;
  typedef typename KeyframeT::Ptr KeyframePtr;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*!
   * \brief Constructor calculating the sparse sampling
   */
  SparseGeometricFactor(const df::PinholeCamera<Scalar>& cam,
                        const KeyframePtr& kf0,
                        const KeyframePtr& kf1,
                        const gtsam::Key& pose0_key,
                        const gtsam::Key& pose1_key,
                        const gtsam::Key& code0_key,
                        const gtsam::Key& code1_key,
                        int num_points,
                        Scalar huber_delta,
                        bool stochastic);

  /*!
   * \brief Constructor that takes sampled points (for clone)
   */
  SparseGeometricFactor(const df::PinholeCamera<Scalar>& cam,
                        const std::vector<Point>& points,
                        const KeyframePtr& kf0,
                        const KeyframePtr& kf1,
                        const gtsam::Key& pose0_key,
                        const gtsam::Key& pose1_key,
                        const gtsam::Key& code0_key,
                        const gtsam::Key& code1_key,
                        Scalar huber_delta,
                        bool stochastic);

  virtual ~SparseGeometricFactor();

  /*!
   * \brief Calculate the error of the factor
   * This is equal to the log of gaussian likelihood, e.g. \f$ 0.5(h(x)-z)^2/sigma^2 \f$
   * \param c Values to evaluate the error at
   * \return The error
   */
  double error(const gtsam::Values& c) const override;

  /*!
   * \brief Linearizes this factor at a linearization point
   * \param c Linearization point
   * \return Linearized factor
   */
  boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

  /*!
   * \brief Get the dimension of the factor (number of rows on linearization)
   * \return
   */
  size_t dim() const override { return points_.size(); }

  /*!
   * \brief Return a string describing this factor
   * \return Factor description string
   */
  std::string Name() const;

  /*!
   * \brief Clone the factor
   * \return shared ptr to a cloned factor
   */
  virtual shared_ptr clone() const;

  /*!
   * \brief Return an image of drawn matches for debug
   * \return
   * TODO
   */
//  cv::Mat DrawMatches() const;

private:
  /* variables we tie with this factor */
  gtsam::Key pose0_key_;
  gtsam::Key pose1_key_;
  gtsam::Key code0_key_;
  gtsam::Key code1_key_;

  /* data required to do the warping */
  df::PinholeCamera<Scalar> cam_;
  mutable std::vector<Point> points_;
  Scalar huber_delta_;

  /* we do modify the keyframes in the factor */
  mutable KeyframePtr kf0_;
  mutable KeyframePtr kf1_;

  bool stochastic_;
};

} // namespace df

#endif // DF_SPARSE_GEOMETRIC_FACTOR_H_
