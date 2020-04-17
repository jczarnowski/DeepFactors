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
#ifndef DF_SPARCE_GEOMETRIC_FACTOR_H_
#define DF_SPARCE_GEOMETRIC_FACTOR_H_

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>

#include "pinhole_camera.h"
#include "keyframe.h"
#include "warping.h"

namespace df
{

template <typename Scalar, int CS>
class ReprojectionFactor : public gtsam::NonlinearFactor
{
  typedef ReprojectionFactor<Scalar,CS> This;
  typedef gtsam::NonlinearFactor Base;
  typedef Sophus::SE3<Scalar> PoseT;
  typedef gtsam::Vector CodeT;
  typedef df::Keyframe<Scalar> KeyframeT;
  typedef df::Frame<Scalar> FrameT;
  typedef typename KeyframeT::Ptr KeyframePtr;
  typedef typename FrameT::Ptr FramePtr;
  typedef df::Correspondence<Scalar> Correspondence;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*!
   * \brief Constructor calculating the keypoint matches between kf0 and kf1
   */
  ReprojectionFactor(const df::PinholeCamera<Scalar>& cam,
                     const KeyframePtr& kf,
                     const FramePtr& fr,
                     const gtsam::Key& pose0_key,
                     const gtsam::Key& pose1_key,
                     const gtsam::Key& code0_key,
                     const Scalar feature_max_dist,
                     const Scalar huber_delta,
                     const Scalar sigma,
                     const int maxiters,
                     const Scalar threshold);

  /*!
   * \brief Constructor that takes calculated matches (for clone)
   */
  ReprojectionFactor(const df::PinholeCamera<Scalar>& cam,
                     const std::vector<cv::DMatch>& matches,
                     const KeyframePtr& kf,
                     const FramePtr& fr,
                     const gtsam::Key& pose0_key,
                     const gtsam::Key& pose1_key,
                     const gtsam::Key& code0_key,
                     const Scalar huber_delta,
                     const Scalar sigma);

  virtual ~ReprojectionFactor();

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
  size_t dim() const override { return 2*matches_.size(); }

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
   */
  cv::Mat DrawMatches() const;

  /*!
   * \brief Produce a reprojection error image
   * \return
   */
  cv::Mat ErrorImage() const;

  Scalar ReprojectionError() const { return total_err_; }

  FramePtr Frame() { return fr_; }
  KeyframePtr Keyframe() { return kf_; }
  std::vector<cv::DMatch> Matches() { return matches_; }

private:
  /* variables we tie with this factor */
  gtsam::Key pose0_key_;
  gtsam::Key pose1_key_;
  gtsam::Key code0_key_;

  /* data required to do the warping */
  df::PinholeCamera<Scalar> cam_;
  std::vector<cv::DMatch> matches_;
  Scalar huber_delta_;

  /* we do modify the keyframes in the factor */
  mutable KeyframePtr kf_;
  FramePtr fr_;

  /* for display */
  mutable std::vector<std::pair<Eigen::Matrix<Scalar,2,1>, Eigen::Matrix<Scalar,2,1>>> corrs_;
  mutable Scalar total_err_;

  Scalar sigma_;
};

} // namespace df

#endif // DF_SPARCE_GEOMETRIC_FACTOR_H_
