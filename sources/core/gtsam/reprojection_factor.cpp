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
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <gtsam/linear/JacobianFactor.h>

#include "reprojection_factor.h"
#include "cu_image_proc.h"
#include "gtsam_traits.h"
#include "warping.h"
#include "dense_sfm.h"
#include "m_estimators.h"
#include "matching.h"

namespace df
{

template <typename Scalar, int CS>
ReprojectionFactor<Scalar,CS>::ReprojectionFactor(const df::PinholeCamera<Scalar>& cam,
  const KeyframePtr& kf,
  const FramePtr& fr,
  const gtsam::Key& pose0_key,
  const gtsam::Key& pose1_key,
  const gtsam::Key& code0_key,
  const Scalar feature_max_dist,
  const Scalar huber_delta,
  const Scalar sigma,
  const int maxiters,
  const Scalar threshold)
    : Base(gtsam::cref_list_of<3>(pose0_key)(pose1_key)(code0_key)),
      pose0_key_(pose0_key),
      pose1_key_(pose1_key),
      code0_key_(code0_key),
      cam_(cam),
      huber_delta_(huber_delta),
      kf_(kf), fr_(fr),
      sigma_(sigma)
{
  CHECK(kf->has_keypoints && fr->has_keypoints) << "Keypoints have not been calculated";

  // calculate matches
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(kf->features.descriptors, fr->features.descriptors, matches_);

  // prune bad matches
  cv::Mat cameraMatrix;
  cv::eigen2cv(cam_.template Matrix<double>(), cameraMatrix);
  matches_ = PruneMatchesEightPoint(kf->features.keypoints, fr->features.keypoints,
                                    matches_, cameraMatrix, threshold, maxiters);
  matches_ = PruneMatchesByThreshold(matches_, feature_max_dist);

//  cv::imshow("matches", DrawMatches());
//  cv::waitKey(0);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
ReprojectionFactor<Scalar,CS>::ReprojectionFactor(const df::PinholeCamera<Scalar>& cam,
  const std::vector<cv::DMatch>& matches,
  const KeyframePtr& kf,
  const FramePtr& fr,
  const gtsam::Key& pose0_key,
  const gtsam::Key& pose1_key,
  const gtsam::Key& code0_key,
  const Scalar huber_delta,
  const Scalar sigma)
    : Base(gtsam::cref_list_of<3>(pose0_key)(pose1_key)(code0_key)),
      pose0_key_(pose0_key),
      pose1_key_(pose1_key),
      code0_key_(code0_key),
      cam_(cam),
      matches_(matches),
      huber_delta_(huber_delta),
      kf_(kf), fr_(fr), sigma_(sigma) {}

/* ************************************************************************* */
template <typename Scalar, int CS>
ReprojectionFactor<Scalar,CS>::~ReprojectionFactor() {}

/* ************************************************************************* */
template <typename Scalar, int CS>
double ReprojectionFactor<Scalar,CS>::error(const gtsam::Values& c) const
{
  if (this->active(c))
  {
    PoseT p0 = c.at<PoseT>(pose0_key_);
    PoseT p1 = c.at<PoseT>(pose1_key_);
    Eigen::Matrix<Scalar,CS,1> c0 = c.at<CodeT>(code0_key_).template cast<Scalar>();

    // TODO: parametrize!
    Scalar avg_dpt = 2.0;

    // do work here
    Scalar total_sqerr = 0;
    corrs_.clear();
    for (uint i = 0; i < matches_.size(); ++i)
    {
      cv::Point2f query = kf_->features.keypoints[matches_[i].queryIdx].pt;
      cv::Point2f train = fr_->features.keypoints[matches_[i].trainIdx].pt;
      Eigen::Matrix<Scalar,2,1> pix0(query.x, query.y);
      Eigen::Matrix<Scalar,2,1> pix1(train.x, train.y);

      // get relative pose
      PoseT pose10 = df::RelativePose(p1, p0);

      // get pixel code data from keyframe
      auto prx0_jac_img = kf_->pyr_jac.GetCpuLevel(0);
      Eigen::Map<Eigen::Matrix<Scalar,1,CS>> tmp(&prx0_jac_img((int)query.x*CS, (int)query.y));
      Eigen::Matrix<Scalar,1,CS> prx_J_cde(tmp);
      Scalar prx_0code = kf_->pyr_prx_orig.GetCpuLevel(0)(query.x, query.y);

      // calculate depth and correspondence
      Scalar dpt0 = df::DepthFromCode(c0, prx_J_cde, prx_0code, avg_dpt);
      df::Correspondence<Scalar> corr = FindCorrespondence(query.x, query.y, dpt0, cam_,
                                                              pose10, 1.f, 0.f, false);
      corrs_.emplace_back(pix1, corr.pix1);

      Eigen::Matrix<Scalar,2,1> diff = pix1 - corr.pix1;

      // calculate residual
      Scalar err = diff.norm();

      // huber weighting
//      Scalar hbr_wgt = df::DenseSfm_RobustLoss(err, huber_delta_
      Scalar hbr_wgt = df::CauchyWeight(err, huber_delta_);
      err *= hbr_wgt;

      total_sqerr += err * err;

      // huber weighting etc..
    }
    total_err_ = total_sqerr;
    return 0.5 * total_sqerr / sigma_ / sigma_;
  }
  else
  {
    return 0.0;
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
boost::shared_ptr<gtsam::GaussianFactor>
ReprojectionFactor<Scalar,CS>::linearize(const gtsam::Values& c) const
{
  // Only linearize if the factor is active
  if (!this->active(c))
    return boost::shared_ptr<gtsam::JacobianFactor>();

  // recover our values
  PoseT p0 = c.at<PoseT>(pose0_key_);
  PoseT p1 = c.at<PoseT>(pose1_key_);
  Eigen::Matrix<Scalar,CS,1> c0 = c.at<CodeT>(code0_key_).template cast<Scalar>();

  // TODO: parametrize!
  Scalar avg_dpt = 2.0;

  std::vector<int> dimensions = {PoseT::DoF, PoseT::DoF, CS, 1};
  gtsam::VerticalBlockMatrix Ab(dimensions, 2*matches_.size());

  // do work here
  Scalar total_err = 0;
  corrs_.clear();
//  LOG(INFO) << Name();
  for (uint i = 0; i < matches_.size(); ++i)
  {
    cv::Point2f query = kf_->features.keypoints[matches_[i].queryIdx].pt;
    cv::Point2f train = fr_->features.keypoints[matches_[i].trainIdx].pt;
    Eigen::Matrix<Scalar,2,1> pix0(query.x, query.y);
    Eigen::Matrix<Scalar,2,1> pix1(train.x, train.y);

    // get relative pose
    Eigen::Matrix<Scalar,6,6> pose10_J_pose0;
    Eigen::Matrix<Scalar,6,6> pose10_J_pose1;
    PoseT pose10 = df::RelativePose(p1, p0, pose10_J_pose1, pose10_J_pose0);

    // get pixel code data from keyframe
    auto prx0_jac_img = kf_->pyr_jac.GetCpuLevel(0);
    Eigen::Map<Eigen::Matrix<Scalar,1,CS>> prx_J_cde(&prx0_jac_img((int)query.x*CS, (int)query.y));
    Scalar prx_0code = kf_->pyr_prx_orig.GetCpuLevel(0)(query.x, query.y);

    // calculate depth and correspondence
    Scalar dpt0 = df::DepthFromCode(c0, prx_J_cde, prx_0code, avg_dpt);
    df::Correspondence<Scalar> corr = FindCorrespondence(query.x, query.y, dpt0, cam_,
                                                            pose10, 1.f, 0.f, false);

    Scalar dpt2 = corr.tpt[2];
//    LOG(INFO) << "dpt = " << dpt2;
    if (not corr.valid)
    {
      LOG(ERROR) << "Correspondence is not valid! Factor: " << Name();
      Ab(0).template block<2,PoseT::DoF>(i,0).setZero();
      Ab(1).template block<2,PoseT::DoF>(i,0).setZero();
      Ab(2).template block<2,CS>(i,0).setZero();
      Ab(3).template block<2,1>(2*i,0).setZero();
      continue;
    }

    corrs_.emplace_back(pix1, corr.pix1);

    // jacobian correspondence w.r.t code
    Eigen::Matrix<Scalar,2,CS> corr_J_cde;
    df::FindCorrespondenceJacobianCode(corr, dpt0, cam_, pose10, prx_J_cde, avg_dpt, corr_J_cde);

    // jacobian correspondence w.r.t relpose
    Eigen::Matrix<Scalar,2,6> corr_J_pose10;
    corr_J_pose10 = df::FindCorrespondenceJacobianPose(corr, dpt0, cam_, pose10);

    //    err_J_pose0 = corr_J_pose10 * pose10_J_pose0;
    //    err_J_pose1 = corr_J_pose10 * pose10_J_pose1;
    //    err_J_cde0 = corr_J_cde;
    Eigen::Matrix<Scalar,2,PoseT::DoF> err_J_pose0 = corr_J_pose10 * pose10_J_pose0;
    Eigen::Matrix<Scalar,2,PoseT::DoF> err_J_pose1 = corr_J_pose10 * pose10_J_pose1;
    Eigen::Matrix<Scalar,2,CS> err_J_cde = corr_J_cde;

    Eigen::Matrix<Scalar,2,1> diff = pix1 - corr.pix1;

    // calculate residual
    Scalar err = diff.norm();

    // huber weighting
//    Scalar hbr_wgt = df::DenseSfm_RobustLoss(err, huber_delta_);
    Scalar hbr_wgt = df::CauchyWeight(err, huber_delta_);

    // apply huber to jacobians and total error
    err_J_pose0 *= hbr_wgt;
    err_J_pose1 *= hbr_wgt;
    err_J_cde *= hbr_wgt;
    diff *= hbr_wgt;

    total_err += err * err;

    // weighting
    err_J_pose0 /= sigma_;
    err_J_pose1 /= sigma_;
    err_J_cde /= sigma_;
    diff /= sigma_;

    Ab(0).template block<2,PoseT::DoF>(2*i,0) = err_J_pose0.template cast<double>();
    Ab(1).template block<2,PoseT::DoF>(2*i,0) = err_J_pose1.template cast<double>();
    Ab(2).template block<2,CS>(2*i,0) = err_J_cde.template cast<double>();
    Ab(3).template block<2,1>(2*i,0) = diff.template cast<double>();
  }

  total_err_ = total_err;

//  std::ofstream f("sparse_mat.csv");
//  f << Ab.full();
//  f.close();

  // create and return hessianfactor
  const std::vector<gtsam::Key> keys = {pose0_key_, pose1_key_, code0_key_};
  return boost::make_shared<gtsam::JacobianFactor>(keys, Ab);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
std::string ReprojectionFactor<Scalar,CS>::Name() const
{
  std::stringstream ss;
  auto fmt = gtsam::DefaultKeyFormatter;
  ss << "ReprojectionFactor " << fmt(pose0_key_) << " -> " << fmt(pose1_key_)
     << " {" << fmt(code0_key_) << "}";
  return ss.str();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename ReprojectionFactor<Scalar,CS>::shared_ptr
ReprojectionFactor<Scalar,CS>::clone() const
{
  auto ptr = boost::make_shared<This>(cam_, matches_, kf_, fr_, pose0_key_,
                                  pose1_key_, code0_key_, huber_delta_, sigma_);
  ptr->corrs_ = corrs_;
  ptr->total_err_ = total_err_;
  return ptr;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
cv::Mat ReprojectionFactor<Scalar,CS>::DrawMatches() const
{
  // draw matches for debug
  cv::Mat img_matches;
  cv::drawMatches(kf_->color_img, kf_->features.keypoints, fr_->color_img,
                  fr_->features.keypoints, matches_, img_matches);
  return img_matches;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
cv::Mat ReprojectionFactor<Scalar,CS>::ErrorImage() const
{
  // draw matches for debug
  cv::Mat img = fr_->color_img.clone();
  for (auto& corr : corrs_)
  {
    cv::Point p1(corr.first(0), corr.first(1));
    cv::Point p2(corr.second(0), corr.second(1));

    // draw lines
    cv::arrowedLine(img, p2, p1, cv::Scalar(0, 0, 255));
  }

  return img;
}

/* ************************************************************************* */
// explicit instantiation
template class ReprojectionFactor<float, DF_CODE_SIZE>;

} // namespace df
