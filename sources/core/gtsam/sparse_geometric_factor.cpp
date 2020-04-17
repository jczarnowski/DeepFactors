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
#include <gtsam/linear/JacobianFactor.h>

#include "sparse_geometric_factor.h"
#include "cu_image_proc.h"
#include "gtsam_traits.h"
#include "warping.h"
#include "dense_sfm.h"

namespace df
{

template <typename Scalar, int CS>
SparseGeometricFactor<Scalar,CS>::SparseGeometricFactor(const df::PinholeCamera<Scalar>& cam,
  const KeyframePtr& kf0,
  const KeyframePtr& kf1,
  const gtsam::Key& pose0_key,
  const gtsam::Key& pose1_key,
  const gtsam::Key& code0_key,
  const gtsam::Key& code1_key,
  int num_points,
  Scalar huber_delta,
  bool stochastic)
    : Base(gtsam::cref_list_of<4>(pose0_key)(pose1_key)(code0_key)(code1_key)),
      pose0_key_(pose0_key),
      pose1_key_(pose1_key),
      code0_key_(code0_key),
      code1_key_(code1_key),
      cam_(cam),
      huber_delta_(huber_delta),
      kf0_(kf0), kf1_(kf1),
      stochastic_(stochastic)
{
  UniformSampler sampler(cam.width(), cam.height());
  points_ = sampler.SamplePoints(num_points);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
SparseGeometricFactor<Scalar,CS>::SparseGeometricFactor(
  const df::PinholeCamera<Scalar>& cam,
  const std::vector<Point>& points,
  const KeyframePtr& kf0,
  const KeyframePtr& kf1,
  const gtsam::Key& pose0_key,
  const gtsam::Key& pose1_key,
  const gtsam::Key& code0_key,
  const gtsam::Key& code1_key,
  Scalar huber_delta,
  bool stochastic)
    : Base(gtsam::cref_list_of<4>(pose0_key)(pose1_key)(code0_key)(code1_key)),
      pose0_key_(pose0_key),
      pose1_key_(pose1_key),
      code0_key_(code0_key),
      code1_key_(code1_key),
      cam_(cam),
      points_(points),
      huber_delta_(huber_delta),
      kf0_(kf0), kf1_(kf1),
      stochastic_(stochastic) {}

/* ************************************************************************* */
template <typename Scalar, int CS>
SparseGeometricFactor<Scalar,CS>::~SparseGeometricFactor() {}

/* ************************************************************************* */
template <typename Scalar, int CS>
double SparseGeometricFactor<Scalar,CS>::error(const gtsam::Values& c) const
{
  if (this->active(c))
  {
    PoseT p0 = c.at<PoseT>(pose0_key_);
    PoseT p1 = c.at<PoseT>(pose1_key_);
    Eigen::Matrix<Scalar,CS,1> c0 = c.at<CodeT>(code0_key_).template cast<Scalar>();
    Eigen::Matrix<Scalar,CS,1> c1 = c.at<CodeT>(code1_key_).template cast<Scalar>();

    // TODO: parametrize!
    Scalar avg_dpt = 2.0;

    // do work here
    Scalar total_sqerr = 0;
    for (uint i = 0; i < points_.size(); ++i)
    {
      auto pt = points_[i];

      // get relative pose
      PoseT pose10 = df::RelativePose(p1, p0);

      // get pixel code data from keyframe
      auto prx0_jac_img = kf0_->pyr_jac.GetCpuLevel(0);
      Eigen::Map<Eigen::Matrix<Scalar,1,CS>> prx0_J_cde(&prx0_jac_img((int)pt.x*CS, (int)pt.y));
      Scalar prx0_0code = kf0_->pyr_prx_orig.GetCpuLevel(0)(pt.x, pt.y);

      // calculate depth and correspondence
      Scalar dpt0 = df::DepthFromCode(c0, prx0_J_cde, prx0_0code, avg_dpt);
      df::Correspondence<Scalar> corr = FindCorrespondence(pt.x, pt.y, dpt0, cam_, pose10);

      // depth0 in frame 1
      Scalar dpt1_p = corr.tpt(2);

      // use nearest neighbour instead of interpolation
      Eigen::Matrix<int,2,1> pix1_nn = corr.pix1.template cast<int>();

      // calculate depth from frame 1
      auto prx1_jac_img = kf1_->pyr_jac.GetCpuLevel(0);
      Eigen::Map<Eigen::Matrix<Scalar,1,CS>> prx1_J_cde(&prx1_jac_img(pix1_nn(0)*CS, pix1_nn(1)));
      Scalar prx1_0code = kf1_->pyr_prx_orig.GetCpuLevel(0)(pix1_nn(0), pix1_nn(1));
      Scalar dpt1 = df::DepthFromCode(c1, prx1_J_cde, prx1_0code, avg_dpt);

      // calculate residual
      Scalar err = dpt1_p - dpt1;

      // huber weighting
      Scalar hbr_wgt = df::DenseSfm_RobustLoss(err, huber_delta_);
      err *= hbr_wgt;

      total_sqerr += err * err;
    }
    return 0.5 * total_sqerr;
  }
  else
  {
    return 0.0;
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
boost::shared_ptr<gtsam::GaussianFactor>
SparseGeometricFactor<Scalar,CS>::linearize(const gtsam::Values& c) const
{
  // Only linearize if the factor is active
  if (!this->active(c))
    return boost::shared_ptr<gtsam::JacobianFactor>();

  if (stochastic_)
  {
    UniformSampler sampler(cam_.width(), cam_.height());
    points_ = sampler.SamplePoints(points_.size());
  }

  // recover our values
  PoseT p0 = c.at<PoseT>(pose0_key_);
  PoseT p1 = c.at<PoseT>(pose1_key_);
  Eigen::Matrix<Scalar,CS,1> c0 = c.at<CodeT>(code0_key_).template cast<Scalar>();
  Eigen::Matrix<Scalar,CS,1> c1 = c.at<CodeT>(code1_key_).template cast<Scalar>();

  // TODO: parametrize!
  Scalar avg_dpt = 2.0;

  std::vector<int> dimensions = {PoseT::DoF, PoseT::DoF, CS, CS, 1};
  gtsam::VerticalBlockMatrix Ab(dimensions, points_.size());

  for (uint i = 0; i < points_.size(); ++i)
  {
    auto pt = points_[i];

    // get relative pose
    Eigen::Matrix<Scalar,6,6> pose10_J_pose0;
    Eigen::Matrix<Scalar,6,6> pose10_J_pose1;
    PoseT pose10 = df::RelativePose(p1, p0, pose10_J_pose1, pose10_J_pose0);

    // get pixel code data from keyframe 0
    auto prx0_jac_img = kf0_->pyr_jac.GetCpuLevel(0);
    Eigen::Map<Eigen::Matrix<Scalar,1,CS>> prx0_J_cde(&prx0_jac_img((int)pt.x*CS, (int)pt.y));
    Scalar prx0_0code = kf0_->pyr_prx_orig.GetCpuLevel(0)(pt.x, pt.y);

    // calculate depth and correspondence
    Scalar dpt0 = df::DepthFromCode(c0, prx0_J_cde, prx0_0code, avg_dpt);
    df::Correspondence<Scalar> corr = FindCorrespondence(pt.x, pt.y, dpt0, cam_, pose10);

    if (!cam_.PixelValid(corr.pix1) || !corr.valid)
    {
      Ab(0).template block<1,PoseT::DoF>(i,0).setZero();
      Ab(1).template block<1,PoseT::DoF>(i,0).setZero();
      Ab(2).template block<1,CS>(i,0).setZero();
      Ab(3).template block<1,CS>(i,0).setZero();
      Ab(4)(i,0) = 0;
      continue;
    }

    // depth0 in frame 1
    Scalar dpt1_p = corr.tpt(2);

    // use nearest neighbour instead of interpolation
    Eigen::Matrix<int,2,1> pix1_nn = corr.pix1.template cast<int>();

    // calculate depth from frame 1
    auto prx1_jac_img = kf1_->pyr_jac.GetCpuLevel(0);
    Eigen::Map<Eigen::Matrix<Scalar,1,CS>> prx1_J_cde(&prx1_jac_img(pix1_nn(0)*CS, pix1_nn(1)));
    Scalar prx1_0code = kf1_->pyr_prx_orig.GetCpuLevel(0)(pix1_nn(0), pix1_nn(1));
    Scalar dpt1 = df::DepthFromCode(c1, prx1_J_cde, prx1_0code, avg_dpt);

    // calculate residual
    Scalar err = dpt1 - dpt1_p;

    // calculate jacobians
    Eigen::Matrix<Scalar,1,PoseT::DoF> err_J_pose0;
    Eigen::Matrix<Scalar,1,PoseT::DoF> err_J_pose1;
    Eigen::Matrix<Scalar,1,CS> err_J_cde0;
    Eigen::Matrix<Scalar,1,CS> err_J_cde1;

    Eigen::Matrix<Scalar,1,2> dpt_grad = kf1_->dpt_grad(pix1_nn(0), pix1_nn(1));

    // pose jacobians
    Eigen::Matrix<Scalar,2,PoseT::DoF> corr_J_pose10 = \
        df::FindCorrespondenceJacobianPose(corr, dpt0, cam_, pose10);

    // err_J_pose0
    Eigen::Matrix<Scalar,3,PoseT::DoF> tpt_J_pose0 = \
        df::TransformJacobianPose(corr.pt, pose10) * pose10_J_pose0;
    Eigen::Matrix<Scalar,1,PoseT::DoF> dpt1p_J_pose0 = \
        tpt_J_pose0.template block<1,PoseT::DoF>(2,0);
    err_J_pose0 = dpt1p_J_pose0 - dpt_grad * corr_J_pose10 * pose10_J_pose0;

    // err_J_pose1
    Eigen::Matrix<Scalar,3,PoseT::DoF> tpt_J_pose1 = \
        df::TransformJacobianPose(corr.pt, pose10) * pose10_J_pose1;
    Eigen::Matrix<Scalar,1,PoseT::DoF> dpt1p_J_pose1 = \
        tpt_J_pose1.template block<1,PoseT::DoF>(2,0);
    err_J_pose1 = dpt1p_J_pose1 - dpt_grad * corr_J_pose10 * pose10_J_pose1;

    // err_J_cde0
    Eigen::Matrix<Scalar,2,CS> corr_J_cde0;
    df::FindCorrespondenceJacobianCode(corr, dpt0, cam_, pose10, prx0_J_cde, avg_dpt, corr_J_cde0);
    Eigen::Matrix<Scalar,3,3> trans_J_pt = df::TransformJacobianPoint(corr.pt, pose10);
    Eigen::Matrix<Scalar,3,CS> trans_J_cde = trans_J_pt * cam_.ReprojectDepthJacobian(corr.pix0, dpt0) \
        * df::DepthJacobianPrx(dpt0, avg_dpt) * prx0_J_cde;
    err_J_cde0 = trans_J_cde.template block<1,CS>(2,0) - dpt_grad * corr_J_cde0;

    // err_J_cde1
    err_J_cde1 = -df::DepthJacobianPrx(dpt1, avg_dpt) * prx1_J_cde;

    // huber weighting
    Scalar hbr_wgt = df::DenseSfm_RobustLoss(err, huber_delta_);

    err *= hbr_wgt;
    err_J_pose0 *= hbr_wgt;
    err_J_pose1 *= hbr_wgt;
    err_J_cde0 *= hbr_wgt;
    err_J_cde1 *= hbr_wgt;

    // fill the jacobian
    Ab(0).template block<1,PoseT::DoF>(i,0) = err_J_pose0.template cast<double>();
    Ab(1).template block<1,PoseT::DoF>(i,0) = err_J_pose1.template cast<double>();
    Ab(2).template block<1,CS>(i,0) = err_J_cde0.template cast<double>();
    Ab(3).template block<1,CS>(i,0) = err_J_cde1.template cast<double>();
    Ab(4)(i,0) = err;
  }

//  std::ofstream f("sparse_mat.csv");
//  f << Ab.full();
//  f.close();

  // create and return hessianfactor
  const std::vector<gtsam::Key> keys = {pose0_key_, pose1_key_, code0_key_, code1_key_};
  return boost::make_shared<gtsam::JacobianFactor>(keys, Ab);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
std::string SparseGeometricFactor<Scalar,CS>::Name() const
{
  std::stringstream ss;
  auto fmt = gtsam::DefaultKeyFormatter;
  ss << "SparseGeometricFactor " << kf0_->id << " -> " << kf1_->id << " keys = {"
     << fmt(pose0_key_) << ", " << fmt(pose1_key_) << ", " << fmt(code0_key_)
     << "}";
  return ss.str();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename SparseGeometricFactor<Scalar,CS>::shared_ptr
SparseGeometricFactor<Scalar,CS>::clone() const
{
  return boost::make_shared<This>(cam_, points_, kf0_, kf1_, pose0_key_,
                                  pose1_key_, code0_key_, code1_key_, huber_delta_, stochastic_);
}

/* ************************************************************************* */
// explicit instantiation
template class SparseGeometricFactor<float, DF_CODE_SIZE>;

} // namespace df
