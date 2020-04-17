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
#include <sstream>
#include <limits>

#include <gtsam/linear/HessianFactor.h>
#include <VisionCore/Image/BufferOps.hpp>

#include "photometric_factor.h"
#include "cu_image_proc.h"
#include "gtsam_traits.h"
#include "display_utils.h"
#include "nearest_psd.h"

namespace df
{

template <typename Scalar, int CS>
PhotometricFactor<Scalar,CS>::PhotometricFactor(
  const df::PinholeCamera<Scalar> cam,
  const KeyframePtr& kf,
  const FramePtr& fr,
  const gtsam::Key& pose0_key,
  const gtsam::Key& pose1_key,
  const gtsam::Key& code0_key,
  int pyrlevel,
  AlignerPtr aligner,
  bool update_valid)
    : Base(gtsam::cref_list_of<3>(pose0_key)(pose1_key)(code0_key)),
      pose0_key_(pose0_key),
      pose1_key_(pose1_key),
      code0_key_(code0_key),
      pyrlevel_(pyrlevel),
      aligner_(aligner),
      cam_(cam),
      kf_(kf), fr_(fr),
      first_(true),
      update_valid_(update_valid) {}

/* ************************************************************************* */
template <typename Scalar, int CS>
PhotometricFactor<Scalar,CS>::~PhotometricFactor() {}

/* ************************************************************************* */
template <typename Scalar, int CS>
double PhotometricFactor<Scalar,CS>::error(const gtsam::Values& c) const
{
  if (this->active(c))
  {
    // get our values
    PoseT p0 = c.at<PoseT>(pose0_key_);
    PoseT p1 = c.at<PoseT>(pose1_key_);
    CodeT c0 = c.at<CodeT>(code0_key_);

    // update depth maps to correspond with codes
    UpdateDepthMaps(c0);

    // RunWarping to get the error
    auto res = RunWarping(p0, p1, c0);
    return 0.5 * res.residual;
  }
  else
  {
    return 0.0;
  }
}

/* ************************************************************************* */
template <typename Scalar, int CS>
boost::shared_ptr<gtsam::GaussianFactor>
PhotometricFactor<Scalar,CS>::linearize(const gtsam::Values& c) const
{
  // Only linearize if the factor is active
  if (!this->active(c))
    return boost::shared_ptr<gtsam::HessianFactor>();

  // recover our values
  PoseT p0 = c.at<PoseT>(pose0_key_);
  PoseT p1 = c.at<PoseT>(pose1_key_);
  CodeT c0 = c.at<CodeT>(code0_key_);

  /*
   * Linearize to a HessianFactor
   * G = JtJ
   * g = Jtr
   * f = r^T*r which is the squared error
   * e = x^Gx - x^g + f
   */
  auto sys = GetJacobiansIfNeeded(p0, p1, c0);
  Eigen::MatrixXd JtJ = sys.JtJ.toDenseMatrix().template cast<double>();
  Eigen::MatrixXd Jtr = -sys.Jtr.template cast<double>();

  // check if we got a valid system
  if (sys.inliers < 0)
  {
    LOG(INFO) << "No overlap between images in factor " << Name();
    return boost::shared_ptr<gtsam::HessianFactor>();
  }

  // correct the system to be PSD
//    tic("nearest_psd");
//    Eigen::MatrixXd M = JtJ.template cast<double>();
//    Eigen::MatrixXd jtj = NearestPsd(M);
//    toc("nearest_psd");

  // need to partition the mats here into separate ones
  const std::vector<gtsam::Key> keys = {pose0_key_, pose1_key_, code0_key_};
  std::vector<gtsam::Matrix> Gs;
  std::vector<gtsam::Vector> gs;

  /*
   * Hessian composition
   *
   *      p0   p1   c0
   * p0 [ G11  G12  G13 ]
   * p1 [      G22  G23 ]
   * c0 [           G33 ]
   *
   */
  const Eigen::MatrixXd G11 = JtJ.template block<6,6>(0,0);
  const Eigen::MatrixXd G12 = JtJ.template block<6,6>(0,6);
  const Eigen::MatrixXd G13 = JtJ.template block<6,CS>(0,12);
  const Eigen::MatrixXd G22 = JtJ.template block<6,6>(6,6);
  const Eigen::MatrixXd G23 = JtJ.template block<6,CS>(6,12);
  const Eigen::MatrixXd G33 = JtJ.template block<CS,CS>(12,12);
  Gs.push_back(G11);
  Gs.push_back(G12);
  Gs.push_back(G13);
  Gs.push_back(G22);
  Gs.push_back(G23);
  Gs.push_back(G33);

  /*
   * Jtr composition
   *
   * p0 [ g1 ]
   * p1 [ g2 ]
   * c0 [ g3 ]
   *
   */
  const Eigen::MatrixXd g1 = Jtr.template block<6,1>(0,0);
  const Eigen::MatrixXd g2 = Jtr.template block<6,1>(6,0);
  const Eigen::MatrixXd g3 = Jtr.template block<CS,1>(12,0);
  gs.push_back(g1);
  gs.push_back(g2);
  gs.push_back(g3);

  //  std::cout << std::endl;
  //  std::cout << std::endl;
  //  std::cout << "Asking to linearize " << Name() << std::endl;
  //  std::cout << "at values:" << std::endl;
  //  std::cout << "code0: " << c0.transpose() << std::endl;
  //  std::cout << "code1: " << c1.transpose() << std::endl;
  //  std::cout << "pose0: " << p0.log().transpose() << std::endl;
  //  std::cout << "pose1: " << p1.log().transpose() << std::endl;
  //  std::cout << "-----------------------------------" << std::endl;

//  std::stringstream ss;
//  ss << "photo_" << gtsam::DefaultKeyFormatter(pose0_key_) << "_" << gtsam::DefaultKeyFormatter(pose1_key_) << "jtj_level" << pyrlevel_ << ".csv";
//  std::ofstream f(ss.str());
//  f << JtJ;
//  f.close();

  // create and return hessianfactor
  return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)sys.residual);
}

/* ************************************************************************* */
template <typename Scalar, int CS>
std::string PhotometricFactor<Scalar,CS>::Name() const
{
  std::stringstream ss;
  auto fmt = gtsam::DefaultKeyFormatter;
  ss << "PhotometricFactor " << fmt(pose0_key_) << " -> " << fmt(pose1_key_)
     << ", pyrlevel = " << pyrlevel_;
  return ss.str();
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename PhotometricFactor<Scalar,CS>::ErrorResult
PhotometricFactor<Scalar,CS>::RunWarping(const PoseT& pose0,
                                         const PoseT& pose1,
                                         const CodeT& code0) const
{
  int i = pyrlevel_;
  auto result = aligner_->EvaluateError(pose0, pose1, cam_,
                                        kf_->pyr_img.GetGpuLevel(i),
                                        fr_->pyr_img.GetGpuLevel(i),
                                        kf_->pyr_dpt.GetGpuLevel(i),
                                        kf_->pyr_stdev.GetGpuLevel(i),
                                        fr_->pyr_grad.GetGpuLevel(i));

  if (result.inliers > 0)
  {
    result.residual = result.residual / result.inliers * cam_.width() * cam_.height();
  }
  else
  {
    result.residual = std::numeric_limits<float>::infinity();
  }

  return result;
}


/* ************************************************************************* */
template <typename Scalar, int CS>
typename PhotometricFactor<Scalar,CS>::JacobianResult
PhotometricFactor<Scalar,CS>::RunAlignmentStep(const PoseT& pose0,
                                               const PoseT& pose1,
                                               const CodeT& code0) const
{
  UpdateDepthMaps(code0);

  // display keyframes
//  std::vector<KeyframePtr> kfs = {kf0_, kf1_};
//  std::vector<cv::Mat> array;
//  for (auto& kf : kfs)
//  {
//    cv::Mat img = GetOpenCV(kf->pyr_img.GetCpuLevel(pyrlevel_));
//    cv::Mat rec = df::DepthToProx(GetOpenCV(kf->pyr_dpt.GetCpuLevel(pyrlevel_)), 2);
//    cv::Mat vld = GetOpenCV(kf->pyr_vld.GetCpuLevel(pyrlevel_));
//    cv::Mat std = GetOpenCV(kf->pyr_stdev.GetCpuLevel(pyrlevel_));
//    cv::Mat std_exp;
//    cv::exp(std,std_exp);

//    array.push_back(img);
//    array.push_back(apply_colormap(rec));
//    array.push_back(5*sqrt(2)*std_exp);
//    array.push_back(vld);
//  }
//  std::stringstream ss;
//  ss << kf0_->id << "->" << kf1_->id << " lvl " << pyrlevel_;
//  cv::imshow(ss.str(), CreateMosaic(array, 2, 4));
//  cv::waitKey(5);

  int i = pyrlevel_;
  Eigen::Matrix<Scalar,CS,1> cde = code0.template cast<Scalar>();

  // update valid only if its inter-keyframe connection
//  vc::Image2DView<Scalar,vc::TargetDeviceCUDA> vld;
//  vc::Image2DManaged<Scalar,vc::TargetDeviceCUDA> dummy_vld(cam_.width(), cam_.height());

//  auto kf = std::dynamic_pointer_cast<KeyframeT>(fr_);
//  if (update_valid_ && kf)
//    vld = kf->pyr_vld.GetGpuLevel(i);
//  else
//    vld = dummy_vld.view();

  vc::Image2DView<Scalar,vc::TargetDeviceCUDA> vld = kf_->pyr_vld.GetGpuLevel(i);
  auto result = aligner_->RunStep(pose0, pose1, cde, cam_,
                                  kf_->pyr_img.GetGpuLevel(i),
                                  fr_->pyr_img.GetGpuLevel(i),
                                  kf_->pyr_dpt.GetGpuLevel(i),
                                  kf_->pyr_stdev.GetGpuLevel(i),
                                  vld,
                                  kf_->pyr_jac.GetGpuLevel(i),
                                  fr_->pyr_grad.GetGpuLevel(i));
  if (result.inliers > 0)
  {
    result.residual = result.residual / result.inliers * cam_.width() * cam_.height();
  }
  else
  {
    result.residual = std::numeric_limits<float>::infinity();
  }

//  LOG(INFO) << "Getting jacobians for " << Name();
//  LOG(INFO) << "reweighted residual: " << result.residual;
//  LOG(INFO) << "inliers: " << result.inliers << " (" << result.inliers / cam_.width() / cam_.height() * 100 << "%)";
//  LOG(INFO) << "code0: " << code0.transpose();
//  LOG(INFO) << "code1: " << code1.transpose();
//  LOG(INFO) << "pose0: " << pose0.log().transpose();
//  LOG(INFO) << "pose1: " << pose1.log().transpose();

  return result;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
typename PhotometricFactor<Scalar,CS>::JacobianResult
PhotometricFactor<Scalar,CS>::GetJacobiansIfNeeded(const PoseT& pose0,
                                                   const PoseT& pose1,
                                                   const CodeT& code0) const
{
  double eps = 1e-6;
  if (!gtsam::traits<PoseT>::Equals(pose0, lin_pose0_, eps) ||
      !gtsam::traits<PoseT>::Equals(pose1, lin_pose1_, eps) ||
      !gtsam::traits<CodeT>::Equals(code0, lin_code0_, eps) ||
      first_) // linearization point is different
  {
    first_ = false;
//    auto kf = std::dynamic_pointer_cast<KeyframeT>(fr_);
////    if (update_valid_ && kf)
////      vc::image::fillBuffer(kf->pyr_vld.GetGpuLevel(0), 0.0f);
    lin_system_ = RunAlignmentStep(pose0, pose1, code0);
    lin_pose0_ = pose0;
    lin_pose1_ = pose1;
    lin_code0_ = code0;
  }
//  else
//  {
//    std::cout << "Did not have to relinearize the photo factor!" << std::endl;
//    std::cout << "Linearization point is: " << std::endl;
//    std::cout << "code0: " << lin_code0_.transpose() << std::endl;
//    std::cout << "code1: " << lin_code1_.transpose() << std::endl;
//    std::cout << "pose0: " << lin_pose0_.log().transpose() << std::endl;
//    std::cout << "pose1: " << lin_pose1_.log().transpose() << std::endl;
//  }

  return lin_system_;
}

/* ************************************************************************* */
template <typename Scalar, int CS>
void PhotometricFactor<Scalar,CS>::UpdateDepthMaps(const CodeT& code0) const
{
  // TODO: remove hardcoded 2.0f avg_dpt!!
  int i = pyrlevel_;
  Eigen::Matrix<Scalar,CS,1> cde0 = code0.template cast<Scalar>();
  df::UpdateDepth(cde0, kf_->pyr_prx_orig.GetGpuLevel(i),
                     kf_->pyr_jac.GetGpuLevel(i),
                     2.0f,
                     kf_->pyr_dpt.GetGpuLevel(i));
}

/* ************************************************************************* */
// explicit instantiation
template class PhotometricFactor<float, DF_CODE_SIZE>;

} // namespace df
