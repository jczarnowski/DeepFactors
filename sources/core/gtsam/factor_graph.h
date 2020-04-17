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
#ifndef DF_FACTOR_GRAPH_H_
#define DF_FACTOR_GRAPH_H_

#include <cstddef>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/Symbol.h>
#include <sophus/se3.hpp>

#include "sparse_geometric_factor.h"
#include "photometric_factor.h"
#include "depth_prior_factor.h"
#include "reprojection_factor.h"
#include "cu_sfmaligner.h"
#include "cu_se3aligner.h"
#include "cu_depthaligner.h"
#include "camera_pyramid.h"
#include "gtsam_traits.h"
#include "gtsam_utils.h"

namespace df
{

/*
 * Class that wraps around adding our specific factors
 * to a gtsam::NonlinearFactorGraph
 * TODO: change this to inheritance from gtsam::NonlinearFactorGraph?
 */
template <typename Scalar, int CS>
class FactorGraph
{
  using PhotometricFactor = df::PhotometricFactor<Scalar, CS>;
  using PhotometricFactorPtr = boost::shared_ptr<PhotometricFactor>;
  using ReprojectionFactor = df::ReprojectionFactor<Scalar, CS>;
  using ReprojectionFactorPtr = boost::shared_ptr<ReprojectionFactor>;
  using GeometricFactor = df::SparseGeometricFactor<Scalar, CS>;
  using GeometricFactorPtr = boost::shared_ptr<GeometricFactor>;
  using DepthPriorFactor = df::DepthPriorFactor<Scalar, CS>;
  using FactorPtr = gtsam::NonlinearFactorGraph::sharedFactor;
  using DiagonalNoise = gtsam::noiseModel::Diagonal;
  using Keyframe = df::Keyframe<Scalar>;
  using KeyframePtr = typename Keyframe::Ptr;
  using Frame = df::Frame<Scalar>;
  using FramePtr = typename Frame::Ptr;
  using SE3T = Sophus::SE3<Scalar>;
  using CameraPyr = df::CameraPyramid<Scalar>;
  using SfmAlignerPtr = typename df::SfmAligner<Scalar,CS>::Ptr;
  using DepthAlignerPtr = typename df::DepthAligner<Scalar,CS>::Ptr;

public:
  FactorGraph(const CameraPyr& cam_pyr,
              SfmAlignerPtr sfmaligner = nullptr,
              DepthAlignerPtr dptaligner = nullptr)
      : cam_pyr_(cam_pyr)
  {
    sfm_aligner_ = sfmaligner;
    dptaligner_ = dptaligner;
  }

  void AddPosePrior(const KeyframePtr& kf, const SE3T& mean, const gtsam::Vector& sigmas)
  {
    AddDiagonalNoisePrior(PoseKey(kf->id), mean, sigmas);
  }

  void AddZeroPosePrior(const KeyframePtr& kf, const Scalar& sigma)
  {
    const SE3T pose_identity;
    const gtsam::Vector sigmas = gtsam::Vector::Constant(SE3T::DoF, sigma);
    AddDiagonalNoisePrior(PoseKey(kf->id), pose_identity, sigmas);
  }

  template <typename Derived>
  void AddCodePrior(const KeyframePtr& kf, const Eigen::MatrixBase<Derived>& mean, const gtsam::Vector& sigmas)
  {
    CHECK_EQ(mean.cols(), 1);
    AddDiagonalNoisePrior(CodeKey(kf->id), mean, sigmas);
  }

  void AddZeroCodePrior(const KeyframePtr& kf, const Scalar& sigma)
  {
    const gtsam::Vector zero_code = gtsam::Vector::Zero(CS);
    const gtsam::Vector sigmas = gtsam::Vector::Constant(CS, sigma);
    AddDiagonalNoisePrior(CodeKey(kf->id), zero_code, sigmas);
  }

  PhotometricFactorPtr AddPhotometricFactor(const KeyframePtr& kf, const FramePtr& fr, int pyrlevel)
  {
    if (!sfm_aligner_)
      sfm_aligner_ = std::make_shared<SfmAligner<Scalar,CS>>();

    auto id0 = kf->id;
    auto id1 = fr->id;

    // add the photometric factor
    auto factor = AddFactor<PhotometricFactor>(cam_pyr_[pyrlevel], kf, fr,
                                               PoseKey(id0), AuxPoseKey(id1),
                                               CodeKey(id0), pyrlevel,
                                               sfm_aligner_);
    photo_factors_.push_back(factor);
    return factor;
  }

  // how to get the parameters needed to build the factor?
  PhotometricFactorPtr AddPhotometricFactor(const KeyframePtr& kf0, const KeyframePtr& kf1, int pyrlevel)
  {
    if (!sfm_aligner_)
      sfm_aligner_ = std::make_shared<SfmAligner<Scalar,CS>>();

    auto id0 = kf0->id;
    auto id1 = kf1->id;

    LOG(INFO) << "adding factor between keyframes: " << id0 << " and " << id1;

    // add the photometric factor
    auto factor = AddFactor<PhotometricFactor>(cam_pyr_[pyrlevel], kf0, kf1,
                                               PoseKey(id0), PoseKey(id1),
                                               CodeKey(id0), pyrlevel,
                                               sfm_aligner_);
    photo_factors_.push_back(factor);
    return factor;
  }

  ReprojectionFactorPtr AddReprojectionFactor(const KeyframePtr& kf, const FramePtr& fr,
                                              Scalar max_dist, Scalar huber_delta)
  {
    auto id0 = kf->id;
    auto id1 = fr->id;

    // add the photometric factor
    auto factor = AddFactor<ReprojectionFactor>(cam_pyr_[0], kf, fr,
                                                PoseKey(id0), AuxPoseKey(id1),
                                                CodeKey(id0), max_dist, huber_delta);
    return factor;
  }

  ReprojectionFactorPtr AddReprojectionFactor(const KeyframePtr& kf0, const KeyframePtr& kf1,
                                              Scalar max_dist, Scalar huber_delta)
  {
    auto id0 = kf0->id;
    auto id1 = kf1->id;

    // add the photometric factor
    auto factor = AddFactor<ReprojectionFactor>(cam_pyr_[0], kf0, kf1,
                                                PoseKey(id0), PoseKey(id1),
                                                CodeKey(id0), max_dist, huber_delta);
    return factor;
  }

  GeometricFactorPtr AddGeometricFactor(const KeyframePtr& kf0, const KeyframePtr& kf1,
                                        int num_points, Scalar huber_delta, bool stochastic)
  {
    auto id0 = kf0->id;
    auto id1 = kf1->id;

    // add the photometric factor
    auto factor = AddFactor<GeometricFactor>(cam_pyr_[0], kf0, kf1,
                                             PoseKey(id0), PoseKey(id1),
                                             CodeKey(id0), CodeKey(id1),
                                             num_points, huber_delta,
                                             stochastic);
    return factor;
  }

  // how to get the parameters needed to build the factor?
  void AddDepthPriorFactor(const KeyframePtr& kf, const cv::Mat& dpt, double diag_sigma)
  {
    if (!dptaligner_)
      dptaligner_ = std::make_shared<DepthAligner<Scalar,CS>>();

    // add the photometric factor
    const Scalar avgdpt = 2;  // TODO: parametrize!
    AddFactor<DepthPriorFactor>(dpt, kf, CodeKey(kf->id), diag_sigma,
                                cam_pyr_.Levels(), avgdpt, dptaligner_);
  }

  template <typename T>
  void AddDiagonalNoisePrior(const gtsam::Key& key, const T& prior_mean, const gtsam::Vector& prior_sigmas)
  {
    auto prior_noise = DiagonalNoise::Sigmas(prior_sigmas);
    AddFactor<gtsam::PriorFactor<T>>(key, prior_mean, prior_noise);
  }

  /*
   * Creates a new factor and adds it to the graph
   * NOTE: with inheritance from NonlinearFactorGraph this would have been exposed already
   */
  template<class Factor, class... Args>
  boost::shared_ptr<Factor> AddFactor(Args&&... args)
  {
    auto factor = boost::make_shared<Factor>(std::forward<Args>(args)...);
    graph_.push_back(factor);
    return factor;
  }

  gtsam::NonlinearFactorGraph& Graph() { return graph_; }

  float GetPhotometricError(const gtsam::Values& vals, bool print_all)
  {
    if (print_all)
      LOG(INFO) << "Photometric factors in graph:";
    float total_error = 0;
    for (const auto& factor : photo_factors_)
    {
      float error = factor->error(vals);
      total_error += error;
      if (print_all)
        LOG(INFO) << factor->Name() << " error = " << error;
    }
    return total_error;
  }

  const std::vector<PhotometricFactorPtr> PhotometricFactors() const
  {
    return photo_factors_;
  }

private:
  CameraPyr cam_pyr_;
  SfmAlignerPtr sfm_aligner_; // for warping/jacobians
  DepthAlignerPtr dptaligner_; // for depth prior
  gtsam::NonlinearFactorGraph graph_;
  std::vector<PhotometricFactorPtr> photo_factors_;
};

} // namespace df

#endif // DF_FACTOR_GRAPH_H_
