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
#include <gtsam/slam/PriorFactor.h>

#include "df_work.h"
#include "gtsam_utils.h"
#include "gtsam_traits.h"

namespace df
{
namespace work
{

template<typename Scalar, int CS>
InitVariables<Scalar,CS>::InitVariables(KeyframePtr kf, Scalar code_prior)
  : first_(true)
{
  typedef gtsam::noiseModel::Diagonal DiagonalNoise;
  typedef gtsam::PriorFactor<gtsam::Vector> PriorFactor;

  gtsam::Vector zero_code = gtsam::Vector::Zero(CS);
  var_init_.insert(PoseKey(kf->id), kf->pose_wk);
  var_init_.insert(CodeKey(kf->id), zero_code);

  gtsam::Vector prior_sigmas = gtsam::Vector::Constant(CS,1,code_prior);
  auto prior_noise = DiagonalNoise::Sigmas(prior_sigmas);
  priors_.emplace_shared<PriorFactor>(CodeKey(kf->id), zero_code, prior_noise);

  name_ = kf->Name();
}

template<typename Scalar, int CS>
InitVariables<Scalar,CS>::InitVariables(KeyframePtr kf, Scalar code_prior, Scalar pose_prior)
  : InitVariables(kf, code_prior)
{
  typedef gtsam::noiseModel::Diagonal DiagonalNoise;
  typedef gtsam::PriorFactor<Sophus::SE3f> PriorFactor;

  gtsam::Vector prior_sigmas = gtsam::Vector::Constant(6, 1, pose_prior);
  auto prior_noise = DiagonalNoise::Sigmas(prior_sigmas);
  priors_.emplace_shared<PriorFactor>(PoseKey(kf->id), Sophus::SE3f{}, prior_noise);
}

template<typename Scalar, int CS>
InitVariables<Scalar,CS>::InitVariables(FramePtr fr)
  : first_(true)
{
  var_init_.insert(AuxPoseKey(fr->id), fr->pose_wk);
  name_ = fr->Name();
}

template<typename Scalar, int CS>
InitVariables<Scalar,CS>::~InitVariables() {}

template<typename Scalar, int CS>
void InitVariables<Scalar,CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                           gtsam::FactorIndices &remove_indices,
                                           gtsam::Values &var_init)
{
  if (first_)
  {
    new_factors += priors_;
    var_init.insert(var_init_);
    first_ = false;
  }
}

template<typename Scalar, int CS>
void InitVariables<Scalar,CS>::Update()
{

}

template<typename Scalar, int CS>
std::string InitVariables<Scalar,CS>::Name()
{
  return Id() + " InitVariables " + name_;
}

// explicit instantiation
template class InitVariables<float, DF_CODE_SIZE>;

/* ************************************************************************* */
template <typename Scalar>
OptimizeWork<Scalar>::OptimizeWork(int iters, bool remove_after)
  : remove_(false),
    first_(true),
    iters_({iters}),
    orig_iters_(iters_),
    active_level_(0),
    remove_after_(remove_after) {}

template <typename Scalar>
OptimizeWork<Scalar>::OptimizeWork(OptimizeWork::IterList iters, bool remove_after)
  : remove_(false),
    first_(true),
    iters_(iters),
    orig_iters_(iters),
    active_level_(iters.size()-1),
    remove_after_(remove_after) {}

template <typename Scalar>
void OptimizeWork<Scalar>::Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                                       gtsam::FactorIndices& remove_indices,
                                       gtsam::Values& var_init)
{
  if (remove_)
  {
    remove_indices.insert(remove_indices.begin(), last_indices_.begin(),
                          last_indices_.end());
    active_level_ = -2;
  }

  if (first_ || (active_level_ >= 0 && IsNewLevelStart()))
  {
    first_ = false;
    new_factors += ConstructFactors();
    remove_indices.insert(remove_indices.begin(), last_indices_.begin(),
                          last_indices_.end());
  }
}

template <typename Scalar>
gtsam::NonlinearFactorGraph OptimizeWork<Scalar>::ConstructFactors()
{
  return gtsam::NonlinearFactorGraph();
}

template <typename Scalar>
void OptimizeWork<Scalar>::Update()
{
  if (active_level_ >= 0 && --iters_[active_level_] < 0)
    active_level_ -= 1;

  if (remove_after_ && active_level_ < 0)
    SignalRemove();
}

template <typename Scalar>
bool OptimizeWork<Scalar>::Finished() const
{
  if (remove_after_)
    return active_level_ == -2;
  else
    return active_level_ == -1;
  return false;
}

template <typename Scalar>
void OptimizeWork<Scalar>::SignalNoRelinearize()
{
  if (!first_)
    active_level_ -= 1;
}

template <typename Scalar>
void OptimizeWork<Scalar>::SignalRemove()
{
  remove_ = true;
}

template <typename Scalar>
void OptimizeWork<Scalar>::LastFactorIndices(gtsam::FactorIndices& indices)
{
  last_indices_ = indices;
}

template <typename Scalar>
bool OptimizeWork<Scalar>::IsCoarsestLevel() const
{
  return active_level_ == (int)iters_.size()-1;
}

template <typename Scalar>
bool OptimizeWork<Scalar>::IsNewLevelStart() const
{
  return active_level_ >= 0 && iters_[active_level_] == orig_iters_[active_level_];
}

template class OptimizeWork<float>;

/* ************************************************************************* */
template<typename Scalar, int CS>
OptimizePhoto<Scalar,CS>::OptimizePhoto(KeyframePtr kf, FramePtr fr,
                                        IterList iters, CamPyramid cam_pyr,
                                        AlignerPtr aligner, bool update_vld, bool remove_after)
  : OptimizeWork<Scalar>(iters, remove_after),
    kf_(kf), fr_(fr),
    cam_pyr_(cam_pyr),
    aligner_(aligner),
    update_vld_(update_vld) {}

template<typename Scalar, int CS>
OptimizePhoto<Scalar,CS>::~OptimizePhoto() {}

template<typename Scalar, int CS>
gtsam::NonlinearFactorGraph OptimizePhoto<Scalar,CS>::ConstructFactors()
{
  DCHECK_GE(this->active_level_, 0);
  gtsam::NonlinearFactorGraph graph;

  gtsam::Key pose0_key = PoseKey(kf_->id);
  gtsam::Key code0_key = CodeKey(kf_->id);
  gtsam::Key pose1_key = fr_->IsKeyframe() ? PoseKey(fr_->id) : AuxPoseKey(fr_->id);

  graph.emplace_shared<PhotoFactor>(cam_pyr_[this->active_level_], kf_, fr_,
                                    pose0_key, pose1_key, code0_key,
                                    this->active_level_, aligner_, update_vld_);
  return graph;
}

template<typename Scalar, int CS>
std::string OptimizePhoto<Scalar,CS>::Name()
{
  std::stringstream ss;
  ss << this->Id() << " OptimizePhoto " << kf_->Name() << " -> " << fr_->Name()
     << " iters = " << (this->active_level_ < 0 ? 0 : this->iters_[this->active_level_])
     << " active_level = " << this->active_level_
     << " new level = " << this->IsNewLevelStart()
     << " finished = " << this->Finished();
  ss << " factor indices = ";
  for (auto& idx : this->last_indices_)
    ss << idx << " ";
  return ss.str();
}

template<typename Scalar, int CS>
bool OptimizePhoto<Scalar,CS>::Involves(OptimizePhoto::FramePtr ptr) const
{
  return fr_ == ptr || kf_ == ptr;
}

// explicit instantiation
template class OptimizePhoto<float,DF_CODE_SIZE>;

/* ************************************************************************* */
template<typename Scalar, int CS>
OptimizeGeo<Scalar,CS>::OptimizeGeo(KeyframePtr kf0, KeyframePtr kf1, int iters,
                                    df::PinholeCamera<Scalar> cam, int num_points,
                                    float huber_delta, bool stochastic)
  : OptimizeWork<Scalar>(iters, false),
    kf0_(kf0), kf1_(kf1),
    cam_(cam),
    num_points_(num_points),
    huber_delta_(huber_delta),
    stochastic_(stochastic) {}

template<typename Scalar, int CS>
OptimizeGeo<Scalar,CS>::~OptimizeGeo() {}

template<typename Scalar, int CS>
gtsam::NonlinearFactorGraph OptimizeGeo<Scalar,CS>::ConstructFactors()
{
  gtsam::NonlinearFactorGraph graph;
  auto pose0_key = PoseKey(kf0_->id);
  auto pose1_key = PoseKey(kf1_->id);
  auto code0_key = CodeKey(kf0_->id);
  auto code1_key = CodeKey(kf1_->id);
  graph.emplace_shared<GeoFactor>(cam_, kf0_, kf1_, pose0_key, pose1_key,
                                  code0_key, code1_key, num_points_,
                                  huber_delta_, stochastic_);
  return graph;
}

template<typename Scalar, int CS>
bool OptimizeGeo<Scalar,CS>::Involves(FramePtr ptr) const
{
  return kf0_ == ptr || kf1_ == ptr;
}

template<typename Scalar, int CS>
std::string OptimizeGeo<Scalar,CS>::Name()
{
  std::stringstream ss;
  ss << this->Id() << " OptimizeGeo " << kf0_->Name() << " -> " << kf1_->Name()
     << " iters = " << this->iters_[this->active_level_]
     << " finished = " << this->Finished();
  return ss.str();
}

// explicit instantiation
template class OptimizeGeo<float,DF_CODE_SIZE>;

/* *************************************************** */
template<typename Scalar, int CS>
OptimizeRep<Scalar,CS>::OptimizeRep(KeyframePtr kf, FramePtr fr, int iters,
                                    df::PinholeCamera<Scalar> cam,
                                    float feature_max_dist, float huber_delta,
                                    float sigma, int maxiters, float threshold)
  : OptimizeWork<Scalar>(iters),
    kf_(kf), fr_(fr),
    cam_(cam), max_dist_(feature_max_dist),
    huber_delta_(huber_delta),
    sigma_(sigma), maxiters_(maxiters),
    threshold_(threshold) {}

template<typename Scalar, int CS>
OptimizeRep<Scalar,CS>::~OptimizeRep() {}

template<typename Scalar, int CS>
gtsam::NonlinearFactorGraph OptimizeRep<Scalar,CS>::ConstructFactors()
{
  gtsam::NonlinearFactorGraph graph;
  gtsam::Key pose0_key = PoseKey(kf_->id);
  gtsam::Key code0_key = CodeKey(kf_->id);
  gtsam::Key pose1_key;

  // check if the second pointer is a keyframe
  if (std::dynamic_pointer_cast<Keyframe<Scalar>>(fr_))
    pose1_key = PoseKey(fr_->id);
  else
    pose1_key = AuxPoseKey(fr_->id);

//  graph.emplace_shared<RepFactor>(cam_, kf_, fr_, pose0_key, pose1_key,
//                                  code0_key, max_dist_, huber_delta_, sigma_);

  boost::shared_ptr<RepFactor> factor = boost::make_shared<RepFactor>(cam_, kf_, fr_, pose0_key, pose1_key,
                                                                      code0_key, max_dist_, huber_delta_, sigma_,
                                                                      maxiters_, threshold_);

  if (factor->Matches().empty())
  {
    LOG(INFO) << "MATCHES ARE EMPTY, NOT ADDING FACTOR!";
    this->finished_ = true;
  }
  else
  {
    graph.add(factor);
  }

  return graph;
}

template<typename Scalar, int CS>
bool OptimizeRep<Scalar,CS>::Finished() const
{
  return OptimizeWork<Scalar>::Finished() || finished_;
}

template<typename Scalar, int CS>
bool OptimizeRep<Scalar,CS>::Involves(FramePtr ptr) const
{
  return fr_ == ptr || kf_ == ptr;
}

template<typename Scalar, int CS>
std::string OptimizeRep<Scalar,CS>::Name()
{
  std::stringstream ss;
  ss << this->Id() << " OptimizeRep " << kf_->Name() << " -> " << fr_->Name()
     << " iters = " << this->iters_[this->active_level_]
     << " finished = " << this->Finished();
  return ss.str();
}

// explicit instantiation
template class OptimizeRep<float,DF_CODE_SIZE>;

} // namespace work
} // namespace df
