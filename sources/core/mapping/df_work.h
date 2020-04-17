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
#ifndef DF_DF_WORK_H_
#define DF_DF_WORK_H_

#include "work.h"

// forward declare these some time later
#include "keyframe.h"
#include "cu_sfmaligner.h"
#include "camera_pyramid.h"
#include "photometric_factor.h"
#include "sparse_geometric_factor.h"
#include "reprojection_factor.h"

namespace df
{
namespace work
{

class CallbackWork : public Work
{
public:
  typedef std::function<void()> CallbackT;
  CallbackWork(CallbackT f) : finished_(false), f_(f) {}

  virtual void Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                           gtsam::FactorIndices& remove_indices,
                           gtsam::Values& var_init) override {}

  virtual void Update() override
  {
    f_();
    finished_ = true;
  }

  virtual bool Finished() const override
  {
    return finished_;
  }

  virtual std::string Name() override { return "CallbackWork"; }

  bool finished_;
  CallbackT f_;
};

template <typename Scalar, int CS>
class InitVariables : public Work
{
public:
  typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
  typedef typename df::Frame<Scalar>::Ptr FramePtr;

  // constructor from keyframe
  InitVariables(KeyframePtr kf, Scalar code_prior);

  // constructor from keyframe with zero-pose prior
  InitVariables(KeyframePtr kf, Scalar code_prior, Scalar pose_prior);

  // constructor from frame
  InitVariables(FramePtr fr);

  virtual ~InitVariables();

  virtual void Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                           gtsam::FactorIndices& remove_indices,
                           gtsam::Values& var_init) override;

  virtual void Update() override;

  virtual bool Finished() const override
  {
    return !first_;
  }

  virtual std::string Name() override;

private:
  bool first_;
  gtsam::Values var_init_;
  gtsam::NonlinearFactorGraph priors_;
  std::string name_;
};

// adds support for multi-scale optimization
template <typename Scalar>
class OptimizeWork : public Work
{
public:
  typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
  typedef typename df::Frame<Scalar>::Ptr FramePtr;
  typedef std::vector<int> IterList;

  // constructor for single scale optimization
  OptimizeWork(int iters, bool remove_after = false);

  // constructor for multi scale optimization
  OptimizeWork(IterList iters, bool remove_after = false);

  virtual ~OptimizeWork() {}

  // basic version of this func adds some factors
  // on first run and then keeps track of iterations
  virtual void Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                           gtsam::FactorIndices& remove_indices,
                           gtsam::Values& var_init) override;

  // override this function to create simple
  // non pyramid levels
  virtual gtsam::NonlinearFactorGraph ConstructFactors();

  // counts iterations and descends pyramid levels
  virtual void Update() override;

  virtual bool Finished() const override;

  virtual void SignalNoRelinearize() override;
  virtual void SignalRemove() override;
  virtual void LastFactorIndices(gtsam::FactorIndices& indices) override;

  virtual bool Involves(FramePtr ptr) const = 0;

  bool IsCoarsestLevel() const;
  bool IsNewLevelStart() const;

protected:
  bool remove_;
  bool first_;
  IterList iters_;
  IterList orig_iters_;
  int active_level_;
  gtsam::FactorIndices last_indices_;
  bool remove_after_;
};

// handles photo error
template <typename Scalar, int CS>
class OptimizePhoto : public OptimizeWork<Scalar>
{
public:
  typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
  typedef typename df::Frame<Scalar>::Ptr FramePtr;
  typedef typename df::SfmAligner<Scalar,CS>::Ptr AlignerPtr;
  typedef typename df::CameraPyramid<Scalar> CamPyramid;
  typedef df::PhotometricFactor<Scalar,CS> PhotoFactor;
  typedef typename OptimizeWork<Scalar>::IterList IterList;

  // constructor with a vector of iterations
  OptimizePhoto(KeyframePtr kf, FramePtr fr, IterList iters,
                CamPyramid cam_pyr, AlignerPtr aligner, bool update_vld = false, bool remove_after=false);

  virtual ~OptimizePhoto();

  gtsam::NonlinearFactorGraph ConstructFactors() override;
  virtual bool Involves(FramePtr ptr) const;
  virtual std::string Name();

private:
  KeyframePtr kf_;
  FramePtr fr_;
  df::CameraPyramid<Scalar> cam_pyr_;
  AlignerPtr aligner_;
  bool update_vld_;
};

/*!
 * \brief Class that manages factors for geometric error
 */
template <typename Scalar, int CS>
class OptimizeGeo : public OptimizeWork<Scalar>
{
public:

  typedef typename df::Frame<Scalar>::Ptr FramePtr;
  typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
  typedef df::SparseGeometricFactor<Scalar,CS> GeoFactor;

  // constructor with single number of iterations
  OptimizeGeo(KeyframePtr kf0, KeyframePtr kf1,
              int iters, df::PinholeCamera<Scalar> cam,
              int num_points, float huber_delta, bool stochastic);

  virtual ~OptimizeGeo();

  virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
  virtual bool Involves(FramePtr ptr) const override;
  virtual std::string Name() override;

private:
  KeyframePtr kf0_;
  KeyframePtr kf1_;
  df::PinholeCamera<Scalar> cam_;
  int num_points_;
  float huber_delta_;
  bool stochastic_;
};

template <typename Scalar, int CS>
class OptimizeRep : public OptimizeWork<Scalar>
{
public:
  typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
  typedef typename df::Frame<Scalar>::Ptr FramePtr;
  typedef df::ReprojectionFactor<Scalar,CS> RepFactor;

  // constructor with single number of iterations
  OptimizeRep(KeyframePtr kf, FramePtr fr, int iters,
              df::PinholeCamera<Scalar> cam,
              float feature_max_dist, float huber_delta,
              float sigma, int maxiters, float threshold);

  virtual ~OptimizeRep();

  virtual bool Finished() const override;
  virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
  virtual bool Involves(FramePtr ptr) const override;
  virtual std::string Name() override;

private:
  KeyframePtr kf_;
  FramePtr fr_;
  df::PinholeCamera<Scalar> cam_;
  float max_dist_;
  float huber_delta_;
  float sigma_;
  int maxiters_;
  float threshold_;
  bool finished_ = false;
};

} // namespace work
} // namespace df

#endif // DF_DF_WORK_H_
