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
#include "work_manager.h"

namespace df
{
namespace work
{

WorkManager::WorkPtr WorkManager::AddWork(WorkPtr work)
{
  work_.push_back(work);
  work_map_.insert({work->Id(), work});
  return work;
}

void WorkManager::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                              gtsam::FactorIndices &remove_indices,
                              gtsam::Values &var_init)
{
  // factors added this iteration
  // will be used in consuming the factor indices after ISAM2::update step
  last_new_factors_.clear();

  // collect bookkeeping from all work
  for (auto& work : work_)
  {
    gtsam::NonlinearFactorGraph add_factors;
    gtsam::FactorIndices rem_factors;
    gtsam::Values init;
    work->Bookkeeping(add_factors, rem_factors, init);

    // keep track which work items added factors
    if (!add_factors.empty())
      last_new_factors_.insert({work->Id(), add_factors.size()});

    // add new factors, vars, etc
    new_factors += add_factors;
    remove_indices.insert(remove_indices.end(), rem_factors.begin(), rem_factors.end());
    var_init.insert(init);
  }
}

void WorkManager::DistributeIndices(gtsam::FactorIndices indices)
{
  for (auto& kv : last_new_factors_)
  {
    std::string id = kv.first;
    auto work = work_map_[id];
    int n = kv.second;

    // first N goes to id
    gtsam::FactorIndices ind(indices.begin(), indices.begin() + n);
    if (work)
      work->LastFactorIndices(ind);

    // remove
    indices.erase(indices.begin(), indices.begin() + n);
  }
  last_new_factors_.clear();
}

void WorkManager::Remove(std::function<bool (WorkPtr)> f)
{
  auto it = std::remove_if(work_.begin(), work_.end(), f);
  for (auto ii = it; ii != work_.end(); ++ii)
    (*ii)->SignalRemove();
}

void WorkManager::Erase(std::function<bool (WorkPtr)> f)
{
  auto it = std::remove_if(work_.begin(), work_.end(), f);
  work_.erase(it, work_.end());
}

void WorkManager::Update()
{
  VLOG(1) << "[WorkManager::Update] Current work:";
  for (auto& work : work_)
    VLOG(1) << work->Name();

  auto it = work_.begin();
  while (it != work_.end())
  {
    WorkPtr work = *it;

    // in case we need to remove this work
    auto this_it = it;
    it++;

    work->Update();

    // check if the work has finished
    if (work->Finished())
    {
      VLOG(1) << "Work " << work->Id() << " has finished";

      // add its child if it has one
      auto child = work->RemoveChild();
      if (child)
      {
        VLOG(1) << "Adding its child: " << child->Id();
        AddWork(child);
      }
      work_map_.erase(work->Id());
      work_.erase(this_it);
    }
  }
}

void WorkManager::SignalNoRelinearize()
{
  for (auto& work : work_)
    work->SignalNoRelinearize();
}

void WorkManager::PrintWork()
{
  for (auto& work : work_)
    LOG(INFO) << work->Name();
}

void WorkManager::Clear()
{
  work_.clear();
  work_map_.clear();
  last_new_factors_.clear();
}

} // namespace work
} // namespace df
