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
#ifndef DF_WORK_MANAGER_H_
#define DF_WORK_MANAGER_H_

#include <list>
#include <functional>
#include <glog/logging.h>

#include "work.h"
#include "df_work.h"

namespace df
{
namespace work
{

class WorkManager
{
public:
  typedef Work::Ptr WorkPtr;

  template <typename T, class... Args>
  WorkPtr AddWork(Args&&... args)
  {
    auto work = std::make_shared<T>(std::forward<Args>(args)...);
    return AddWork(work);
  }

  WorkPtr AddWork(WorkPtr work);

  void Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                   gtsam::FactorIndices& remove_indices,
                   gtsam::Values& var_init);

  void DistributeIndices(gtsam::FactorIndices indices);
  void Remove(std::function<bool (WorkPtr)> f);
  void Erase(std::function<bool (WorkPtr)> f);

  void Update();
  void SignalNoRelinearize();

  void PrintWork();
  bool Empty() const { return work_.empty(); }
  void Clear();

private:
  std::list<WorkPtr> work_;
  std::map<std::string, WorkPtr> work_map_;
  std::map<std::string, int> last_new_factors_;
};

} // namespace work
} // namespace df

#endif // DF_WORK_MANAGER_H_
