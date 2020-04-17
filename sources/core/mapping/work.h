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
#ifndef DF_WORK_H_
#define DF_WORK_H_

#include <memory>
#include <string>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/base/FastVector.h>

// typedef FactorIndices so that we don't have to include ISAM2 header here
namespace gtsam
{
typedef gtsam::FastVector<size_t> FactorIndices;
}

namespace df
{
namespace work
{

class Work
{
public:
  typedef std::shared_ptr<Work> Ptr;

  Work();

  virtual ~Work();

  // interface that needs to be implemented
  // by child functions
  virtual void Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                           gtsam::FastVector<size_t>& remove_indices,
                           gtsam::Values& var_init) = 0;
  virtual void Update() = 0;
  virtual bool Finished() const = 0;
  virtual std::string Name() = 0;

  virtual void SignalNoRelinearize() {}
  virtual void SignalRemove() {}
  virtual void LastFactorIndices(gtsam::FactorIndices& indices) {}

  template<class T, class... Args>
  Ptr AddChild(Args&&... args)
  {
    auto child = std::make_shared<T>(std::forward<Args>(args)...);
    return AddChild(child);
  }

  Ptr AddChild(Ptr child);
  Ptr RemoveChild();

  std::string Id() const { return id; }

private:
  Ptr child_;
  std::string id;

  static int next_id;
};

} // namespace work
} // namespace df

#endif // DF_WORK_H_
