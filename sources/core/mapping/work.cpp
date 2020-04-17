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
#include "work.h"

namespace df
{
namespace work
{

int Work::next_id = 0;

Work::~Work() {}

Work::Work()
{
  id = "[" + std::to_string(next_id++) + "]";
}

Work::Ptr Work::AddChild(Ptr child)
{
  child_ = child;
  return child;
}

Work::Ptr Work::RemoveChild()
{
  auto child = child_;
  child_.reset();
  return child;
}

} // namespace work
} // namespace df
