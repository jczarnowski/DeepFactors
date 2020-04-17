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
#ifndef CUDA_CONTEXT_H_
#define CUDA_CONTEXT_H_

#include <cuda.h> // driver api
#include "device_info.h"

namespace cuda
{

DeviceInfo Init(uint gpuId = 0);

DeviceInfo GetDeviceInfo(uint gpuId);
DeviceInfo GetCurrentDeviceInfo();

CUcontext CreateRuntimeContext(std::size_t device_id);

CUcontext CreateAndBindContext(std::size_t device_id);

CUcontext GetCurrentContext();

void SetCurrentContext(const CUcontext& ctx);

void PushContext(const CUcontext& ctx);

CUcontext PopContext();

class ScopedContextPop
{
public:
  ScopedContextPop()
  {
    ctx_ = PopContext();
  }

  ~ScopedContextPop()
  {
    PushContext(ctx_);
  }

private:
  CUcontext ctx_;
};


} // namespace df

#endif // CUDA_CONTEXT_H_

