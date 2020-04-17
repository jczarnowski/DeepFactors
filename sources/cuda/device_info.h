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
#ifndef DF_DEVICE_INFO_H_
#define DF_DEVICE_INFO_H_

namespace cuda
{

struct DeviceInfo
{
  std::string Name;
  std::size_t TotalGlobalMem;
  std::size_t FreeGlobalMem;
  std::size_t SharedMemPerBlock;
  int         RegistersPerBlock;
  int         WarpSize;
  std::size_t MemPitch;
  int         MaxThreadsPerBlock;
  int         MaxThreadsDim[3];
  int         MaxGridSize[3];
  std::size_t TotalConstMem;
  int         Major;
  int         Minor;
  int         ClockRate;
  std::size_t TextureAlignment;
  int         DeviceOverlap;
  int         MultiProcessorCount;
  int         KernelExecTimeoutEnabled;
  int         Integrated;
  int         CanMapHostMemory;
  int         ComputeMode;
  int         ConcurrentKernels;
  int         ECCEnabled;
  int         PCIBusID;
  int         PCIDeviceID;
  int         MemoryClockRate;
  int         MemoryBusWidth;
  int         MaxThreadsPerMultiprocessor;
};

} // namespace cuda

#endif // DF_DEVICE_INFO_H_
