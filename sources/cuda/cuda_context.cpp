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
#include "cuda_context.h"

#include <cuda_runtime.h> // runtime api
#include <VisionCore/CUDAException.hpp>
#include <glog/logging.h>

namespace cuda
{

inline void ThrowOnErrorRuntime(const cudaError_t& err, const std::string& msg)
{
  if (err != cudaSuccess)
    throw vc::CUDAException(err, msg);
}

inline void ThrowOnErrorDriver(const CUresult& res, const std::string& msg)
{
  if (res != CUDA_SUCCESS)
  {
    const char *errname, *msgname;

    cuGetErrorName(res, &errname);
    cuGetErrorString(res, &msgname);

    std::string err(errname);
    std::string errmsg(msgname);

    std::stringstream ss;
    ss << msg << ": " << err << ", " << errmsg << std::endl;
    throw std::runtime_error(ss.str());
  }
}

DeviceInfo GetDeviceInfo(uint gpuId)
{
  // probe nvidia for device info
  cudaDeviceProp cdp;
  ThrowOnErrorRuntime(cudaGetDeviceProperties(&cdp, gpuId), "cudaGetDeviceProperties failed");

  // fill out DeviceInfo structure
  DeviceInfo devInfo;
  devInfo.Name = cdp.name;
  devInfo.SharedMemPerBlock = cdp.sharedMemPerBlock;
  devInfo.RegistersPerBlock = cdp.regsPerBlock;
  devInfo.WarpSize = cdp.warpSize;
  devInfo.MemPitch = cdp.memPitch;
  devInfo.MaxThreadsPerBlock = cdp.maxThreadsPerBlock;
  devInfo.MaxThreadsDim[0] = cdp.maxThreadsDim[0];
  devInfo.MaxThreadsDim[1] = cdp.maxThreadsDim[1];
  devInfo.MaxThreadsDim[2] = cdp.maxThreadsDim[2];
  devInfo.MaxGridSize[0] = cdp.maxGridSize[0];
  devInfo.MaxGridSize[1] = cdp.maxGridSize[1];
  devInfo.MaxGridSize[2] = cdp.maxGridSize[2];
  devInfo.TotalConstMem = cdp.totalConstMem;
  devInfo.Major = cdp.major;
  devInfo.Minor = cdp.minor;
  devInfo.ClockRate = cdp.clockRate;
  devInfo.TextureAlignment = cdp.textureAlignment;
  devInfo.DeviceOverlap = cdp.deviceOverlap;
  devInfo.MultiProcessorCount = cdp.multiProcessorCount;
  devInfo.KernelExecTimeoutEnabled = cdp.kernelExecTimeoutEnabled;
  devInfo.Integrated = cdp.integrated;
  devInfo.CanMapHostMemory = cdp.canMapHostMemory;
  devInfo.ComputeMode = cdp.computeMode;
  devInfo.ConcurrentKernels = cdp.concurrentKernels;
  devInfo.ECCEnabled = cdp.ECCEnabled;
  devInfo.PCIBusID = cdp.pciBusID;
  devInfo.PCIDeviceID = cdp.pciDeviceID;
  devInfo.MemoryClockRate = cdp.memoryClockRate;
  devInfo.MemoryBusWidth = cdp.memoryBusWidth;
  devInfo.MaxThreadsPerMultiprocessor = cdp.maxThreadsPerMultiProcessor;

  // probe nvidia for free and total memory
  ThrowOnErrorRuntime(cudaMemGetInfo(&devInfo.FreeGlobalMem, &devInfo.TotalGlobalMem),
                      "cudaMemGetInfo failed");

  return devInfo;
}

DeviceInfo GetCurrentDeviceInfo()
{
  int dev;
  ThrowOnErrorRuntime(cudaGetDevice(&dev), "cudaGetDevice failed");
  return GetDeviceInfo(dev);
}

DeviceInfo Init(uint gpuId)
{
  // explicitly init driver api
  ThrowOnErrorDriver(cuInit(0), "cuInit failed");

  // select gpu
  ThrowOnErrorRuntime(cudaSetDevice(gpuId), "cudaSetDevice failed");

  // get device info
  DeviceInfo devInfo = GetDeviceInfo(gpuId);
  LOG(INFO) << "Selected GPU " << gpuId << ": " << devInfo.Name;

  return devInfo;
}

CUcontext CreateRuntimeContext(std::size_t device_id)
{
  // trick here is to trigger implicit context creation
  // with runtime api
  ThrowOnErrorRuntime(cudaSetDevice(device_id), "cudaSetDevice failed");
  ThrowOnErrorRuntime(cudaFree(0), "cudaFree failed");

  // get and return current context
  CUcontext ctx;
  ThrowOnErrorDriver(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent failed");
  return ctx;
}

CUcontext CreateAndBindContext(std::size_t device_id)
{
  CUcontext ctx;
  ThrowOnErrorDriver(cuCtxCreate(&ctx, 0, device_id), "cuCtxCreate failed");
  return ctx;
}

CUcontext GetCurrentContext()
{
  CUcontext ctx;
  ThrowOnErrorDriver(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent failed");
  return ctx;
}

void SetCurrentContext(const CUcontext& ctx)
{
  ThrowOnErrorDriver(cuCtxSetCurrent(ctx), "cuCtxSetCurrent failed");
}

void PushContext(const CUcontext& ctx)
{
  ThrowOnErrorDriver(cuCtxPushCurrent(ctx), "cuCtxPushCurrent failed");
}

CUcontext PopContext()
{
  CUcontext ctx;
  ThrowOnErrorDriver(cuCtxPopCurrent(&ctx), "cuCtxPopCurrent failed");
  return ctx;
}

} // namespace cuda
