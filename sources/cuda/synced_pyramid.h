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
#ifndef DF_SYNCED_PYRAMID_H_
#define DF_SYNCED_PYRAMID_H_

#include <glog/logging.h>
#include <VisionCore/Buffers/BufferPyramid.hpp>

#define LOG_LEVEL_COPY 1

namespace df
{

template <typename T>
class SyncedBufferPyramid
{
public:
  typedef vc::RuntimeBufferPyramidManaged<T, vc::TargetHost> CpuBuffer;
  typedef vc::RuntimeBufferPyramidManaged<T, vc::TargetDeviceCUDA> GpuBuffer;
  typedef std::shared_ptr<SyncedBufferPyramid<T>> Ptr;

  SyncedBufferPyramid() = delete;

  SyncedBufferPyramid(std::size_t pyramid_levels, std::size_t width, std::size_t height)
      : gpu_modified_(false),
        cpu_modified_(false),
        pyramid_levels_(pyramid_levels),
        width_(width),
        height_(height) {}

  virtual ~SyncedBufferPyramid() {}

  SyncedBufferPyramid(const SyncedBufferPyramid<T>& other)
   {
     pyramid_levels_ = other.pyramid_levels_;
     width_ = other.width_;
     height_ = other.height_;
     code_size_ = other.code_size_;
     gpu_modified_ = other.gpu_modified_;
     cpu_modified_ = other.cpu_modified_;

     if (other.cpu_buf_)
     {
       cpu_buf_ = std::make_shared<CpuBuffer>(pyramid_levels_, width_, height_);
       cpu_buf_->copyFrom(*other.cpu_buf_);
     }

     if (other.gpu_buf_)
     {
       gpu_buf_ = std::make_shared<GpuBuffer>(pyramid_levels_, width_, height_);
       gpu_buf_->copyFrom(*other.gpu_buf_);
     }
   }

  //////////////////////////////////////////////////////////////////////////////
  // General access methods
  //////////////////////////////////////////////////////////////////////////////

  std::shared_ptr<const GpuBuffer> GetGpu() const
  {
    VLOG(LOG_LEVEL_COPY) << "Requesting const GPU pointer";
    SynchronizeGpu();
    return gpu_buf_;
  }

  std::shared_ptr<const CpuBuffer> GetCpu() const
  {
    VLOG(LOG_LEVEL_COPY) << "Requesting const CPU pointer";
    SynchronizeCpu();
    return cpu_buf_;
  }

  std::shared_ptr<CpuBuffer> GetCpuMutable()
  {
    VLOG(LOG_LEVEL_COPY) << "Requesting mutable CPU pointer";
    SynchronizeCpu();
    FlagCpuNewData();
    return cpu_buf_;
  }

  std::shared_ptr<GpuBuffer> GetGpuMutable()
  {
    VLOG(LOG_LEVEL_COPY) << "Requesting mutable GPU pointer";
    SynchronizeGpu();
    FlagGpuNewData();
    return gpu_buf_;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pyramid level accessors
  //////////////////////////////////////////////////////////////////////////////

  const typename CpuBuffer::ViewType& GetCpuLevel(int lvl) const
  {
    return GetCpu()->operator[](lvl);
  }

  typename CpuBuffer::ViewType& GetCpuLevel(int lvl)
  {
    return GetCpuMutable()->operator[](lvl);
  }

  const typename GpuBuffer::ViewType& GetGpuLevel(int lvl) const
  {
    return GetGpu()->operator[](lvl);
  }

  typename GpuBuffer::ViewType& GetGpuLevel(int lvl)
  {
    return GetGpuMutable()->operator[](lvl);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Synchronization management
  //////////////////////////////////////////////////////////////////////////////

  void FlagCpuNewData()
  {
    cpu_modified_ = true;
    CheckDivergence();
    VLOG(LOG_LEVEL_COPY) << "CPU buffer has been modified";
  }

  void FlagGpuNewData()
  {
    gpu_modified_ = true;
    CheckDivergence();
    VLOG(LOG_LEVEL_COPY) << "GPU buffer has been modified";
  }

  bool IsSynchronized() { return !gpu_modified_ && !cpu_modified_; }
  void UnloadCpu() { cpu_buf_.reset(); FlagCpuNewData(); }
  void UnloadGpu() { gpu_buf_.reset(); FlagGpuNewData(); }

  void CheckDivergence() const
  {
    if (gpu_modified_ && cpu_modified_)
      LOG(FATAL) << "Doth CPU and GPU data modified! Not sure which is valid";
  }

  void SynchronizeCpu() const
  {
    // if cpu kf hasnt been yet created
    if (!cpu_buf_)
    {
      cpu_buf_ = std::make_shared<CpuBuffer>(pyramid_levels_, width_, height_);
      VLOG(LOG_LEVEL_COPY) << "Allocating a CPU buffer";
    }

    CheckDivergence();

    // if there is new data from cpu, copy it to gpu
    if (gpu_modified_)
    {
      VLOG(LOG_LEVEL_COPY) << "Copying buffer from GPU to CPU";
      cpu_buf_->copyFrom(*gpu_buf_);
    }

    // we've synchronized so no new data on gpu
    gpu_modified_ = false;
  }

  void SynchronizeGpu() const
  {
    // if cpu kf hasnt been yet created
    if (!gpu_buf_)
    {
      gpu_buf_ = std::make_shared<GpuBuffer>(pyramid_levels_, width_, height_);
      VLOG(LOG_LEVEL_COPY) << "Allocating a GPU buffer";
    }

    CheckDivergence();

    // if there is new data from cpu, copy it to gpu
    if (cpu_modified_)
    {
      VLOG(LOG_LEVEL_COPY) << "Copying buffer from CPU to GPU";
      gpu_buf_->copyFrom(*cpu_buf_);
    }

    // we've synchronized so no new data on cpu
    cpu_modified_ = false;
  }

  std::size_t Width() const { return width_; }
  std::size_t Height() const { return height_; }
  std::size_t Levels() const { return pyramid_levels_; }
  std::size_t CodeSize() const { return code_size_; }
  std::size_t Area() const { return Width() * Height(); }

private:
  mutable std::shared_ptr<CpuBuffer> cpu_buf_;
  mutable std::shared_ptr<GpuBuffer> gpu_buf_;

  mutable bool gpu_modified_;   // gpu nas new data
  mutable bool cpu_modified_;   // cpu has new data

  std::size_t pyramid_levels_;
  std::size_t width_;
  std::size_t height_;
  std::size_t code_size_;
};

} // namespace df

#endif // DF_SYNCED_PYRAMID_H_
