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
#include <ctime>
#include <mutex>
#include <map>
#include <glog/logging.h>

std::map<std::string, std::clock_t> __clocks;
std::mutex clocks_mutex;
bool enable_timing = false;

void EnableTiming(bool enable)
{
  enable_timing = enable;
}

void tic(std::string name)
{
  if (enable_timing)
  {
#ifdef ENABLE_TIMING
    std::lock_guard<std::mutex> lock(clocks_mutex);
    if (__clocks.find(name) != __clocks.end())
      LOG(INFO) << "Warning: restarting an existing clock (" << name << ")";
    __clocks[name] = std::clock();
#else
    LOG(FATAL) << "Requesting enable_timing but it was disabled during compilation!";
#endif
  }
}

void toc(std::string name)
{
  if (enable_timing)
  {
#ifdef ENABLE_TIMING
    std::lock_guard<std::mutex> lock(clocks_mutex);
    if (__clocks.find(name) == __clocks.end())
      LOG(FATAL) << "Trying to use measure a clock that was not started with tic (" << name << ")";

    // print and remove from clocks
    double elapsed_ms = double(std::clock() - __clocks[name]) / CLOCKS_PER_SEC * 1000.0f;
    LOG(INFO) << name << " time: " << elapsed_ms;

    __clocks.erase(name);
#else
    LOG(FATAL) << "Requesting enable_timing but it was disabled during compilation!";
#endif
  }
}
