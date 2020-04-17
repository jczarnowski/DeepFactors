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
#ifndef DF_TIMING_H_
#define DF_TIMING_H_

#include <ctime>
#include <string>

void EnableTiming(bool enable);
void tic(std::string name);
void toc(std::string name);

template <typename Func>
double MeasureTime(Func f)
{
  auto start = std::clock();
  f();
  return (std::clock()-start) / (double)( CLOCKS_PER_SEC / 1000);
}

template <typename Func>
double MeasureTimeAverage(Func f, std::size_t ntests, bool skip_first=true)
{
  if (skip_first)
    MeasureTime(f);
  double total = 0;
  for (std::size_t i = 0; i < ntests; ++i)
    total += MeasureTime(f);
  return total / ntests;
}

#endif // DF_TIMING_H_
