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
#ifndef DF_UNIFORM_SAMPLER_H_
#define DF_UNIFORM_SAMPLER_H_

#include<random>
#include<cmath>
#include<chrono>

namespace df
{

struct Point
{
  int x;
  int y;
};

class UniformSampler
{
public:
  UniformSampler(int im_width, int im_height);

  std::vector<Point> SamplePoints(int num_points);

private:
  std::mt19937 generator_;
  std::uniform_real_distribution<double> uniform01_;

  int im_width_;
  int im_height_;
};

} // namespace df

#endif // DF_UNIFORM_SAMPLER_H_
