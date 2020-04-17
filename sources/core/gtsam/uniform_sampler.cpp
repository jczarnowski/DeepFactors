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
#include "uniform_sampler.h"

namespace df
{

UniformSampler::UniformSampler(int im_width, int im_height)
    : uniform01_(0.0, 1.0), im_width_(im_width), im_height_(im_height)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator_ = std::mt19937(seed);
}

std::vector<Point> UniformSampler::SamplePoints(int num_points)
{
  std::vector<Point> pts;
  for (int i = 0; i < num_points; ++i)
  {
    Point p;
    p.x = im_width_ * uniform01_(generator_);
    p.y = im_height_ * uniform01_(generator_);
    pts.push_back(p);
  }
  return pts;
}

} // namespace df
