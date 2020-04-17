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
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "uniform_sampler.h"

TEST(UniformSampling, ShowSampling)
{
  int w = 800;
  int h = 600;

  df::UniformSampler sampler(w, h);

  for (int npts = 500; npts < 1000; npts += 100)
  {
    cv::Mat img(h,w,CV_8UC3);
    auto pts = sampler.SamplePoints(npts);

    for (auto p : pts)
      cv::circle(img, {p.x, p.y}, 1, {255,0,0});

    cv::imshow("sampling" + std::to_string(npts), img);
  }

  cv::waitKey(0);
}
