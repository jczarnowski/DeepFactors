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
#ifndef DF_INTERP_H_
#define DF_INTERP_H_

#include <Eigen/Dense>
#include <sophus/se3.hpp>

template <typename T>
T lerp(float x, const T& a, const T& b)
{
  return x*a + (1-x)*b;
}

template <typename T>
T SmoothDecay(const T& from, const T& to, float rate, float dt)
{
  return from + (to-from) * rate * dt;
}

/*
 * Game Programming Gems 4, Chapter 1.10
 * Critically damped spring
 */
template <typename T>
T SmoothDamp(const T& from,
             const T& to,
             T& vel,
             float delta_time,
             float smooth_time,
             float max_speed)
{
  float omega = 2.f / smooth_time;
  float x = omega * delta_time;
  float exp = 1.f / (1.f + x + 0.48f * x * x + 0.235f * x * x * x);

  T change = from - to;
  float max_change = max_speed * delta_time;
  if (change.norm() > max_change)
    change = (change / change.norm()) * max_change;
  T temp = (vel + omega * change) * delta_time;
  vel = (vel - omega * temp) * exp;
  return to + (change + temp) * exp;
}

/*
 * Hacky damped spring for Quaternions but it works
 * (tracks velocity over x y z and w of the quat...)
 * https://gist.github.com/sketchpunk/3568150a04b973430dfe8fd29bf470c8
 */
Eigen::Quaternionf QuatDamp(const Eigen::Quaternionf& from,
                            const Eigen::Quaternionf& to,
                            Eigen::Vector4f& vel,
                            float delta_time,
                            float damping,
                            float max_speed);

/**
 * @brief Interpolate  Linear interpolation for poses.
 */
Sophus::SE3f Interpolate(float x, const Sophus::SE3f& pose0, const Sophus::SE3f& pose1);

#endif // DF_INTERP_H_
