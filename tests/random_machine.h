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
#ifndef DF_RANDOM_MACHINE_H_
#define DF_RANDOM_MACHINE_H_

#include <random>
#include <type_traits>
#include <Eigen/Dense>

#include "pinhole_camera.h"

namespace df
{

namespace dist
{
struct normal;
struct uniform;
}

template <typename RealType, typename DistT>
struct random_distribution;

#define SPECIALIZE_DIST(TYPE, FUNC) \
  template <typename RealType> \
  struct random_distribution<RealType, TYPE> { \
    typedef std::FUNC<RealType> Type; \
  };

SPECIALIZE_DIST(dist::normal, normal_distribution)
SPECIALIZE_DIST(dist::uniform, uniform_real_distribution)

template <typename RandomScalar=double, typename DistType=dist::uniform>
class RandomMachine
{
public:
  typedef typename random_distribution<RandomScalar, DistType>::Type DistT;

  RandomMachine(RandomScalar first, RandomScalar second)
      : mt_(rd_()), dist_(first, second) { }

  template <typename T>
  T RandomNumber()
  {
    return dist_(mt_);
  }

  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> RandomMatrix(int rows, int cols)
  {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        mat(i,j) = RandomNumber<T>();
    return mat;
  }

  template <typename T>
  Eigen::Matrix<T,3,1> RandomPoint()
  {
    return RandomMatrix<T>(3,1);
  }

  template <typename T>
  Eigen::Matrix<T,2,1> RandomPixel()
  {
    return RandomMatrix<T>(2,1);
  }

  template <typename T>
  Sophus::SE3<T> RandomPose()
  {
    return Sophus::SE3<T>::exp(RandomMatrix<T>(6,1));
  }

  template <typename T>
  df::PinholeCamera<T> RandomCamera()
  {
    int w = (int) RandomNumber<T>();
    int h = (int) RandomNumber<T>();
    T u0 = w / 2.;
    T v0 = h / 2.;
    T fx = RandomNumber<T>();
    T fy = RandomNumber<T>();
    return df::PinholeCamera<T>(fx,fy,u0,v0,w,h);
  }

private:
  std::random_device rd_;
  std::mt19937 mt_;
  DistT dist_;
};

} // namespace df

#endif
