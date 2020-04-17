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
#ifndef DF_M_ESTIMATORS_H
#define DF_M_ESTIMATORS_H

#include <Eigen/Dense>

namespace df
{

template <typename Scalar>
EIGEN_DEVICE_FUNC
Scalar TukeyWeight(Scalar x, Scalar delta)
{
  Scalar a = delta / x;
  if (abs(x) <= delta)
  {
    Scalar first = 1 - 1 / a / a; // 1 - (delta/x)^2
    return abs(a) * sqrt((1 - first * first * first) / 6.);
  }
  else
  {
    return abs(a) * sqrt(1. / 6.);
  }
}

template <typename Scalar>
EIGEN_DEVICE_FUNC
Scalar CauchyWeight(Scalar x, Scalar delta)
{
  Scalar a = delta / x;
  return abs(a) / sqrt(2.) * sqrt(log(1 + 1 / a / a));
}

template <typename Scalar>
EIGEN_DEVICE_FUNC
Scalar HuberWeight(Scalar x, Scalar delta)
{
  const Scalar aa = abs(x);
  return aa <= delta ? Scalar(1) : sqrt(delta * (2*aa - delta)) / aa;
}

} // namespace df

#endif // DF_M_ESTIMATORS_H
