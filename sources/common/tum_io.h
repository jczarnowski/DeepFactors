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
#ifndef DF_TUM_IO_H_
#define DF_TUM_IO_H_

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

struct TumPose
{
  double timestamp;
  Eigen::Quaternionf rot;
  Eigen::Vector3f trs;
};

inline std::istream& operator>>(std::istream& is, TumPose& pose)
{
  is >> std::setprecision(16) >> pose.timestamp >> pose.trs(0) >> pose.trs(1) >> pose.trs(2)
     >> pose.rot.x() >> pose.rot.y() >> pose.rot.z() >> pose.rot.w();
  return is;
}

inline std::ostream& operator<<(std::ostream& is, TumPose& pose)
{
  is << std::setprecision(16) << std::to_string(pose.timestamp) << " " << pose.trs(0) << " " << pose.trs(1) << " "
     << pose.trs(2) << " " << pose.rot.x() << " " << pose.rot.y() << " "
     << pose.rot.z() << " " << pose.rot.w();
  return is;
}

#endif // DF_TUM_IO_H_
