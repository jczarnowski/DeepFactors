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
#ifndef DF_GTSAM_UTILS_H_
#define DF_GTSAM_UTILS_H_

#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/Key.h>

/*
 * Shorthand functions to get keys for certain keyframe id
 */
inline gtsam::Key AuxPoseKey(std::uint64_t j) { return gtsam::Symbol('a', j); }
inline gtsam::Key PoseKey(std::uint64_t j) { return gtsam::Symbol('p', j); }
inline gtsam::Key CodeKey(std::uint64_t j) { return gtsam::Symbol('c', j); }

#endif // DF_GTSAM_UTILS_H_
