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
#ifndef DF_FBRISK_H_
#define DF_FBRISK_H_

#include <stdint.h>

#include <string>
#include <vector>

#include <FClass.h>
#include <opencv2/opencv.hpp>

namespace DBoW2 {

/**
 * @brief Interface for Brisk features to work with DBoW2.
 * Inherits from FClass and overrides the required methods.
 */
struct FBrisk : protected FClass
{
  typedef std::vector<unsigned char> TDescriptor;
  typedef const TDescriptor *pDescriptor;
  static const int L = 48;

  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  static double distance(const TDescriptor &a,
                         const TDescriptor &b);

  static std::string toString(const TDescriptor &a);

  static void fromString(TDescriptor& a, const std::string& s);

  static void toMat32F(const std::vector<TDescriptor>& descriptors,
                       cv::Mat& mat);
};

} // namespace DBoW2

#endif // DF_FBRISK_H_
