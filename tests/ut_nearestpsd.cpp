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
#include "nearest_psd.h"

class TestNearestPsd : public ::testing::Test
{
public:
  TestNearestPsd()
  {

  }


};

TEST_F(TestNearestPsd, UnitTest)
{
  Eigen::MatrixXf M = Eigen::MatrixXf::Random(50,50);
  Eigen::MatrixXf B = NearestPsd(M);

  EXPECT_TRUE(IsPsd(B));
}
