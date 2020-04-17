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
#include <random>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <sophus/se3.hpp>

#include "random_machine.h"
#include "testing_utils.h"
#include "pinhole_camera.h"
#include "lucas_kanade_se3.h"

template <typename Scalar>
class TestPinholeCamera : public ::testing::Test
{
public:
  typedef df::PinholeCamera<Scalar> CameraT;
  typedef Sophus::SE3<typename CameraT::Type> SE3T;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DynamicMatrix;
  typedef df::RandomMachine<Scalar, df::dist::uniform> RandomT;

  TestPinholeCamera() : rand_(1.0, 10.0)
  {
    test_cam_ = rand_.template RandomCamera<Scalar>();
  }

  CameraT test_cam_;
  RandomT rand_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(TestPinholeCamera, MyTypes);

TYPED_TEST(TestPinholeCamera, ProjectionPointJacobian)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PointT PointT;
  typedef typename CameraT::PixelT PixelT;

  TypeParam eps = 1e-3;
  TypeParam tol = 1e-1;

  // random test point
  PointT pt = this->rand_.template RandomPoint<TypeParam>();

  // evaluate function and jacobian at test point
  Eigen::Matrix<TypeParam, 2, 3> jac = this->test_cam_.ProjectPointJacobian(pt);
  PixelT pt_proj = this->test_cam_.Project(pt);

  // find the findiff jacobian and compare
  for (int i = 0; i < 3; ++i)
  {
    PointT pt_forward(pt);
    pt_forward[i] += eps;

    PixelT pix_front = this->test_cam_.Project(pt_forward);
    PixelT diff_jac = (pix_front - pt_proj) / eps;
    Eigen::Matrix<TypeParam,2,1> our_jac = jac.template block<2,1>(0,i);
    df::CompareWithTol(diff_jac, our_jac, tol);
  }
}

TYPED_TEST(TestPinholeCamera, ReprojectionDepthJacobian)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PointT PointT;
  typedef typename CameraT::PixelT PixelT;

  TypeParam eps = 1e-3;
  TypeParam tol = 1e-1;

  // random test data
  PixelT pix = this->rand_.template RandomPixel<TypeParam>();
  TypeParam depth = this->rand_.template RandomNumber<TypeParam>();

  // evaluate function and jacobian at test point
  PointT pt = this->test_cam_.Reproject(pix, depth);
  Eigen::Matrix<TypeParam, 3, 1> jac = this->test_cam_.ReprojectDepthJacobian(pix, depth);

  // find the findiff jacobian
  PointT pt_front = this->test_cam_.Reproject(pix, depth + eps);
  PointT diff_jac = (pt_front - pt) / eps;

  LOG(INFO) << "findiff:    " << diff_jac.transpose();
  LOG(INFO) << "analytical: " << jac.transpose();

  df::CompareWithTol(diff_jac, jac, tol);
}

TYPED_TEST(TestPinholeCamera, ReprojectionPixelJacobian)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PointT PointT;
  typedef typename CameraT::PixelT PixelT;

  TypeParam eps = 1e-3;
  TypeParam tol = 1e-3;

  // random test data
  PixelT pix = this->rand_.template RandomPixel<TypeParam>();
  TypeParam depth = this->rand_.template RandomNumber<TypeParam>();

  // evaluate function and jacobian at test point
  PointT pt_reproj = this->test_cam_.Reproject(pix, depth);
  Eigen::Matrix<TypeParam, 3, 2> jac_pix = this->test_cam_.ReprojectPixelJacobian(pix, depth);

  // find the findiff jacobian and compare
  for (int i = 0; i < 2; ++i)
  {
    PixelT pix_forward(pix);
    pix_forward[i] += eps;

    PointT pt_front = this->test_cam_.Reproject(pix_forward, depth);
    PointT diff_jac = (pt_front - pt_reproj) / eps;
    Eigen::Matrix<TypeParam,3,1> our_jac = jac_pix.template block<3,1>(0,i);
    df::CompareWithTol(diff_jac, our_jac, tol);
  }
}
