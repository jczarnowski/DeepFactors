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

#include "testing_utils.h"
#include "random_machine.h"
#include "pinhole_camera.h"
#include "lucas_kanade_se3.h"

template <typename T>
struct FindiffParams;

template <>
struct FindiffParams<float>
{
  static float Eps() { return 1e-3; }
  static float Tol() { return 1e-2; }
};

template <>
struct FindiffParams<double>
{
  static double Eps() { return 1e-6; }
  static double Tol() { return 5e-4; }
};

template <typename Scalar>
class TestWarping : public ::testing::Test
{
public:
  typedef df::PinholeCamera<Scalar> CameraT;
  typedef Sophus::SE3<typename CameraT::Type> SE3T;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DynamicMatrix;
  typedef df::RandomMachine<Scalar, df::dist::uniform> RandomT;

  TestWarping() : rand_(3.0, 10.0)
  {
    width = 800;
    height = 600;
    test_cam_ = df::GetSceneNetCam<Scalar>(width, height);
  }

  CameraT test_cam_;
  RandomT rand_;

  std::size_t width;
  std::size_t height;
};

typedef ::testing::Types<double> MyTypes;
TYPED_TEST_CASE(TestWarping, MyTypes);

TYPED_TEST(TestWarping, TransformJacobianPose)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PointT PointT;
  typedef typename TestFixture::SE3T SE3T;

  // random test point
  PointT pt = this->rand_.template RandomPoint<TypeParam>();
  SE3T pose = this->rand_.template RandomPose<TypeParam>();

  // transformed point
  PointT tpt = pose * pt;

  // jacobian
  Eigen::Matrix<TypeParam,3,6> dXdT = df::TransformJacobianPose(pt, pose);

  TypeParam eps = 1e-5;
  TypeParam tol = 1e-2;
  for (int i = 0; i < 6; ++i)
  {
    // perturb transformation
    SE3T pose_forward = df::GetPerturbedPose(pose, i, eps);

    PointT tpt_front = pose_forward * pt;
    PointT diff_jac = (tpt_front - tpt) / eps;
    Eigen::Matrix<TypeParam,3,1> our_jac = dXdT.template block<3,1>(0,i);
    df::CompareWithTol(diff_jac, our_jac, tol);
  }
}

TYPED_TEST(TestWarping, TransformJacobianPoint)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PointT PointT;
  typedef typename TestFixture::SE3T SE3T;

  // random test point
  PointT pt = this->rand_.template RandomPoint<TypeParam>();
  SE3T pose = this->rand_.template RandomPose<TypeParam>();

  // transformed point
  PointT tpt = pose * pt;

  // jacobian
  Eigen::Matrix<TypeParam,3,3> dXdT = df::TransformJacobianPoint(pt, pose);

  TypeParam eps = 1e-3;
  TypeParam tol = 1e-2;
  for (int i = 0; i < 3; ++i)
  {
    // perturb transformation
    PointT pt_forward(pt);
    pt_forward[i] += eps;

    PointT tpt_front = pose * pt_forward;
    PointT diff_jac = (tpt_front - tpt) / eps;
    Eigen::Matrix<TypeParam,3,1> our_jac = dXdT.template block<3,1>(0,i);
    df::CompareWithTol(diff_jac, our_jac, tol);
  }
}

TYPED_TEST(TestWarping, DepthJacobianPrx)
{
  TypeParam avg_dpt = TypeParam(2);

  TypeParam prx = this->rand_.template RandomNumber<TypeParam>();
  TypeParam dpt = df::ProxToDepth(prx, avg_dpt);

  TypeParam dpt_J_prx = df::DepthJacobianPrx(dpt, avg_dpt);

  TypeParam eps = FindiffParams<TypeParam>::Eps();
  TypeParam tol = FindiffParams<TypeParam>::Tol();
  TypeParam dpt_front = df::ProxToDepth(prx + eps, avg_dpt);

  TypeParam findiff = (dpt_front - dpt) / eps;
  EXPECT_NEAR(findiff, dpt_J_prx, tol);
}

TYPED_TEST(TestWarping, RelativePose)
{
  typedef typename TestFixture::SE3T SE3T;
  typedef Eigen::Matrix<TypeParam,SE3T::DoF,SE3T::DoF> PoseJac;

  SE3T pose0 = this->rand_.template RandomPose<TypeParam>();
  SE3T pose1 = this->rand_.template RandomPose<TypeParam>();

//   Eigen::Matrix<TypeParam,6,1> twist0, twist1;
//   twist0 << 0.221406,-0.108413,-0.112945, 0, 0, 0;
//   twist1 << 0.00161003, 0.00210592, 0.000381243, 0, 0, 0;
//   SE3T pose0 = SE3T::exp(twist0);
//   SE3T pose1 = SE3T::exp(twist1);

  PoseJac jac_pose0;
  PoseJac jac_pose1;
  SE3T pose01 = df::RelativePose(pose0, pose1, jac_pose0, jac_pose1);

  LOG(INFO) << "Testing poses:";
  LOG(INFO) << "pose0 = " << pose0;
  LOG(INFO) << "pose1 = " << pose1;

  TypeParam eps = 1e-6;
  TypeParam tol = 1e-5;

  LOG(INFO);
  LOG(INFO) << "Jacobian w.r.t first pose";
  LOG(INFO) << "----------------------------";

  // check jacobian w.r.t pose0
  Eigen::Matrix<TypeParam,6,6> findiff_pose0;
  for (int i = 0; i < SE3T::DoF; ++i)
  {
    SE3T pose0_forward = df::GetPerturbedPose(pose0, i, eps);
    SE3T pose01_forward = df::RelativePose(pose0_forward, pose1);

    // use the proper difference here (boxminus)
    findiff_pose0.template block<3,1>(0,i) = (pose01_forward.translation() - pose01.translation()) / eps;
    findiff_pose0.template block<3,1>(3,i) = (pose01_forward.so3() * pose01.so3().inverse()).log() / eps;
  }
  LOG(INFO) << "findiff:    " << std::endl << std::setfill(' ') << findiff_pose0;
  LOG(INFO) << "analytical: " << std::endl << std::setfill(' ') << jac_pose0;
  df::CompareWithTol(findiff_pose0, jac_pose0, tol);

  LOG(INFO);
  LOG(INFO) << "Jacobian w.r.t second pose";
  LOG(INFO) << "----------------------------";

  // check jacobian w.r.t pose1
  Eigen::Matrix<TypeParam,6,6> findiff_pose1;
  for (int i = 0; i < SE3T::DoF; ++i)
  {
    SE3T pose1_forward = df::GetPerturbedPose(pose1, i, eps);
    SE3T pose01_forward = df::RelativePose(pose0, pose1_forward);

    // use the proper difference here (boxminus)
    findiff_pose1.template block<3,1>(0,i) = (pose01_forward.translation() - pose01.translation()) / eps;
    findiff_pose1.template block<3,1>(3,i) = (pose01_forward.so3() * pose01.so3().inverse()).log() / eps;
  }
  LOG(INFO) << "findiff:    " << std::endl << std::setfill(' ') << findiff_pose1;
  LOG(INFO) << "analytical: " << std::endl << std::setfill(' ') << jac_pose1;
  df::CompareWithTol(findiff_pose1, jac_pose1, tol);
}

TYPED_TEST(TestWarping, FindCorrespondenceJacobianPose)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PixelT PixelT;
  typedef typename TestFixture::SE3T SE3T;

  // get scenennet camera for test
  int w = 800;
  int h = 600;
  TypeParam dpt = 5;
  std::size_t x = w / 2 - 20;
  std::size_t y = h / 2 + 10;
  CameraT cam = df::GetSceneNetCam<TypeParam>(w, h);

  // get a pose for testing
  Eigen::Matrix<TypeParam,6,1> twist;
  twist << 1.0, 0.0, 0.5, 0, -0.05, 0;
  SE3T pose = SE3T::exp(twist);

  LOG(INFO) << cam;
  LOG(INFO) << "dpt: " << dpt;
  LOG(INFO) << "x,y = " << x << ", " << y;
  LOG(INFO) << pose;

  // find jacobian
  df::Correspondence<TypeParam> corresp =
      df::FindCorrespondence(x, y, dpt, cam, pose);
  Eigen::Matrix<TypeParam,2,6> jacobian = df::FindCorrespondenceJacobianPose(corresp, dpt, cam, pose);

  EXPECT_TRUE(corresp.valid);

  TypeParam eps = FindiffParams<TypeParam>::Eps();
  TypeParam tol = FindiffParams<TypeParam>::Tol();
  const std::vector<std::string> names = {"tx", "ty", "tz", "rotx", "roty", "rotz"};
  for (int i = 0; i < 6; ++i)
  {
    SE3T pose_forward = df::GetPerturbedPose(pose, i, eps);
    df::Correspondence<TypeParam> corresp_forward = df::FindCorrespondence(x,y, dpt, cam, pose_forward);

    PixelT findiff_jac = (corresp_forward.pix1 - corresp.pix1) / eps;
    Eigen::Matrix<TypeParam,2,1> our_jac = jacobian.template block<2,1>(0,i);

    // print/check the results
    LOG(INFO) << std::endl;
    LOG(INFO) << names[i] << " derivative:";
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Finite difference:"
              << std::setw(10) << std::left << findiff_jac.transpose();
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Our derivative:"
              << std::setw(10) <<  std::left << our_jac.transpose();

    df::CompareWithTol(findiff_jac, our_jac, tol);
  }

}

TYPED_TEST(TestWarping, FindCorrespondenceJacobianDepth)
{
  typedef typename TestFixture::CameraT CameraT;
  typedef typename CameraT::PixelT PixelT;
  typedef typename TestFixture::SE3T SE3T;

  // get a pose for testing
  Eigen::Matrix<TypeParam,6,1> twist;
  twist << 1.0, 0.0, 0.5, 0, -0.05, 0;
  SE3T pose = SE3T::exp(twist);

  TypeParam eps = FindiffParams<TypeParam>::Eps();
  TypeParam tol = FindiffParams<TypeParam>::Tol();

  int skip = 30;
  for (std::size_t x = 0; x < this->width; x += skip)
  {
    for (std::size_t y = 0; y < this->height; y += skip)
    {
      TypeParam dpt = this->rand_.template RandomNumber<TypeParam>();
      TypeParam dpt_front = dpt + eps;

      df::Correspondence<TypeParam> corresp = df::FindCorrespondence(x, y, dpt, this->test_cam_, pose);
      Eigen::Matrix<TypeParam,2,1> jacobian;
      df::FindCorrespondenceJacobianDepth(corresp,  dpt, this->test_cam_, pose, jacobian);
      df::Correspondence<TypeParam> corresp_forward = df::FindCorrespondence(x, y, dpt_front, this->test_cam_, pose);

      if (!corresp.valid || !corresp_forward.valid)
        continue;

      PixelT findiff_jac = (corresp_forward.pix1 - corresp.pix1) / eps;

      // print/check the results
      df::CompareWithTol(findiff_jac, jacobian, tol);
//       LOG(INFO) << std::endl;
//       LOG(INFO) << "findiff:    " << findiff_jac.transpose();
//       LOG(INFO) << "analytical: " << jacobian.transpose();
    }
  }
}

TYPED_TEST(TestWarping, TransformedPointJacobianGlobalPose)
{
  typedef typename TestFixture::SE3T SE3T;
  typedef Eigen::Matrix<TypeParam,SE3T::DoF,SE3T::DoF> PoseJacPose;
  typedef Eigen::Matrix<TypeParam,3,1> Vec3;
  typedef Eigen::Matrix<TypeParam,3,6> PointJacPose;

  // get test data
  Vec3 pt = this->rand_.template RandomPoint<TypeParam>();
  SE3T pose0 = this->rand_.template RandomPose<TypeParam>();
  SE3T pose1 = this->rand_.template RandomPose<TypeParam>();

//   Eigen::Matrix<TypeParam,6,1> twist0, twist1;
//   twist0 << 0.221406,-0.108413,-0.112945, 0, 0, 0;
//   twist1 << 0.00161003, 0.00210592, 0.000381243, 0, 0, 0;
//   SE3T pose0 = SE3T::exp(twist0);
//   SE3T pose1 = SE3T::exp(twist1);

  LOG(INFO) << "pose0 = " << pose0;
  LOG(INFO) << "pose1 = " << pose1;

  // findif check parameters
  TypeParam eps = FindiffParams<TypeParam>::Eps();
  TypeParam tol = FindiffParams<TypeParam>::Tol();

  // calculate jacobian at point X, pose0, pose1
  PoseJacPose jac_pose0;
  PoseJacPose jac_pose1;
  SE3T pose10 = df::RelativePose(pose0, pose1, jac_pose0, jac_pose1);
  Vec3 tpt = pose10 * pt;

  PointJacPose pt_J_relpose = df::TransformJacobianPose(pt, pose10);
  PointJacPose pt_J_pose0 = pt_J_relpose * jac_pose0;
  PointJacPose pt_J_pose1 = pt_J_relpose * jac_pose1;

  LOG(INFO) << "Jacobian w.r.t first pose";
  LOG(INFO) << "----------------------------";

  // check jacobian w.r.t pose0
  PointJacPose findiff_pose0;
  for (int i = 0; i < SE3T::DoF; ++i)
  {
    SE3T pose0_forward = df::GetPerturbedPose(pose0, i, eps);
    SE3T pose_10_forward= df::RelativePose(pose0_forward, pose1);
    Vec3 tpt_forward = pose_10_forward * pt;

    findiff_pose0.template block<3,1>(0,i) = (tpt_forward - tpt) / eps;
  }

  LOG(INFO) << "findiff:    " << std::endl << std::setfill(' ') << findiff_pose0;
  LOG(INFO) << "analytical: " << std::endl << std::setfill(' ') << pt_J_pose0;
  df::CompareWithTol(findiff_pose0, pt_J_pose0, tol);

  LOG(INFO) << "Jacobian w.r.t second pose";
  LOG(INFO) << "----------------------------";

  // check jacobian w.r.t pose1
  PointJacPose findiff_pose1;
  for (int i = 0; i < SE3T::DoF; ++i)
  {
    SE3T pose1_forward = df::GetPerturbedPose(pose1, i, eps);
    SE3T pose_10_forward= df::RelativePose(pose0, pose1_forward);
    Vec3 tpt_forward = pose_10_forward * pt;

    findiff_pose1.template block<3,1>(0,i) = (tpt_forward - tpt) / eps;
  }

  LOG(INFO) << "findiff:    " << std::endl << std::setfill(' ') << findiff_pose1;
  LOG(INFO) << "analytical: " << std::endl << std::setfill(' ') << pt_J_pose1;
  df::CompareWithTol(findiff_pose1, pt_J_pose1, tol);
}
