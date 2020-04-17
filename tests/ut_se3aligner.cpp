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
#include <memory>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <VisionCore/Buffers/Image2D.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "cu_image_proc.h"
#include "cu_se3aligner.h"
#include "pinhole_camera.h"
#include "testing_utils.h"
#include "lucas_kanade_se3.h"
#include "timing.h"

template <typename Scalar>
class SE3AlignerTest : public ::testing::Test
{
public:
  typedef df::SE3Aligner<Scalar> AlignerT;
  typedef df::PinholeCamera<Scalar> CameraT;
  typedef Sophus::SE3<Scalar> SE3T;

  template <typename Target>
  using ImageBuffer = vc::Image2DManaged<Scalar, Target>;
  template <typename Target>
  using GradientBuffer = vc::Image2DManaged<Eigen::Matrix<Scalar,1,2>, Target>;

  void SetUp()
  {
    LoadTestData();
    PreprocessImages();
    CreateAndFillBuffers(img0_, img1_, dpt0_);
  }

  void LoadTestData(const std::string& img0path = "data/testimg/1047.jpg",
                    const std::string& img1path = "data/testimg/1052.jpg",
                    const std::string& dpt0path = "data/testimg/1047.png")
  {
    img0_ = cv::imread(img0path, cv::IMREAD_GRAYSCALE);
    img1_ = cv::imread(img1path, cv::IMREAD_GRAYSCALE);
    dpt0_ = cv::imread(dpt0path, cv::IMREAD_ANYDEPTH);
    if (img0_.empty() || img1_.empty() || dpt0_.empty())
    {
      std::cout << "Failed loading test images" << std::endl;
      exit(1);
    }
    cam_ = df::GetSceneNetCam<Scalar>(img0_.cols, img0_.rows);
  }

  void PreprocessImages()
  {
    int mat_type = CV_MAKETYPE(vc::internal::OpenCVType<Scalar>::TypeCode, 1);
    dpt0_.convertTo(dpt0_, mat_type, 1 / 1000.0); // convert depth to meters
    img1_.convertTo(img1_, mat_type, 1 / 255.0);
    img0_.convertTo(img0_, mat_type, 1 / 255.0);

    const int blur = 25;
    cv::blur(img0_, img0_, {blur,blur});
    cv::blur(img1_, img1_, {blur,blur});
  }

  void CreateAndFillBuffers(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& dpt0)
  {
    int w = img0.cols;
    int h = img0.rows;

    // initialize buffers
    img0gpu_ = std::make_unique<ImageBuffer<vc::TargetDeviceCUDA>>(w, h);
    img1gpu_ = std::make_unique<ImageBuffer<vc::TargetDeviceCUDA>>(w, h);
    dpt0gpu_ = std::make_unique<ImageBuffer<vc::TargetDeviceCUDA>>(w, h);
    grad1gpu_ = std::make_unique<GradientBuffer<vc::TargetDeviceCUDA>>(w, h);

    // upload data
    img0gpu_->copyFrom(img0);
    img1gpu_->copyFrom(img1);
    dpt0gpu_->copyFrom(dpt0);

    // calculate gradient
    df::SobelGradients(*img1gpu_, *grad1gpu_);
  }

  typename AlignerT::ReductionItem RunAlignerStep(const SE3T& se3)
  {
    return aligner_.RunStep(se3, cam_, *img0gpu_, *img1gpu_, *dpt0gpu_, *grad1gpu_);
  }

  void RunAlignerWarp(const SE3T& se3, vc::Image2DView<Scalar, vc::TargetDeviceGPU>& img2gpu)
  {
    aligner_.Warp(se3, cam_, *img0gpu_, *img1gpu_, *dpt0gpu_, img2gpu);
  }

  cv::Mat img0_;
  cv::Mat img1_;
  cv::Mat dpt0_;

  std::unique_ptr<ImageBuffer<vc::TargetDeviceCUDA>> img0gpu_;
  std::unique_ptr<ImageBuffer<vc::TargetDeviceCUDA>> img1gpu_;
  std::unique_ptr<ImageBuffer<vc::TargetDeviceCUDA>> dpt0gpu_;
  std::unique_ptr<GradientBuffer<vc::TargetDeviceCUDA>> grad1gpu_;

  CameraT cam_;
  AlignerT aligner_;

  // for image alignment tests
  const int align_iters_ = 40;
  const Scalar converge_tol_ = 1e-3;

  // for jacobian tests
  const Scalar start_eps_ = 1e-2;
  const int eps_steps_ = 3;
};

typedef ::testing::Types<float> MyTypes;
TYPED_TEST_CASE(SE3AlignerTest, MyTypes);

TYPED_TEST(SE3AlignerTest, FullJacobianTest)
{
  typedef typename TestFixture::SE3T SE3T;

  TypeParam eps = this->start_eps_;
  for (int i = 0; i < this->eps_steps_; i++)
  {
    LOG(INFO) << "----------------------------------------------";
    LOG(INFO) << "Testing full jacobian with eps = " << eps;

    // verify the jacobians
    SE3T pose_0;

    // get jacobians at the test point 'pose_0'
    auto result_0 = this->RunAlignerStep(pose_0);

    const std::vector<std::string> names = {"tx", "ty", "tz", "rotx", "roty", "rotz"};
    for (int i = 0; i < 6; ++i)
    {
      // perturb current test point
      SE3T pose = df::GetPerturbedPose(pose_0, i, eps);

      // evaluate residual
      auto result = this->RunAlignerStep(pose);
      TypeParam findiff = 0.5 * (result.residual - result_0.residual) / eps;

      // print/check the results
      LOG(INFO) << std::endl;
      LOG(INFO) << names[i] << " derivative:";
      LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Finite difference:"
                << std::setw(10) << std::left << findiff;
      LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Our derivative:"
                << std::setw(10) <<  std::left << result_0.Jtr(i,0);

    }

    eps /= 10;
  }
}

TYPED_TEST(SE3AlignerTest, ImageAlignmentTest)
{
  typedef typename TestFixture::AlignerT::ReductionItem ReductionItem;

  // do optimization
  Sophus::SE3<TypeParam> se3;
  TypeParam average_ms = 0;
  TypeParam error = 0;
  for (int i = 0; i < this->align_iters_; ++i)
  {
    ReductionItem result;
    auto elapsed_ms = MeasureTime([&] () {
      result = this->RunAlignerStep(se3);
    });
    average_ms += elapsed_ms;

    df::SE3SolveAndUpdate(result.JtJ.toDenseMatrix(), result.Jtr, se3);

    error = result.residual / result.inliers;
//     LOG(INFO) << "Step " << i << " error:" << error
//               << " inliers:" << result.inliers / (TypeParam) this->img0gpu_->area() * 100 << "%";

//     // warp to show the img
//     int w = this->img0_.cols;
//     int h = this->img0_.rows;
//     vc::Image2DManaged<TypeParam, vc::TargetDeviceCUDA> img2gpu(w, h);
//     vc::Image2DManaged<TypeParam, vc::TargetHost> img2(w, h);
//     this->RunAlignerWarp(se3, img2gpu);
//     img2.copyFrom(img2gpu);
//     cv::imshow("img0", this->img0_);
//     cv::imshow("img1", this->img1_);
//     cv::imshow("img2", img2.getOpenCV());
//     cv::imshow("res", img2.getOpenCV()-this->img0_);
//     cv::waitKey(0);
  }

  std::cout << "Average time for single step: " << average_ms / this->align_iters_ << " ms" << std::endl;
  ASSERT_LE(error, this->converge_tol_) << "Failed to converge";
}
