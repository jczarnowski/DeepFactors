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
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>

#include "pinhole_camera.h"
#include "decoder_network.h"
#include "cu_sfmaligner.h"
#include "cu_se3aligner.h"
#include "cu_image_proc.h"
#include "cuda_context.h"
#include "dense_sfm.h"
#include "keyframe.h"
#include "warping.h"
#include "timing.h"

#include "testing_utils.h"
#include "display_utils.h"

class TestSfmAligner : public ::testing::Test
{
public:
  static constexpr int CS = 32;
  static constexpr const char* TEST_CFG_PATH = "data/nets/scannet256_32.cfg";
  static constexpr const char* IMG0_PATH = "data/testimg/0.jpg";
  static constexpr const char* IMG1_PATH = "data/testimg/25.jpg";

  typedef Eigen::Matrix<float,1,2> Grad;
  typedef vc::Image2DManaged<float, vc::TargetHost> ImageBuf;
  typedef vc::Image2DManaged<float, vc::TargetDeviceGPU> ImageBufGpu;
  typedef vc::Image2DManaged<Grad, vc::TargetHost> GradBuf;
  typedef df::SfmAligner<float,CS> AlignerT;
  typedef df::SE3Aligner<float> Se3AlignerT;
  typedef AlignerT::ReductionItem ReductionItem;

  void SetUp()
  {
    cuda::Init();

    CreateAndInitNetwork(TEST_CFG_PATH);

    width = netcfg_.input_width;
    height = netcfg_.input_height;
    codesize = netcfg_.code_size;
    pyrlevels = netcfg_.pyramid_levels;
    avg_dpt = netcfg_.avg_dpt;

    auto cam = netcfg_.camera;
    cam_ = df::PinholeCamera<float>(cam.fx, cam.fy, cam.u0, cam.v0, width, height);

    sfm_params.avg_dpt = avg_dpt;
    sfm_params.huber_delta = 0.5;

    LoadPreprocessData();

    df::SfmAlignerParams aligner_params;
    aligner_params.sfmparams = sfm_params;
    aligner_ = std::make_shared<AlignerT>(aligner_params);
    se3aligner_ = std::make_shared<Se3AlignerT>();
  }

  void CreateAndInitNetwork(const std::string& cfgpath)
  {
    cuda::ScopedContextPop pop;
    netcfg_ = df::LoadJsonNetworkConfig(cfgpath);
    network_ = std::make_shared<df::DecoderNetwork>(netcfg_);

    CHECK_EQ(CS, netcfg_.code_size) << "Tests were compiled for code size " << CS;
  }

  void LoadPreprocessData()
  {
    // load in an example scenenet camera. It will be resized to
    // the actual input image size in LoadPreprocessImage
    df::PinholeCamera<float> in_cam = df::GetSceneNetCam<float>(800, 600);

    img0_ = df::LoadPreprocessImage(IMG0_PATH, in_cam, cam_);
    img1_ = df::LoadPreprocessImage(IMG1_PATH, in_cam, cam_);

     const int blur = 25;
     cv::blur(img0_, img0_, {blur,blur});
     cv::blur(img1_, img1_, {blur,blur});

    // allocate buffers
    img0buf_ = std::make_shared<ImageBuf>(width, height);
    img1buf_ = std::make_shared<ImageBuf>(width, height);
    img0_gpu_ = std::make_shared<ImageBufGpu>(width, height);
    img1_gpu_ = std::make_shared<ImageBufGpu>(width, height);

    // copy images to buffers
    img0buf_->copyFrom(ImageBuf::ViewT(img0_));
    img1buf_->copyFrom(ImageBuf::ViewT(img1_));

    // copy images to gpu
    img0_gpu_->copyFrom(*img0buf_);
    img1_gpu_->copyFrom(*img1buf_);
  }

  df::DenseSfmParams sfm_params;
  df::PinholeCamera<float> cam_;
  df::DecoderNetwork::NetworkConfig netcfg_;
  AlignerT::Ptr aligner_;
  Se3AlignerT::Ptr se3aligner_;
  df::DecoderNetwork::Ptr network_;

  cv::Mat img0_;
  cv::Mat img1_;
  std::shared_ptr<ImageBuf> img0buf_;
  std::shared_ptr<ImageBuf> img1buf_;
  std::shared_ptr<ImageBufGpu> img0_gpu_;
  std::shared_ptr<ImageBufGpu> img1_gpu_;

  int width;
  int height;
  int codesize;
  int pyrlevels;
  float avg_dpt;
};

TEST_F(TestSfmAligner, CorrespondenceJacobianCode)
{
  // get zero code
  Eigen::VectorXf code(codesize);
  code.setZero();

  // get relpose
  Eigen::Matrix<float,3,1> trs;
  Eigen::Matrix<float,3,1> rot;
  rot.setZero();
  trs.setZero();
  rot[1] = -0.67;
  rot[0] = 0.67;
  trs[0] = 1.5;
  trs[1] = 1.5;
  Sophus::SE3f pose(Sophus::SO3f::exp(rot), trs);
  pose = pose.inverse();

  // decode zero code
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> pyr_prx(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> pyr_jac(pyrlevels, width*codesize, height);
  {
    cuda::ScopedContextPop pop;
    network_->Decode(*img0buf_, code, &pyr_prx, nullptr, &pyr_jac);
  }

  // perturb each code element
  float eps = 1e-3;
  float tol = 1e-1;
  for (int i = 0; i < codesize; ++i)
  {
//    LOG(INFO);
    LOG(INFO) << "Checking code entry " << i;
//    LOG(INFO) << "-----------------------------------------------";
    Eigen::VectorXf code_front(code);
    code_front[i] += eps;

    auto dpt_code1 = df::RenderDpt(width, height, avg_dpt, code_front,
                                      pyr_jac[0], pyr_prx[0]);

    // check each pixel
    int skip = 5;
    for (int x = 0; x < width; x += skip)
    {
      for (int y = 0; y < height; y += skip)
      {
        float dpt = df::ProxToDepth(pyr_prx[0](x,y), avg_dpt);
        float dpt_front = dpt_code1(x,y);

        // calculate the corresp at the test point and the analytical jacobian
        Eigen::Map<Eigen::Matrix<float,1,CS>> prx_J_cde(&pyr_jac[0](x*codesize,y));

        df::Correspondence<float> corresp = df::FindCorrespondence(x, y, dpt, cam_, pose);
        Eigen::Matrix<float,2,CS> jac;
        df::FindCorrespondenceJacobianCode(corresp, dpt, cam_, pose,
                                              prx_J_cde, avg_dpt, jac);

        // calculate correspondence at the perturbed code
        auto corresp_front = df::FindCorrespondence(x, y, dpt_front, cam_, pose);

        if (!corresp_front.valid || !corresp.valid)
          continue;

        // calculate findiff
        Eigen::MatrixXf findiff = (corresp_front.pix1 - corresp.pix1) / eps;
        Eigen::MatrixXf analytical = jac.block<2,1>(0,i);

        // compare
        df::CompareWithTol(findiff, analytical, tol);

         const Eigen::IOFormat fmt(4, 0, "\t", "", "", "", "", "");
//         LOG(INFO);
//         LOG(INFO) << "moving depth: " << dpt_front - dpt;
//         LOG(INFO) << "(" << x << ", " << y << ")";
//         LOG(INFO) << "findiff:    " << findiff.transpose().format(fmt);
//         LOG(INFO) << "analytical: " << jac.block<2,1>(0,i).transpose().format(fmt);
      }
    }
  }
}

TEST_F(TestSfmAligner, RunStepDummyData)
{
  Sophus::SE3f pose0, pose1;
  Eigen::Matrix<float,CS,1> code;
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> img0_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> img1_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> dpt0_gpu(width, height);
  vc::Image2DManaged<Grad, vc::TargetDeviceCUDA>  grad1_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> std0_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> vld0_gpu(width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_prx_gpu(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_jac_gpu(pyrlevels, width*codesize, height);

  auto gpu_result = aligner_->RunStep(pose0, pose1, code, cam_, img0_gpu, img1_gpu,
                                      dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
}

TEST_F(TestSfmAligner, FullJacobianCompareWithCpu)
{
  // depth, valid and gradient buffers
  vc::Image2DManaged<float, vc::TargetHost>       dpt0(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> dpt0_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetHost>       vld0(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> vld0_gpu(width, height);
  vc::Image2DManaged<Grad, vc::TargetHost>        grad1(width, height);
  vc::Image2DManaged<Grad, vc::TargetDeviceCUDA>  grad1_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> std0_gpu(width, height);

  // gpu and cpu buffers for network output
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>       pyr_prx(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_prx_gpu(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>       pyr_jac(pyrlevels, width*codesize, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_jac_gpu(pyrlevels, width*codesize, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>       pyr_std(pyrlevels, width, height);

  // generate test camera poses: pose0, pose1
  Sophus::SE3f pose0;
  Eigen::Matrix<float,3,1> trs;
  Eigen::Matrix<float,3,1> rot;
  rot.setZero();
  trs.setZero();
  rot[1] = 0.1;
  rot[0] = 0.1;
  trs[0] = -0.5;
  trs[1] = -0.5;
  Sophus::SE3f pose(Sophus::SO3f::exp(rot), trs);
  Sophus::SE3f pose1 = pose.inverse();

  // decode zero code for frame 0
  Eigen::Matrix<float,CS,1> code;
  code.setZero();
  {
    cuda::ScopedContextPop pop;
    network_->Decode(*img0buf_, code, &pyr_prx, &pyr_std, &pyr_jac);
  }

  // copy network results to GPU
  pyr_prx_gpu.copyFrom(pyr_prx);
  pyr_jac_gpu.copyFrom(pyr_jac);
  std0_gpu.copyFrom(pyr_std[0]);

  // convert proximity to depth (done on GPU) and download to CPU
  df::UpdateDepth<float,CS,vc::Buffer2DView<float, vc::TargetDeviceCUDA>>(code, pyr_prx_gpu[0], pyr_jac_gpu[0], avg_dpt, dpt0_gpu);
  dpt0.copyFrom(dpt0_gpu);

  // calculate image gradient for frame 1
  df::SobelGradients(*img1_gpu_, grad1_gpu);
  grad1.copyFrom(grad1_gpu);

  ////////////////////////////////
  /// calculate the GPU result
  ////////////////////////////////
  // run alignment step
  auto gpu_result = aligner_->RunStep(pose0, pose1, code, cam_, *img0_gpu_, *img1_gpu_,
                                      dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);

  ////////////////////////////////
  /// calculate the CPU result
  ////////////////////////////////
  // calculate relative pose and its jacobian
  Eigen::Matrix<float,6,6> pose10_J_pose0;
  Eigen::Matrix<float,6,6> pose10_J_pose1;
  const Sophus::SE3f pose_10 = df::RelativePose(pose1, pose0, pose10_J_pose1, pose10_J_pose0);

  // calculate jacobian for each pixel
  ReductionItem cpu_result;
  vc::Image2DView<float, vc::TargetHost> std(pyr_std[0]);
  vc::Image2DView<float, vc::TargetHost> jac_img(pyr_jac[0]);
  vc::Image2DView<float, vc::TargetHost> prx_img(pyr_prx[0]);
  for (int x = 0; x < width; x += 1)
  {
    for (int y = 0; y < height; y += 1)
    {
      df::DenseSfm(x, y, pose_10, pose10_J_pose0, pose10_J_pose1,
                      code, cam_,*img0buf_, *img1buf_, dpt0, std,
                      vld0, jac_img, grad1, sfm_params, cpu_result);
    }
  }

  ////////////////////////////////
  /// compare GPU and CPU inliers and hessian
  ////////////////////////////////
  EXPECT_EQ(cpu_result.inliers, gpu_result.inliers);
  EXPECT_NE(cpu_result.inliers, 0);
  EXPECT_NE(gpu_result.inliers, 0);

  ReductionItem::HessianType::DenseMatrixType H_cpu = cpu_result.JtJ.toDenseMatrix();
  ReductionItem::HessianType::DenseMatrixType H_gpu = gpu_result.JtJ.toDenseMatrix();
  df::CompareWithTol(H_cpu, H_gpu, 1e-1);
}

TEST_F(TestSfmAligner, FullJacobianFiniteDiff)
{
  typedef Sophus::SE3<float> SE3T;
  typedef Eigen::Matrix<float,CS,1> CodeT;

  // depth, valid and gradient buffers
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> dpt0_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> vld0_gpu(width, height);
  vc::Image2DManaged<Grad, vc::TargetDeviceCUDA>  grad1_gpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> std0_gpu(width, height);

  // gpu and cpu buffers for network output
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>       pyr_prx(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_prx_gpu(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>       pyr_jac(pyrlevels, width*codesize, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_jac_gpu(pyrlevels, width*codesize, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>       pyr_std(pyrlevels, width, height);

  // generate poses for camera 0 and 1
  Sophus::SE3f pose0;
  Eigen::Matrix<float,3,1> trs;
  Eigen::Matrix<float,3,1> rot;
  rot.setZero();
  trs.setZero();
  rot[1] = 0.1;
  rot[0] = 0.1;
  trs[0] = -0.5;
  trs[1] = -0.5;
  Sophus::SE3f pose(Sophus::SO3f::exp(rot), trs);
  Sophus::SE3f pose1 = pose.inverse();

  // decode zero code for camera 0
  CodeT code;
  code.setZero();
  {
    cuda::ScopedContextPop pop;
    network_->Decode(*img0buf_, code, &pyr_prx, &pyr_std, &pyr_jac);
  }

  cv::imshow("cpu buffer", img0buf_->getOpenCV());

  // calculate spatial gradient of second view
  df::SobelGradients(*img1_gpu_, grad1_gpu);

  // copy network outputs to GPU
  pyr_prx_gpu.copyFrom(pyr_prx);
  pyr_jac_gpu.copyFrom(pyr_jac);
  std0_gpu.copyFrom(pyr_std[0]);

  // calculate depth from prx
  df::UpdateDepth<float, CS, vc::Buffer2DView<float,vc::TargetDeviceCUDA>>(code, pyr_prx_gpu[0], pyr_jac_gpu[0], avg_dpt, dpt0_gpu);

  vc::Image2DManaged<float, vc::TargetHost> warped_cpu(width, height);
  vc::Image2DManaged<float, vc::TargetDeviceCUDA> warped_gpu(width, height);
  se3aligner_->Warp(pose1, cam_, *img0_gpu_, *img1_gpu_, dpt0_gpu, warped_gpu);
  warped_cpu.copyFrom(warped_gpu);
  cv::imshow("orig", img0_);
  cv::imshow("warped", warped_cpu.getOpenCV());
  cv::waitKey(0);

  // run the aligner to get jacobians at linearisation point
  auto result_0 = aligner_->RunStep(pose0, pose1, code, cam_, *img0_gpu_, *img1_gpu_,
                                    dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);

  LOG(INFO) << "At linearisation point:";
  LOG(INFO) << "residual = " << result_0.residual;
  LOG(INFO) << "inliers = " << result_0.inliers;

  float eps = 1e-5;
  float tol_pose = 2e1;
  float tol_code = 1.5e-2;

  LOG(INFO);
  LOG(INFO) << "Jacobian w.r.t first pose";
  LOG(INFO) << "----------------------------";
  const std::vector<std::string> names = {"tx", "ty", "tz", "rotx", "roty", "rotz"};
  double total_time = 0;
  for (int i = 0; i < 6; ++i)
  {
    // perturb current test point
    SE3T pose0_forward = df::GetPerturbedPose(pose0, i, eps);

    // evaluate residual
    ReductionItem result;
    auto elapsed_ms = MeasureTime([&] () {
      result = aligner_->RunStep(pose0_forward, pose1, code, cam_, *img0_gpu_, *img1_gpu_,
                                 dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
    });
    if (i != 0) total_time += elapsed_ms;
    float findiff = 0.5 * (result.residual - result_0.residual) / eps;

    LOG(INFO) << "residual = " << result.residual;
    LOG(INFO) << "inliers = " << result.inliers;

    // print/check the results
    LOG(INFO);
    LOG(INFO) << names[i];
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Finite difference:"
              << std::setw(10) << std::left << findiff;
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Our derivative:"
              << std::setw(10) <<  std::left << result_0.Jtr(i,0);

    EXPECT_NEAR(findiff, result_0.Jtr(i,0), tol_pose);
  }
  double avg_time = total_time / 5;

  LOG(INFO);
  LOG(INFO) << "Jacobian w.r.t second pose";
  LOG(INFO) << "----------------------------";
  for (int i = 0; i < 6; ++i)
  {
    // perturb current test point
    SE3T pose1_forward = df::GetPerturbedPose(pose1, i, eps);

    // evaluate residual
    auto result = aligner_->RunStep(pose0, pose1_forward, code, cam_, *img0_gpu_, *img1_gpu_,
                                    dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
    float findiff = 0.5 * (result.residual - result_0.residual) / eps;

    // print/check the results
    LOG(INFO);
    LOG(INFO) << names[i];
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Finite difference:"
              << std::setw(10) << std::left << findiff;
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Our derivative:"
              << std::setw(10) <<  std::left << result_0.Jtr(SE3T::DoF + i, 0);
    EXPECT_NEAR(findiff, result_0.Jtr(SE3T::DoF + i,0), tol_pose);
  }

  LOG(INFO);
  LOG(INFO) << "Jacobian w.r.t code";
  LOG(INFO) << "----------------------------";
  float code_eps = 1e-3;
  for (int i = 0; i < codesize; ++i)
  {
    // perturb current test point
    CodeT code_forward(code);
    code_forward(i) += code_eps;

    // update depth
    df::UpdateDepth<float, CS, vc::Buffer2DView<float,vc::TargetDeviceCUDA>>(code_forward, pyr_prx_gpu[0], pyr_jac_gpu[0], avg_dpt, dpt0_gpu);

    // evaluate residual
    auto result = aligner_->RunStep(pose0, pose1, code_forward, cam_, *img0_gpu_, *img1_gpu_,
                                    dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
    float findiff = 0.5 * (result.residual - result_0.residual) / code_eps;

    // print/check the results
    LOG(INFO);
    LOG(INFO) << "entry " << i;
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Finite difference:"
              << std::setw(10) << std::left << findiff;
    LOG(INFO) << "\t" << std::setfill(' ') << std::setw(20) << std::left << "Our derivative:"
              << std::setw(10) <<  std::left << result_0.Jtr(2*SE3T::DoF + i, 0);
    EXPECT_NEAR(findiff, result_0.Jtr(2*SE3T::DoF + i,0), tol_code);
  }

  LOG(INFO) << "Average time for SfmAligner::RunStep: " << avg_time << " ms";
}
