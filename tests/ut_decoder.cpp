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

#include "testing_utils.h"
#include "display_utils.h"
#include "pinhole_camera.h"
#include "cuda_context.h"
#include "warping.h"
#include "decoder_network.h"

class TestDecoder : public ::testing::Test
{
public:
  static constexpr std::size_t CS = 32;
  static constexpr const char* TEST_IMG_PATH = "data/testimg/0.jpg";
  static constexpr const char* TEST_CFG_PATH = "data/nets/scannet256_32.cfg";
  typedef vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> CpuPyramid;
  typedef Eigen::Matrix<float, CS, 1> Code;

  TestDecoder()
  {
    CreateAndInitNetwork(TEST_CFG_PATH);
    LoadPreprocessData();
  }

  void CreateAndInitNetwork(const std::string& cfgpath)
  {
    netcfg_ = df::LoadJsonNetworkConfig(cfgpath);
    network_ = std::make_shared<df::DecoderNetwork>(netcfg_);

    width = netcfg_.input_width;
    height = netcfg_.input_height;
    codesize = netcfg_.code_size;
    pyrlevels = netcfg_.pyramid_levels;
  }

  /*
   * Loads and preprocesses a SceneNet image
   * Corrects focal length to the network focal length
   */
  void LoadPreprocessData()
  {
    auto cam = netcfg_.camera;
    df::PinholeCamera<float> in_cam = df::GetSceneNetCam<float>(width, height); // any size cam works because it's resized in LoadPreprocessImage
    df::PinholeCamera<float> net_cam(cam.fx, cam.fy, cam.u0, cam.v0, width, height);
    img0_ = df::LoadPreprocessImage(TEST_IMG_PATH, in_cam, net_cam);
  }

  df::DecoderNetwork::NetworkConfig netcfg_;
  std::shared_ptr<df::DecoderNetwork> network_;
  cv::Mat img0_;

  int width;
  int height;
  int codesize;
  int pyrlevels;
};

TEST_F(TestDecoder, DecodeZero)
{
  // create buffers to pass to DecoderNetwork::Decode
  vc::Image2DView<float, vc::TargetHost> imgbuf((float*)img0_.data, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> pyr_dpt(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> pyr_jac(pyrlevels, width * codesize, height);

  // get a zero code
  Eigen::Matrix<float,CS,1> code;
  code.setZero();

  // decode without jacobian
  for (int i = 0; i < 5; ++i)
  {
    auto start = std::clock();
    network_->Decode(imgbuf, code, &pyr_dpt);
    LOG(INFO) << "[without jacobian] time: " << (std::clock()-start) / (double)( CLOCKS_PER_SEC / 1000);
  }

  // decode with jacobian
  for (int i = 0; i < 5; ++i)
  {
    auto start = std::clock();
    network_->Decode(imgbuf, code, &pyr_dpt, nullptr, &pyr_jac);
    LOG(INFO) << "[with jacobian] time: " << (std::clock()-start) / (double)( CLOCKS_PER_SEC / 1000);
  }

  // display results
  for (uint i = 0; i < netcfg_.pyramid_levels; ++i)
  {
    auto dptbuf = pyr_dpt[i];
    cv::Mat retimg(dptbuf.height(), dptbuf.width(), CV_32FC1, dptbuf.rawPtr());
    cv::Mat depth_cmapped;
    retimg.convertTo(depth_cmapped, CV_8UC1, 255.0);
    cv::applyColorMap(depth_cmapped, depth_cmapped, cv::COLORMAP_JET);
    cv::imshow("[TestDecoder.DecodeZero] depth_est " + std::to_string(i), depth_cmapped);
  }
  cv::waitKey(0);
}

TEST_F(TestDecoder, ExploreCode)
{
  Eigen::VectorXf code(codesize);
  code.setZero();

  // decode zero code
  vc::Image2DView<float, vc::TargetHost> img0buf(img0_);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> pyr_prx(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> pyr_jac(pyrlevels, width*codesize, height);
  {
    cuda::ScopedContextPop pop;
    network_->Decode(img0buf, code, &pyr_prx, nullptr, &pyr_jac);
  }

  // display zero code depth
  cv::imshow("[TestDecoder.ExploreCode] prx_0code", prx_to_color(pyr_prx[0]));

  // perturb each element of the code and render depth
  float eps = 15;
  std::vector<cv::Mat> array;
  for (int i = 0; i < codesize; ++i)
  {
    Eigen::VectorXf code_front(code);
    code_front[i] += eps;

    auto dpt_code1 = df::RenderDpt(width, height, (float)netcfg_.avg_dpt, code_front,
                                      pyr_jac[0], pyr_prx[0]);

    auto prx = df::DepthToProx(dpt_code1.getOpenCV(), netcfg_.avg_dpt);
    array.push_back(apply_colormap(prx));
  }
  cv::imshow("[TestDecoder.ExploreCode] prx_jacobian", CreateMosaic(array));
  cv::waitKey(0);
}

/**
 * jacobian test
 * get the jacobian for the image
 * decode the image with initial code c0 into D0
 * decode the image with code c1 = c0 + ei * eps into D1
 * compute D1' using the linear jacobian J, D1' = D0 + J * ei * eps
 * compare D1 with D1'
 */
TEST_F(TestDecoder, Jacobian)
{
  // image to cpu and gpu buffers
  vc::Image2DView<float, vc::TargetHost> img((float*)img0_.data, width, height);

  // buffers for results
  CpuPyramid prx0(pyrlevels, width, height);
  CpuPyramid prx1(pyrlevels, width, height);
  vc::RuntimeBufferPyramidManaged<float, vc::TargetHost> jac(pyrlevels, width * codesize, height);

  // decode the image with initial code c0 into D0
  Code cde0 = Code::Constant(0);
  network_->Decode(img, cde0, &prx0, nullptr, &jac);

  std::vector<cv::Mat> array_prx1;
  std::vector<cv::Mat> array_prx1_p;

  // decode the image with code c1 = c0 + ei * eps into D1
  float eps = 15;
  for (int i = 0; i < codesize; ++i)
  {
    Code cde1(cde0);
    cde1(i) += eps;

    network_->Decode(img, cde1, &prx1);
    array_prx1_p.push_back(prx_to_color(prx1[0]));

    // compute D1' using the linear jacobian J, D1' = D0 + J * ei * eps
    vc::Image2DManaged<float, vc::TargetHost> prx1_p(width, height);
    double prx_tol = 1e-5;
    for (uint u = 0; u < netcfg_.input_width; ++u)
    {
      for (uint v = 0; v < netcfg_.input_height; ++v)
      {
        Eigen::Map<Eigen::MatrixXf> pixjac(&jac[0](u*codesize, v), 1, codesize);
        prx1_p(u,v) = df::ProxFromCode(cde1, pixjac, prx0[0](u,v));
        EXPECT_NEAR(prx1_p(u,v), prx1[0](u,v), prx_tol) << "Jacobian propagated depth does not match evaluated depth";
      }
    }
    array_prx1.push_back(prx_to_color(prx1_p));
  }

  // display results
  std::string title = "[TestDecoder.Jacobian] img / prx_0code / prx_forward / prx_with_jac";
  std::vector<cv::Mat> array = {img0_, prx_to_color(prx0[0]), CreateMosaic(array_prx1_p), CreateMosaic(array_prx1)};
  cv::Mat mosaic = CreateMosaic(array, 1,  array.size());
  cv::namedWindow(title, cv::WINDOW_NORMAL);
  cv::imshow(title, mosaic);
  cv::waitKey(0);
}
