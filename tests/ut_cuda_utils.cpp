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
#include <VisionCore/Buffers/Image2D.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "cu_image_proc.h"

class CudaUtilsTest : public ::testing::Test
{
public:
  typedef vc::Image2DManaged<float, vc::TargetDeviceCUDA> GPUImage;
  typedef vc::Image2DManaged<float, vc::TargetHost> CPUImage;
  
  CudaUtilsTest()
  {
    img0_ = LoadTestImage("data/testimg/1047.jpg");
    
    imwidth_ = img0_.cols;
    imheight_ = img0_.rows;

    // initialize buffers
    input_buf_ = std::make_unique<GPUImage>(imwidth_, imheight_);
    downsampled_gpu_ = std::make_unique<GPUImage>(imwidth_/2, imheight_/2);
    downsampled_cpu_ = std::make_unique<CPUImage>(imwidth_/2, imheight_/2);

    input_buf_->copyFrom(img0_);
  }

  cv::Mat LoadTestImage(std::string path)
  {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
      std::cout << "Failed loading test image" << std::endl;
      exit(1);
    }

    img.convertTo(img, CV_32FC1, 1 / 255.0);
    return img;
  }

  std::unique_ptr<GPUImage> input_buf_;
  std::unique_ptr<GPUImage> downsampled_gpu_;
  std::unique_ptr<CPUImage> downsampled_cpu_;
  
  cv::Mat img0_;
  std::size_t imwidth_;
  std::size_t imheight_;
  
  static constexpr float blur_eps_ = 1e-1;
  static constexpr float sobel_eps_ = 1e-4;
  static constexpr std::size_t num_runs_ = 10;
};

TEST_F(CudaUtilsTest, Downsample)
{
  float average_ms = 0;
  for (std::size_t i = 0; i < num_runs_+1; ++i)
  {
    auto start = std::clock();
    df::GaussianBlurDown(*input_buf_, *downsampled_gpu_);
    auto ms_elapsed = (std::clock()-start) / (double)(CLOCKS_PER_SEC / 1000);

    if (i != 0) // skip first run as cuda is warming up
      average_ms += ms_elapsed;
  }

  std::cout << "Average time for GaussianBlurDown: " << average_ms / num_runs_ << " ms" << std::endl;
  
  downsampled_cpu_->copyFrom(*downsampled_gpu_);
  
  cv::Mat blur_ocv, downsample_ocv;
  cv::GaussianBlur(img0_, blur_ocv, {5,5}, 0, 0);
  cv::pyrDown(blur_ocv, downsample_ocv, {(int)imwidth_/2, (int)imheight_/2});
  
  for (int px = 1; px < downsample_ocv.cols-1; ++px)
  {
    for (int py = 1; py < downsample_ocv.rows-1; ++py)
    {
      float ocv_pix = downsample_ocv.at<float>(py, px);
      float my_pix = (*downsampled_cpu_)(px, py);
      
      ASSERT_NEAR(ocv_pix, my_pix, blur_eps_);
    }
  }

//   cv::imshow("original", img0_);
//   cv::imshow("downsampled", downsampled_cpu_->getOpenCV());
//   cv::waitKey(0);
}

TEST_F(CudaUtilsTest, SobelGradients)
{
  vc::Image2DManaged<Eigen::Matrix<float,1,2>, vc::TargetDeviceCUDA> sobel_gpu(imwidth_, imheight_);
  vc::Image2DManaged<Eigen::Matrix<float,1,2>, vc::TargetHost> sobel_cpu(imwidth_, imheight_);
  
  float average_ms = 0;
  for (std::size_t i = 0; i < num_runs_+1; ++i)
  {
    auto start = std::clock();
    df::SobelGradients(*input_buf_, sobel_gpu);
    auto ms_elapsed = (std::clock()-start) / (double)(CLOCKS_PER_SEC / 1000);

    if (i != 0) // skip first run as cuda is warming up
      average_ms += ms_elapsed;
  }
  std::cout << "Average time for SobelGradients: " << average_ms / num_runs_ << " ms" << std::endl;
  
  sobel_cpu.copyFrom(sobel_gpu);
  
  cv::Mat ocv_gradx, ocv_grady;
  cv::Sobel(img0_, ocv_gradx, CV_32F, 1, 0, 3, 1/8.0);
  cv::Sobel(img0_, ocv_grady, CV_32F, 0, 1, 3, 1/8.0);
  
  for (std::size_t px = 1; px < imwidth_-1; ++px)
  {
    for (std::size_t py = 1; py < imheight_-1; ++py)
    {
      Eigen::Matrix<float,1,2> grad = sobel_cpu(px, py);
      auto ocv_gx = ocv_gradx.at<float>(py,px);
      auto ocv_gy = ocv_grady.at<float>(py,px);
      ASSERT_NEAR(grad[0], ocv_gx, sobel_eps_);
      ASSERT_NEAR(grad[1], ocv_gy, sobel_eps_);
    }
  }
}
