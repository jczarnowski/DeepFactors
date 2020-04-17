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
#ifndef DF_FRAME_H_
#define DF_FRAME_H_

#include <memory>

#include <DBoW2.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include "synced_pyramid.h"
#include "cu_image_proc.h"
#include "feature_detection.h"

namespace df
{

template <typename Scalar>
class Frame
{
public:
  typedef Frame<Scalar> This;
  typedef std::shared_ptr<This> Ptr;
  typedef Sophus::SE3<Scalar> SE3T;
  typedef Eigen::Matrix<Scalar,1,2> GradT;
  typedef df::SyncedBufferPyramid<Scalar> ImagePyramid;
  typedef df::SyncedBufferPyramid<GradT> GradPyramid;
  typedef std::size_t IdType;

  Frame() = delete;
  Frame(std::size_t pyrlevels, std::size_t w, std::size_t h)
      : pyr_img(pyrlevels, w, h),
        pyr_grad(pyrlevels, w, h),
        id(0),
        width(w),
        height(h) {}

  Frame(const Frame& other)
    : pyr_img(other.pyr_img),
      pyr_grad(other.pyr_grad),
      pose_wk(other.pose_wk),
      id(other.id),
      features(other.features),
      timestamp(other.timestamp),
      bow_vec(other.bow_vec),
      has_keypoints(other.has_keypoints),
      marginalized(other.marginalized)
  {
    color_img = other.color_img.clone();
  }

  virtual ~Frame() {}

  virtual Ptr Clone()
  {
    return std::make_shared<This>(*this);
  }

  virtual std::string Name() { return "fr" + std::to_string(id); }

  virtual bool IsKeyframe() { return false; }

  void FillPyramids(const cv::Mat& img, uint pyrlevels)
  {
    for(uint i = 0; i < pyrlevels; ++i)
    {
      if (i == 0)
      {
        vc::Image2DView<float, vc::TargetHost> tmp1(img);
        pyr_img.GetGpuLevel(0).copyFrom(tmp1);
        df::SobelGradients(pyr_img.GetGpuLevel(0), pyr_grad.GetGpuLevel(0));
        continue;
      }
      df::GaussianBlurDown(pyr_img.GetGpuLevel(i-1), pyr_img.GetGpuLevel(i));
      df::SobelGradients(pyr_img.GetGpuLevel(i), pyr_grad.GetGpuLevel(i));
    }
  }

  // buffers that can exist on CPU or GPU
  ImagePyramid pyr_img;
  GradPyramid pyr_grad;

  // data only on CPU
  SE3T      pose_wk;
  cv::Mat   color_img;    // original color image for display
  IdType    id;

  // size
  std::size_t width;
  std::size_t height;

  // sparse features
  Features features;

  // time of rgb image acquisition
  double timestamp;

  // bag of words representation
  DBoW2::BowVector bow_vec;

  bool has_keypoints = false;
  bool marginalized = false;
};

} // namespace df

#endif // DF_FRAME_H_
