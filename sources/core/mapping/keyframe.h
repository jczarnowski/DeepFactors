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
#ifndef DF_KEYFRAME_H_
#define DF_KEYFRAME_H_

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include "synced_pyramid.h"
#include "frame.h"

namespace df
{

template <typename Scalar>
class Keyframe : public df::Frame<Scalar>
{
public:
  typedef Keyframe<Scalar> This;
  typedef Frame<Scalar> Base;
  typedef typename Base::GradT GradT;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> CodeT;
  typedef df::SyncedBufferPyramid<Scalar> ImagePyramid;
  typedef vc::Buffer2DManaged<GradT, vc::TargetHost> CpuGradBuffer;
  typedef std::shared_ptr<Keyframe<Scalar>> Ptr;

  Keyframe() = delete;
  Keyframe(std::size_t pyrlevels, std::size_t w, std::size_t h, std::size_t cs)
      : Base(pyrlevels, w, h),
        pyr_dpt(pyrlevels, w, h),
        pyr_vld(pyrlevels, w, h),
        pyr_stdev(pyrlevels, w, h),
        pyr_prx_orig(pyrlevels, w, h),
        pyr_jac(pyrlevels, cs*w, h),
        dpt_grad(w,h)
  {
    code = CodeT::Zero(cs, 1);
  }

  Keyframe(const Keyframe& other)
    : Base(other),
      pyr_dpt(other.pyr_dpt),
      pyr_vld(other.pyr_vld),
      pyr_stdev(other.pyr_stdev),
      pyr_prx_orig(other.pyr_prx_orig),
      pyr_jac(other.pyr_jac),
      dpt_grad(other.pyr_img.Width(), other.pyr_img.Height())
  {
    code = other.code;
    dpt_grad.copyFrom(other.dpt_grad);
  }

  virtual ~Keyframe() {}

  virtual typename Base::Ptr Clone() override
  {
    return std::make_shared<This>(*this);
  }

  virtual std::string Name() override { return "kf" + std::to_string(this->id); }

//  virtual bool operator==(const Frame<Scalar>& other) override
//  {
//    return other.IsKeyframe() && (id == other.id);
//  }

  virtual bool IsKeyframe() override { return true; }

  // buffers that can exist on CPU or GPU
  ImagePyramid pyr_dpt;
  ImagePyramid pyr_vld;
  ImagePyramid pyr_stdev;
  ImagePyramid pyr_prx_orig;
  ImagePyramid pyr_jac;

  // data only on CPU
  CpuGradBuffer dpt_grad;
  CodeT     code;
};

} // namespace df

#endif // DF_KEYFRAME_H_
