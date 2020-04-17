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
#ifndef DF_KEYFRAME_RENDERER_H_
#define DF_KEYFRAME_RENDERER_H_

#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>
#include <VisionCore/Buffers/Image2D.hpp>

#include "pinhole_camera.h"

namespace df
{

class KeyframeRenderer
{
public:
  struct DisplayData
  {
    typedef vc::Image2DManaged<float, vc::TargetHost> CpuBuffer;

    DisplayData(int width, int height)
        : dpt(width,height),
          dpt_orig(width,height),
          vld(width,height),
          std(width,height) {}

    cv::Mat color_img;
    CpuBuffer dpt;
    CpuBuffer dpt_orig;
    CpuBuffer vld;
    CpuBuffer std;
    Sophus::SE3f pose_wk;
  };

  void Init(const df::PinholeCamera<float>& cam);
  void RenderKeyframe(const pangolin::OpenGlMatrix& vp, const DisplayData& data);

  void SetPhong(bool enabled);
  void SetDrawNoisyPixels(bool enabled);
  void SetLightPos(float x, float y, float z);
  void SetStdevThresh(float thresh);
  void SetSltThresh(float thresh);
  void SetCropPix(int pix);

private:
  std::size_t width_;
  std::size_t height_;
  df::PinholeCamera<float> cam_;

  // options
  bool phong_enabled_;
  bool draw_noisy_pixels_;
  float3 light_pos_;
  float stdev_thresh_;
  float slt_thresh_;
  int crop_pix_;

  pangolin::GlSlProgram shader_;
  pangolin::GlTexture col_tex_;
  pangolin::GlTexture dpt_tex_;
  pangolin::GlTexture val_tex_;
  pangolin::GlTexture std_tex_;
};

} // namespace df

#endif // DF_KEYFRAME_RENDERER_H_
