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
#include <glog/logging.h>
#include <VisionCore/Buffers/Buffer1D.hpp>

#include "keyframe_renderer.h"

namespace df
{

void KeyframeRenderer::Init(const df::PinholeCamera<float>& cam)
{
  cam_ = cam;
  width_ = cam.width();
  height_ = cam.height();

  // Load and compile shader
  std::string shader_dir(DF_SHADER_DIR);
  shader_.AddShaderFromFile(pangolin::GlSlVertexShader,   shader_dir + "/empty.vert");
  shader_.AddShaderFromFile(pangolin::GlSlGeometryShader, shader_dir + "/drawkf.geom");
  shader_.AddShaderFromFile(pangolin::GlSlFragmentShader, shader_dir + "/phong.frag");
  if (!shader_.Link())
    LOG(FATAL) << "Failed to compile shader";

  // initialize textures
  col_tex_.Reinitialise(width_, height_, GL_RGBA8, true, 0, GL_BGR, GL_UNSIGNED_BYTE);
  dpt_tex_.Reinitialise(width_, height_, GL_R32F, true, 0, GL_RED, GL_FLOAT);
  val_tex_.Reinitialise(width_, height_, GL_R32F, true, 0, GL_RED, GL_FLOAT);
  std_tex_.Reinitialise(width_, height_, GL_R32F, true, 0, GL_RED, GL_FLOAT);
}

void KeyframeRenderer::RenderKeyframe(const pangolin::OpenGlMatrix& vp, const DisplayData &data)
{
  // upload data from keyframe to buffers
  // TODO: check if a keyframe is currently on GPU, if yes then use that!
  col_tex_.Upload(data.color_img.data, GL_BGR, GL_UNSIGNED_BYTE);
  dpt_tex_.Upload(data.dpt.rawPtr(), GL_RED, GL_FLOAT);
  val_tex_.Upload(data.vld.rawPtr(), GL_RED, GL_FLOAT);
  std_tex_.Upload(data.std.rawPtr(), GL_RED, GL_FLOAT);

  // activate program
  shader_.Bind();

  pangolin::OpenGlMatrix mvp = vp * pangolin::OpenGlMatrix(data.pose_wk.matrix());

  // fill uniforms
  shader_.SetUniform("mvp", mvp);
  shader_.SetUniform("cam", cam_.fx(), cam_.fy(), cam_.u0(), cam_.v0());
  shader_.SetUniform("lightpos", light_pos_.x, light_pos_.y, light_pos_.z);
  shader_.SetUniform("width", (int) width_);
  shader_.SetUniform("height", (int) height_);
  shader_.SetUniform("phong_enabled", phong_enabled_);
  shader_.SetUniform("draw_noisy_pixels", draw_noisy_pixels_);
  shader_.SetUniform("stdev_thresh", stdev_thresh_);
  shader_.SetUniform("slt_thresh", slt_thresh_);
  shader_.SetUniform("crop_pix", crop_pix_);

  // texture bank ids
  shader_.SetUniform("image", 0);
  shader_.SetUniform("depth", 1);
  shader_.SetUniform("valid", 2);

  // setup texture banks
  glActiveTexture(GL_TEXTURE0);
  col_tex_.Bind();
  glActiveTexture(GL_TEXTURE1);
  dpt_tex_.Bind();
  glActiveTexture(GL_TEXTURE2);
  val_tex_.Bind();

  // draw keyframe
  glDrawArrays(GL_POINTS, 0, width_ * height_);

  // TODO: go back to previous active texture
  glActiveTexture(GL_TEXTURE0);

  // deactivate shader
  shader_.Unbind();
}

void KeyframeRenderer::SetPhong(bool enabled)
{
  phong_enabled_ = enabled;
}

void KeyframeRenderer::SetDrawNoisyPixels(bool enabled)
{
  draw_noisy_pixels_ = enabled;
}

void KeyframeRenderer::SetLightPos(float x, float y, float z)
{
  light_pos_ = make_float3(x,y,z);
}

void KeyframeRenderer::SetStdevThresh(float thresh)
{
  stdev_thresh_ = thresh;
}

void KeyframeRenderer::SetSltThresh(float thresh)
{
  slt_thresh_ = thresh;
}

void KeyframeRenderer::SetCropPix(int pix)
{
  crop_pix_ = pix;
}

} // namespace df

