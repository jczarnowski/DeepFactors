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
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include "display_utils.h"
#include "visualizer.h"
#include "timing.h"
#include "interp.h"

namespace df
{

VisualizerConfig::VisualizeMode
VisualizerConfig::VisualizeModeTranslator(const std::string& s)
{
  if (s == "ALL")
    return VisualizeMode::ALL;
  else if (s == "FIRST")
    return VisualizeMode::FIRST;
  else if (s == "LAST")
    return VisualizeMode::LAST;
  else if (s == "LAST_N")
    return VisualizeMode::LAST_N;
  else
    LOG(FATAL) << "Unknown visualization mode: " << s;
  return VisualizeMode::ALL;
}

void OglDebugCallback(GLenum source,
                      GLenum type,
                      GLuint id,
                      GLenum severity,
                      GLsizei length,
                      const GLchar* message,
                      const void* userParam)
{
  LOG(INFO) << "OpenGL Debug message: " << message;
  if (type == GL_DEBUG_TYPE_ERROR)
    LOG(FATAL) << "OpenGL error: " << message;
}

std::map<int, GuiEvent> default_keymap =
  {{'r', RESET}, {' ', INIT}, {'p', PAUSE}, {'n', NEW_KF}};

/* ************************************************************************* */
Visualizer::Visualizer() : Visualizer(VisualizerConfig()) {}

/* ************************************************************************* */
Visualizer::Visualizer(const VisualizerConfig& cfg)
    : cfg_(cfg),
      map_viewport_(nullptr),
      frame_viewport_(nullptr),
      s_cam(nullptr),
      image_tex(nullptr),
      new_map_(true)
{
  camera_rot_vel_.setZero();
  camera_vel_.setZero();
}

/* ************************************************************************* */
Visualizer::~Visualizer()
{
  delete image_tex;
  delete s_cam;
  delete frame_viewport_;
  delete map_viewport_;
  delete reset_;
  // NO TIME FOR THIS NOW!!!
}

/* ************************************************************************* */
void Visualizer::Init(const df::PinholeCamera<float>& cam, df::DeepFactorsOptions df_opts)
{
  cam_ = cam;
  df_opts_ = df_opts;

  pangolin::CreateWindowAndBind(cfg_.win_title, cfg_.win_width, cfg_.win_height);

  SetKeymap(default_keymap);

  InitOpenGL();
  BuildGui();

  kfrenderer_.Init(cam_);

  residual_img_.create(cam_.height(), cam_.width(), CV_32FC1);

  // here synchronize current GUI options with settings
  SyncOptions();

  // initialize camera position
  Eigen::Vector3f pos(0,0,-1);
  Eigen::Vector3f rot(0,0,0);

  // 0084_00
//  Eigen::Vector3f pos(1.5277, -0.924338, -0.996986);
//  Eigen::Vector3f rot(-0.36894, -0.738385, -0.503094);

  // desk_validation
//  Eigen::Vector3f pos(-0.514423,  0.590621, -0.807093);
//  Eigen::Vector3f rot(0.426826, 0.0828716,  0.237679);


  Sophus::SE3f cam_init;
  cam_init.translation() = pos;
  cam_init.so3() = Sophus::SO3f::exp(rot);
  s_cam->SetModelViewMatrix(cam_init.inverse().matrix());

  initialized_ = true;
}

/* ************************************************************************* */
void Visualizer::SyncOptions()
{
  kfrenderer_.SetPhong(phong_->Get());
  kfrenderer_.SetDrawNoisyPixels(draw_noisy_pix_->Get());
  kfrenderer_.SetStdevThresh(stdev_thresh_->Get());
  kfrenderer_.SetSltThresh(slt_thresh_->Get());
  kfrenderer_.SetLightPos(light_pos_x_->Get(), light_pos_y_->Get(), light_pos_z_->Get());
  kfrenderer_.SetCropPix(crop_pix_->Get());
  cfg_.vis_mode = (VisualizerConfig::VisualizeMode)vis_mode_->Get();

  df_opts_.tracking_huber_delta = tracking_huber_delta_->Get();
  df_opts_.huber_delta = huber_delta_->Get();
  df_opts_.inlier_threshold = inlier_thresh_->Get();
  df_opts_.dist_threshold = dist_thresh_->Get();
  df_opts_.tracking_error_threshold = error_thresh_->Get();
  df_opts_.tracking_dist_threshold = tracking_dist_threshold_->Get();
}

/* ************************************************************************* */
bool Visualizer::ShouldQuit()
{
  return pangolin::ShouldQuit();
}

/* ************************************************************************* */
void Visualizer::HandleEvents()
{
  if (pangolin::Pushed(*reset_))
    EmitEvent(RESET);

  if (ShouldQuit())
    EmitEvent(EXIT);

  if(pangolin::Pushed(*record_))
//      pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=8388608,unique_filename]//screencap.avi");
    map_viewport_->RecordOnRender("ffmpeg:[fps=30,bps=8388608,unique_filename]//screencap.avi");

  if (pangolin::Pushed(*save_))
    map_viewport_->SaveOnRender("reconstruction");

  if (pangolin::GuiVarHasChanged())
  {
    SyncOptions();
    kf_plotter_->ClearMarkers();
    error_plotter_->ClearMarkers();
    kf_plotter_->AddMarker(pangolin::Marker::Horizontal, df_opts_.inlier_threshold, pangolin::Marker::LessThan, pangolin::Colour::Red().WithAlpha(0.2f));
    kf_plotter_->AddMarker(pangolin::Marker::Horizontal, df_opts_.dist_threshold, pangolin::Marker::GreaterThan, pangolin::Colour::Green().WithAlpha(0.2f));
    error_plotter_->AddMarker(pangolin::Marker::Horizontal, df_opts_.tracking_error_threshold, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f));
    EmitEvent(PARAM_CHANGE);
  }
}

/* ************************************************************************* */
void Visualizer::Draw(float delta_time)
{
  // Clear entire screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Activate 3d window
  map_viewport_->Activate(*s_cam);

  // bilateral filter
  tic("[visualizer] bilateral filter");
  if (new_map_)
  {
    std::lock_guard<std::mutex> guard(map_mutex_);
    for (auto& kv : display_data_)
    {
      if (kv.second.active)
      {
        cv::Mat dpt = kv.second.data.dpt_orig.getOpenCV();
        cv::Mat dpt_filtered = kv.second.data.dpt.getOpenCV();
        cv::bilateralFilter(dpt, dpt_filtered, 5, dpt_filter_sigma_->Get(), dpt_filter_sigma_->Get());
      }
    }
    new_map_ = false;
  }
  toc("[visualizer] bilateral filter");

  // Draw camera frustrum in the current pose
  if (draw_traj_->Get())
  {
    std::lock_guard<std::mutex> guard(campose_mutex_);
    glColor4f(cfg_.cam_color[0], cfg_.cam_color[1], cfg_.cam_color[2], cfg_.cam_color[3]);
    pangolin::glDrawFrustum<double>(cam_.InverseMatrix<double>(),
                                     cam_.width(), cam_.height(),
                                     cam_pose_wc_.matrix().cast<double>(), 0.1f);
  }

  // Set the opengl camera position
//  Eigen::Matrix4f mv = Eigen::Matrix4f(s_cam->GetModelViewMatrix()).inverse();
//  Eigen::Quaternionf curr_rot(mv.topLeftCorner<3,3>());
//  Eigen::Vector3f curr_pos = mv.topRightCorner<3,1>();
//  LOG(INFO) << "curr_pos: " << curr_pos.transpose();
//  LOG(INFO) << "curr_rot: " << Sophus::SO3f{curr_rot}.log().transpose();

  if (!freelook_->Get())
  {
    std::lock_guard<std::mutex> guard(map_mutex_);

    // calculate current target pose
    Eigen::VectorXf off(6);
    off << 0,0,-1,0,0,0;
    Sophus::SE3f target = follow_pose_ * Sophus::SE3f::exp(off);

    // calculate smoothed camera pose
    Eigen::Matrix4f mv = Eigen::Matrix4f(s_cam->GetModelViewMatrix()).inverse();
    Eigen::Vector3f curr_pos = mv.topRightCorner<3,1>();
    Eigen::Quaternionf curr_rot(mv.topLeftCorner<3,3>());
    curr_pos = SmoothDamp(curr_pos, target.translation(), camera_vel_, delta_time / 1000.0f, trs_damping_->Get(), 10000000.f);
    curr_rot = QuatDamp(curr_rot, target.so3().unit_quaternion(), camera_rot_vel_, delta_time / 1000.f, rot_damping_->Get(), 1e-4);

    // set opengl cam
    Sophus::SE3f current_pose(curr_rot, curr_pos);
    s_cam->SetModelViewMatrix(current_pose.inverse().matrix());
  }
  else
  {
    // zero out velocities and just let the user control it
    camera_vel_.setZero();
    camera_rot_vel_.setZero();
  }

  // render all keyframes
  RenderKeyframes(delta_time);

  // upload live frame to texture
  if (!live_frame_.empty())
  {
    std::lock_guard<std::mutex> guard(frame_mutex_);
    if (!image_tex)
      image_tex = new pangolin::GlTexture(live_frame_.cols, live_frame_.rows,
                                          GL_RGBA8, false, 0, GL_BGR,
                                          GL_UNSIGNED_BYTE);
    image_tex->Upload(live_frame_.data, GL_BGR, GL_UNSIGNED_BYTE);
  }

  // upload residual image
  if (!residual_img_.empty())
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (!residual_tex)
      residual_tex = new pangolin::GlTexture(residual_img_.cols, residual_img_.rows,
                                             GL_RGB, false, 0, GL_LUMINANCE, GL_FLOAT);
    residual_tex->Upload(residual_img_.data, GL_LUMINANCE, GL_FLOAT);
  }

  // Activate live frame window
  if (image_tex && !cfg_.demo_mode)
  {
    frame_viewport_->Activate();
    glColor3f(1.0,1.0,1.0);
    image_tex->RenderToViewportFlipY();
  }

  if (residual_tex && !cfg_.demo_mode)
  {
    residual_viewport_->Activate();
    glColor3f(1.0,1.0,1.0);
    residual_tex->RenderToViewportFlipY();
  }

  // Swap frames and Process Events
//  std::lock_guard<std::mutex> lock(stats_mutex_); // for pulling data from logs (I assume this is when it happens)
  pangolin::FinishFrame();
}

/* ************************************************************************* */
void Visualizer::OnNewFrame(const cv::Mat& frame)
{
  if (!initialized_)
    return;

  std::lock_guard<std::mutex> guard(frame_mutex_);
  assert(frame.type() == CV_8UC3);
  cv::resize(frame, live_frame_, {(int)cam_.width(), (int)cam_.height()});
}

/* ************************************************************************* */
void Visualizer::OnNewCameraPose(const Sophus::SE3f& pose_wc)
{
  std::lock_guard<std::mutex> guard(campose_mutex_);
  cam_pose_wc_ = pose_wc;
}

/* ************************************************************************* */
void Visualizer::OnNewMap(MapT::Ptr map)
{
  if (!initialized_)
    return;

  tic("[visualizer] copying new map");
  std::lock_guard<std::mutex> guard(map_mutex_);
  keyframe_links_ = map->keyframes.GetLinks();

  // copy frame -> keyframe links
  frame_links_.clear();
  for (auto& link : map->frames.GetLinks())
      frame_links_.push_back(link);

  // get the frame poses for display
  frame_poses_.clear();
  for (auto& id : map->frames.Ids())
  {
    frame_poses_[id] = map->frames.Get(id)->pose_wk;
    frame_mrg_[id] = map->frames.Get(id)->marginalized;
  }


  // mark all cache items as inactive
  for (auto& kv : display_data_)
    kv.second.active = false;

  last_id_ = map->keyframes.LastId();

  // copy all keyframe poses
  keyframe_poses_.clear();
  for (auto& id : map->keyframes.Ids())
    keyframe_poses_[id] = map->keyframes.Get(id)->pose_wk;

  // Copy only last N frames
  //  for (auto& id : map->keyframes.Ids())
  int max = static_cast<int>(last_id_) - num_visible_keyframes_;
  for (int id = last_id_; id > 0 && id > max; --id)
  {
    // check if kf already has an KeyframeDisplayData object
    if (display_data_.find(id) == display_data_.end())
      display_data_.emplace(id, DisplayCacheItem{(int)cam_.width(), (int)cam_.height()});

    // fill the data using keyframe
    auto kf = map->keyframes.Get(id);
    auto& item = display_data_.at(id);
    auto& data = item.data;
    data.color_img = kf->color_img;
    data.pose_wk = kf->pose_wk;
    data.dpt_orig.copyFrom(kf->pyr_dpt.GetCpuLevel(0));
    data.std.copyFrom(kf->pyr_stdev.GetCpuLevel(0));
    data.vld.copyFrom(kf->pyr_vld.GetCpuLevel(0));
    item.active = true;
  }

  if (last_id_ > 0)
    follow_pose_ = map->keyframes.Get(last_id_)->pose_wk;
  new_map_ = true;

  toc("[visualizer] copying new map");
}

/* ************************************************************************* */
void Visualizer::OnNewStats(const DisplayStats& stats)
{
  std::lock_guard<std::mutex> lock(stats_mutex_);
  keyframe_log_.Log(stats.inliers, stats.distance);
  error_log_.Log(stats.tracker_error);
  residual_img_ = stats.residual_img;
  relin_info_ = stats.relin_info;
}

/* ************************************************************************* */
void Visualizer::SetEventCallback(EventCallback cb)
{
  event_callback_ = cb;
}

/* ************************************************************************* */
void Visualizer::SetKeymap(KeyConfig keycfg)
{
  keymap_ = keycfg;

  // register callback for required keys
  for (auto& map : keymap_)
    pangolin::RegisterKeyPressCallback(map.first, std::bind(&Visualizer::KeyCallback, this, std::placeholders::_1));
}

/* ************************************************************************* */
void Visualizer::KeyCallback(int key)
{
  if (keymap_.find(key) != keymap_.end())
    EmitEvent(keymap_[key]);
}

/* ************************************************************************* */
void Visualizer::RenderKeyframes(float delta_time)
{
  std::lock_guard<std::mutex> guard(map_mutex_);

  pangolin::OpenGlMatrix vp = s_cam->GetProjectionModelViewMatrix();
  int last_drawn_depth_id = static_cast<int>(last_id_) - num_visible_keyframes_;
  int last_drawn_frustum_id = static_cast<int>(last_id_) - max_visible_frustums_;

  std::map<KeyframeT::IdType, bool> relin_info;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    relin_info = std::map<KeyframeT::IdType, bool>(relin_info_);
  }

  tic("[visualizer] rendering depthmaps");
  for (auto& kv : display_data_)
  {
    if (!kv.second.active)
      continue;

    const auto& id = kv.first;
    auto& data = kv.second.data;

    if ((cfg_.vis_mode == VisualizerConfig::ALL) ||
        (cfg_.vis_mode == VisualizerConfig::LAST && id == last_id_) ||
        (cfg_.vis_mode == VisualizerConfig::FIRST && id == 1) ||
        (cfg_.vis_mode == VisualizerConfig::LAST_N && (id > last_id_-5 || last_id_ < 5)))
      kfrenderer_.RenderKeyframe(vp, data);
  }
  toc("[visualizer] rendering depthmaps");

  if (!draw_traj_->Get())
    return;

  // render keyframe frustums
  for (auto& kv : keyframe_poses_)
  {
    auto id = kv.first;
    auto pose_wk = kv.second;

    // set the appropriate color
    if (static_cast<int>(id) >= last_drawn_depth_id) // if the depth is displayed
    {
      if (color_buffer_.find(id) == color_buffer_.end())
        color_buffer_[id] = cfg_.keyframe_color;
      color_buffer_[id] = SmoothDecay(color_buffer_[id], cfg_.keyframe_color, 10.f, delta_time / 1000.f);
    }
    else if (static_cast<int>(id) >= last_drawn_frustum_id) // if the frustum is to be displayed
    {
      color_buffer_[id] = SmoothDecay(color_buffer_[id], cfg_.frame_color, fade_rate_, delta_time / 1000.f);
    }
    else  // fade out
    {
      color_buffer_[id][3] = SmoothDecay(color_buffer_[id][3], 0.f, fate_out_rate_, delta_time / 1000.f);
    }

    if (relin_info[id])
    {
      color_buffer_[id] = cfg_.relin_color;
      relin_info[id] = false;
    }

    glLineWidth(2.);
    Eigen::Vector4f color = color_buffer_[id];
    if (color[3] < 0.01)
      continue;
    glColor4f(color[0], color[1], color[2], color[3]);
    pangolin::glDrawFrustum<double>(cam_.InverseMatrix<double>(),
                                    cam_.width(), cam_.height(),
                                    pose_wk.matrix().cast<double>(), 0.1f);
  }

  // render links between keyframes
  tic("[visualizer] rendering keyframe links");
  for (auto& link : keyframe_links_)
  {
    auto pos1 = keyframe_poses_[link.first].translation();
    auto pos2 = keyframe_poses_[link.second].translation();

    auto col1 = color_buffer_[link.first];
    auto col2 = color_buffer_[link.first];

    if (col1[3] < 0.01 || col2[3] < 0.01)
      continue;

    glLineWidth(1.);
    GLfloat vertices[6] = {pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]};
    GLfloat colors[8] = {col1[0], col1[1], col1[2], col1[3], col2[0], col2[1], col2[2], col2[3]};
    pangolin::glDrawColoredVertices<float, float>(2, vertices, colors, GL_LINES, 3, 4);
  }
  toc("[visualizer] rendering keyframe links");

  // render all frames
  tic("[visualizer] rendering frame frustum and links to keyframes");
  for (auto& kv : frame_poses_)
  {
    // find the connected keyframe
    std::size_t conn_kf = 0;
    for (auto link : frame_links_)
    {
      if (kv.first == link.first)
      {
        conn_kf = link.second;
        break;
      }
    }

    // use the transparency of the connected keyframe
    float alpha = color_buffer_[conn_kf][3];

    // draw frustum and line
    if (draw_all_frames_->Get() || conn_kf == last_id_)
    {
      glLineWidth(1.0);
      glColor4f(cfg_.frame_color[0], cfg_.frame_color[1], cfg_.frame_color[2], alpha);
      pangolin::glDrawFrustum<double>(cam_.InverseMatrix<double>(),
                                      cam_.width(), cam_.height(),
                                      kv.second.matrix().cast<double>(), 0.1f);

      // draw a line
      auto pos1 = keyframe_poses_[conn_kf].translation();
      auto pos2 = kv.second.translation();

      glLineWidth(2);
      Eigen::Vector4f color = cfg_.frame_color;
      glColor4f(color[0], color[1], color[2], alpha);
      pangolin::glDrawLine(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]);
    }
  }
  toc("[visualizer] rendering frame frustum and links to keyframes");
}

/* ************************************************************************* */
void Visualizer::InitOpenGL()
{
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

  glClearColor(cfg_.bg_color[0], cfg_.bg_color[1], cfg_.bg_color[2], cfg_.bg_color[3]);

  // setup a callback to report opengl errors
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback((GLDEBUGPROC) OglDebugCallback, 0);
}

/* ************************************************************************* */
void Visualizer::BuildGui()
{
  // Define Camera Render Object (for view / scene browsing)
  df::PinholeCamera<float> cam(cam_);
  cam.ResizeViewport(cfg_.win_width, cfg_.win_height);
  s_cam = new pangolin::OpenGlRenderState(
//    pangolin::ProjectionMatrixRDF_TopLeft(640,480,420,420,320,240,0.1,1000),
//    pangolin::ProjectionMatrixRDF_TopLeft(cam_.width(),cam_.height(),cam_.fx(),cam_.fy(),cam_.u0(),cam_.v0(),0.1,1000),
    pangolin::ProjectionMatrixRDF_TopLeft(cam.width(), cam.height(), cam.fx(),cam.fy(),cam.u0(),cam.v0(),0.1,1000),
    pangolin::ModelViewLookAtRDF(0,0,-1,0,0,0,0,-1,0)
  );

  keyframe_log_.SetLabels({"Inliers", "Distance"});
  error_log_.SetLabels({"Track error"});

  // Create viewport for map visualization
  pangolin::Attach left_bound = cfg_.demo_mode ? 0.f : pangolin::Attach::Pix(cfg_.panel_width);
  map_viewport_ = &pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, left_bound, 1.0, -640.0f/480.0f)
    .SetHandler(new pangolin::Handler3D(*s_cam));

  // Create viewport for live frame visualization
  if (!cfg_.demo_mode)
  {
    frame_viewport_ = &pangolin::CreateDisplay()
      .SetLock(pangolin::LockLeft, pangolin::LockTop)
      .SetBounds(pangolin::Attach::Pix(cfg_.frame_gap),
                 pangolin::Attach::Pix(cfg_.frame_gap+cfg_.frame_height),
                 pangolin::Attach::Pix(cfg_.panel_width+cfg_.frame_gap),
                 pangolin::Attach::Pix(cfg_.panel_width+cfg_.frame_gap+cfg_.frame_width));
  }

  // Create viewport for residual visualization
  if (!cfg_.demo_mode)
  {
    residual_viewport_ = &pangolin::CreateDisplay()
      .SetLock(pangolin::LockLeft, pangolin::LockTop)
      .SetBounds(pangolin::Attach::Pix(cfg_.frame_gap),
                 pangolin::Attach::Pix(cfg_.frame_gap+cfg_.frame_height),
                 pangolin::Attach::Pix(cfg_.panel_width+2*cfg_.frame_gap+cfg_.frame_width),
                 pangolin::Attach::Pix(cfg_.panel_width+2*cfg_.frame_gap+cfg_.frame_width*2));
  }

  // create a plotter and add it to its view
  kf_plotter_ = new pangolin::Plotter(&keyframe_log_, 0, 100, 0, df_opts_.dist_threshold + 1, 0.1, 0.1);
  kf_plotter_->SetBounds(pangolin::Attach::Pix(cfg_.frame_gap),
                      pangolin::Attach::Pix(cfg_.frame_gap+cfg_.frame_height),
                      pangolin::Attach::Pix(cfg_.panel_width+3*cfg_.frame_gap+cfg_.frame_width*2),
                      pangolin::Attach::Pix(cfg_.panel_width+3*cfg_.frame_gap+cfg_.frame_width*3));
  kf_plotter_->AddMarker(pangolin::Marker::Horizontal, df_opts_.inlier_threshold, pangolin::Marker::LessThan, pangolin::Colour::Red().WithAlpha(0.2f));
  kf_plotter_->AddMarker(pangolin::Marker::Horizontal, df_opts_.dist_threshold, pangolin::Marker::GreaterThan, pangolin::Colour::Green().WithAlpha(0.2f));
  kf_plotter_->Track("$i");
//  pangolin::DisplayBase().AddDisplay(*kf_plotter_);

  // create a plotter and add it to its view
  error_plotter_ = new pangolin::Plotter(&error_log_, 0, 100, 0, 0.05, 0.1, 0.1);
  error_plotter_->SetBounds(pangolin::Attach::Pix(cfg_.frame_gap),
                      pangolin::Attach::Pix(cfg_.frame_gap+cfg_.frame_height),
                      pangolin::Attach::Pix(cfg_.panel_width+4*cfg_.frame_gap+cfg_.frame_width*3),
                      pangolin::Attach::Pix(cfg_.panel_width+4*cfg_.frame_gap+cfg_.frame_width*4));
  error_plotter_->AddMarker(pangolin::Marker::Horizontal, df_opts_.tracking_error_threshold, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f));
  error_plotter_->Track("$i");
//  pangolin::DisplayBase().AddDisplay(*error_plotter_);

  // Create and populate left panel with buttons
  BuildPanel();

  pangolin::RegisterKeyPressCallback('f', pangolin::ToggleVarFunctor("ui.FreeLook"));
  pangolin::RegisterKeyPressCallback('t', pangolin::ToggleVarFunctor("ui.DrawTrajectory"));
  pangolin::RegisterKeyPressCallback('h', pangolin::ToggleVarFunctor("ui.DrawAllFrames"));
}

/* ************************************************************************* */
void Visualizer::BuildPanel()
{
  if (!cfg_.demo_mode)
  {
    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(cfg_.panel_width));
  }

  reset_ = new pangolin::Var<bool>("ui.Reset", false, false);
  record_ = new pangolin::Var<bool>("ui.Record", false, false);
  save_ = new pangolin::Var<bool>("ui.Save", false, false);
  init_ = new pangolin::Var<bool>("ui.Init", false, false);
  phong_ = new pangolin::Var<bool>("ui.Phong", true , true);
  freelook_ = new pangolin::Var<bool>("ui.FreeLook", false, true);
  draw_traj_ = new pangolin::Var<bool>("ui.DrawTrajectory", true, true);
  light_pos_x_ = new pangolin::Var<float>("ui.LightPosX", 0.0, -3.0, 3.0);
  light_pos_y_ = new pangolin::Var<float>("ui.LightPosY", 0.0, -3.0, 3.0);
  light_pos_z_ = new pangolin::Var<float>("ui.LightPosZ", 0.0, -3.0, 3.0);
  trs_damping_ = new pangolin::Var<float>("ui.TransDamping", 0.5, 0.01, 1.5);
  rot_damping_ = new pangolin::Var<float>("ui.RotDamping", 2.0, 0.01, 20.0);
  dpt_filter_sigma_ = new pangolin::Var<float>("ui.Filter sigma", 2, 0, 2);
  draw_noisy_pix_ = new pangolin::Var<bool>("ui.DrawNoisy", false, true);
  draw_all_frames_ = new pangolin::Var<bool>("ui.DrawAllFrames", false, true);
  stdev_thresh_ = new pangolin::Var<float>("ui.StdevThresh", 4.2, 1, 5.0);
  slt_thresh_ = new pangolin::Var<float>("ui.SltThresh", 0.f, 0, 1.0);
  crop_pix_ = new pangolin::Var<int>("ui.CropPix", 20, 0, 30);
  vis_mode_ = new pangolin::Var<int>("ui.Visualize Mode", cfg_.vis_mode, 0, 3);

  huber_delta_ = new pangolin::Var<float>("ui.Mapping huber", df_opts_.huber_delta, 0, 0.5);
  tracking_huber_delta_ = new pangolin::Var<float>("ui.Tracking huber", df_opts_.tracking_huber_delta, 0, 0.5);
  inlier_thresh_ = new pangolin::Var<float>("ui.Inlier thresh", df_opts_.inlier_threshold, 0.1, 1);
  dist_thresh_ = new pangolin::Var<float>("ui.Distance thresh", df_opts_.dist_threshold, 0.5, 10);
  error_thresh_ = new pangolin::Var<float>("ui.Track err thresh", df_opts_.tracking_error_threshold, 0, 1);
  tracking_dist_threshold_ = new pangolin::Var<float>("ui.Track dist thresh", df_opts_.tracking_dist_threshold, 1, 4);
}

/* ************************************************************************* */
void Visualizer::SaveResults()
{

}

/* ************************************************************************* */
void Visualizer::EmitEvent(GuiEvent evt)
{
  event_callback_(evt);
}

} // namespace df
