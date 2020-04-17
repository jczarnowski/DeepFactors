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
#ifndef DF_VISUALIZER_H_
#define DF_VISUALIZER_H_

#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

#include "keyframe_map.h"
#include "pinhole_camera.h"
#include "keyframe_renderer.h"
#include "deepfactors_options.h"

namespace cv { class Mat; }

namespace df
{

enum GuiEvent
{
  RESET,
  PAUSE,
  INIT,
  EXIT,
  PARAM_CHANGE,
  NEW_KF,
};

struct VisualizerConfig
{
  enum VisualizeMode {ALL=0, FIRST, LAST, LAST_N};

  VisualizeMode   vis_mode = ALL;
  std::string     win_title = "DeepFactors";
  int             win_width = 1600;
  int             win_height = 900;

  // colorscheme
  Eigen::Vector4f bg_color{0., 0., 0., 1.}; // for video
//  Eigen::Vector4f bg_color{1., 1., 1., 1.}; // for the paper
  Eigen::Vector4f cam_color{0.929, 0.443, 0.576, 1.};
  Eigen::Vector4f link_color{1.0, 0.0, 0.0, 1.};
  Eigen::Vector4f keyframe_color{0.58, 0.235, 1., 1.};
  Eigen::Vector4f frame_color{0.255, 0.063, 0.557, 1.};
  Eigen::Vector4f relin_color{0.992, 0.616, 0.322, 1.};

  int             panel_width = 200;
  int             frame_gap = 10;
  int             frame_width = 200;
  int             frame_height = 200;

  bool            demo_mode = false;

  static VisualizeMode VisualizeModeTranslator(const std::string& s);
};

struct DisplayStats
{
  float inliers;
  float distance;
  float tracker_error;
  cv::Mat residual_img;
  std::map<std::size_t, bool> relin_info;
};

class Visualizer
{
public:
  typedef std::function<void(GuiEvent)> EventCallback;
  typedef df::Map<float> MapT;
  typedef MapT::KeyframeT KeyframeT;
  typedef std::map<int, GuiEvent> KeyConfig;
  typedef MapT::KeyframeGraphT FrameGraphT;
  typedef FrameGraphT::LinkContainer LinkContainerT;
  typedef FrameGraphT::LinkT LinkT;
  typedef KeyframeRenderer::DisplayData KeyframeDisplayData;

  struct DisplayCacheItem
  {
    DisplayCacheItem(int width, int height)
      : data(width, height), active(false) {}

    KeyframeRenderer::DisplayData data;
    bool active;
  };

  Visualizer();
  Visualizer(const VisualizerConfig& cfg);
  virtual ~Visualizer();

  void Init(const df::PinholeCamera<float>& cam, df::DeepFactorsOptions df_opts = df::DeepFactorsOptions{});
  bool ShouldQuit();

  void HandleEvents();
  void SyncOptions();
  void Draw(float delta_time);

  void OnNewFrame(const cv::Mat& frame);
  void OnNewCameraPose(const Sophus::SE3f& pose_wc);
  void OnNewMap(MapT::Ptr map);
  void OnNewStats(const DisplayStats& stats);

  void SetEventCallback(EventCallback cb);
  void SetKeymap(KeyConfig keycfg);

  void KeyCallback(int key);

  float GetHuberDelta() { return huber_delta_->Get(); }
  df::DeepFactorsOptions GetSlamOpts() const { return df_opts_; }

private:
  void RenderKeyframes(float delta_time);
  void InitOpenGL();
  void BuildGui();
  void BuildPanel();
  void SaveResults();

  void EmitEvent(GuiEvent evt);

  // data to display
  df::PinholeCamera<float> cam_;
  Sophus::SE3f cam_pose_wc_;
  cv::Mat live_frame_;
  LinkContainerT keyframe_links_;
  LinkContainerT frame_links_;
  pangolin::DataLog keyframe_log_;
  pangolin::DataLog error_log_;
  cv::Mat residual_img_;
  uint last_id_;
  bool initialized_ = false;

  // visualizer related things
  VisualizerConfig cfg_;
  KeyConfig keymap_;
  EventCallback event_callback_;
  df::KeyframeRenderer kfrenderer_;

  // GUI-related objects
  pangolin::View* map_viewport_;
  pangolin::View* frame_viewport_;
  pangolin::View* residual_viewport_;
  pangolin::OpenGlRenderState* s_cam;
  pangolin::GlTexture* image_tex = nullptr;
  pangolin::GlTexture* residual_tex = nullptr;
  pangolin::Plotter* error_plotter_;
  pangolin::Plotter* kf_plotter_;

  // gui/control options
  pangolin::Var<bool>* reset_;
  pangolin::Var<bool>* record_;
  pangolin::Var<bool>* save_;
  pangolin::Var<bool>* init_;
  pangolin::Var<bool>* phong_;
  pangolin::Var<bool>* freelook_;
  pangolin::Var<bool>* draw_traj_;
  pangolin::Var<bool>* draw_noisy_pix_;
  pangolin::Var<bool>* draw_all_frames_;
  pangolin::Var<float>* stdev_thresh_;
  pangolin::Var<float>* slt_thresh_;
  pangolin::Var<int>* crop_pix_;
  pangolin::Var<float>* light_pos_x_;
  pangolin::Var<float>* light_pos_y_;
  pangolin::Var<float>* light_pos_z_;
  pangolin::Var<float>* trs_damping_;
  pangolin::Var<float>* rot_damping_;
  pangolin::Var<float>* dpt_filter_sigma_;
  pangolin::Var<int>* vis_mode_;

  // algorithm related options
  pangolin::Var<float>* huber_delta_;
  pangolin::Var<float>* tracking_huber_delta_;
  pangolin::Var<int>* keyframe_mode_;
  pangolin::Var<float>* dist_thresh_;
  pangolin::Var<float>* inlier_thresh_;
  pangolin::Var<float>* error_thresh_;
  pangolin::Var<float>* jump_thresh_;
  pangolin::Var<float>* tracking_dist_threshold_;

  // mutexes
  std::mutex frame_mutex_;
  std::mutex campose_mutex_;
  std::mutex map_mutex_;
  std::mutex stats_mutex_;

  std::map<KeyframeT::IdType, DisplayCacheItem> display_data_;
  std::map<KeyframeT::IdType, Sophus::SE3f> keyframe_poses_;
  std::map<KeyframeT::IdType, Sophus::SE3f> frame_poses_;
  std::map<KeyframeT::IdType, bool> frame_mrg_;
  std::map<KeyframeT::IdType, bool> relin_info_;
  df::DeepFactorsOptions df_opts_;
  bool new_map_;

  // camera follow target
  Sophus::SE3f follow_pose_;

  // state for the smooth camera follow
  Eigen::Vector3f camera_vel_;
  Eigen::Vector4f camera_rot_vel_;

  const int num_visible_keyframes_ = 5;  // number of copied and displayed keyframes
  const int max_visible_frustums_ = 25;  // number of last keyframes frustums to display

  // buffer for holding fading colors of keyframe frustums
  std::map<std::size_t, Eigen::Vector4f> color_buffer_;
  float fade_rate_ = 1.f;
  float fate_out_rate_ = 1.f;
};

} // namespace df

#endif // DF_VISUALIZER_H_
