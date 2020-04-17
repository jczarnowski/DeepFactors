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
#ifndef DF_LIVE_DEMO_H_
#define DF_LIVE_DEMO_H_

#include <memory>
#include <vector>
#include <thread>

#include "deepfactors.h"
#include "visualizer.h" // for enum GuiEvent

namespace df
{

// forward declarations
class Visualizer;
enum GuiEvent;
namespace drivers { class CameraInterface; }

struct LiveDemoOptions
{
  using VisMode = VisualizerConfig::VisualizeMode;
  enum InitType {ONEFRAME, TWOFRAME};

  VisMode               vis_mode = VisMode::ALL;
  InitType              init_type = TWOFRAME;
  df::DeepFactorsOptions   df_opts;
  std::string           source_url;
  std::string           calib_path;
  bool                  record_input;
  std::string           record_path;
  bool                  init_on_start;
  bool                  pause_step;
  std::string           log_dir;
  std::string           res_dir_name;
  bool                  enable_timing;
  bool                  quit_on_finish;
  int                   frame_limit;
  int                   skip_frames;
  bool                  demo_mode;

  static InitType InitTypeTranslator(const std::string& s);
};

/*
 Class representing the live demo program
 Handles system things:
    Starting/stopping threads
    Fetching new frame from camera
    Feeding the slam system
*/
template <int CS>
class LiveDemo
{
public:
  enum DemoState { RUNNING, PAUSED, INIT };

  typedef df::DeepFactors<float,CS> SlamT;

  LiveDemo(LiveDemoOptions opts);
  ~LiveDemo();

  void Run();

private:
  void Init();
  void ProcessingLoop();
  void VisualizerLoop();
  void StatsCallback(const DeepFactorsStatistics &stats);
  void HandleGuiEvent(GuiEvent evt);

  void StartThreads();
  void JoinThreads();

  void CreateLogDirs();
  void SavePostCrashInfo(bool crashed=false);


  df::LiveDemoOptions opts_;
  std::unique_ptr<SlamT> slam_system_;
  std::unique_ptr<df::Visualizer> visualizer_;
  std::unique_ptr<df::drivers::CameraInterface> caminterface_;

  double live_timestamp_;
  double saved_timestamp_;
  cv::Mat live_frame_;
  cv::Mat saved_frame_;
  std::atomic<DemoState> state_;
  bool quit_ = false;

  std::thread vis_thread_;
  std::mutex slam_mutex_;
  df::PinholeCamera<float> cam_; // connected camera intrinsics
  df::PinholeCamera<float> netcam_; // network camera intrinsics

  // directories that demo is logging to
  std::string dir_input;
  std::string dir_crash;
};

} // namespace df

#endif // DF_LIVE_DEMO_H_

