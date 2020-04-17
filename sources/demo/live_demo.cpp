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
#include "live_demo.h"

#include <opencv2/opencv.hpp>
#include <functional>
#include <boost/filesystem.hpp>

#include "tum_io.h"
#include "logutils.h"
#include "visualizer.h"
#include "timing.h"
#include "camera_interface_factory.h"

namespace df
{

LiveDemoOptions::InitType LiveDemoOptions::InitTypeTranslator(const std::string& s)
{
  if (s == "ONEFRAME")
    return InitType::ONEFRAME;
  else if (s == "TWOFRAME")
    return InitType::TWOFRAME;
  else
    LOG(FATAL) << "Unknown init type: " << s;
  return InitType::ONEFRAME;
}

/* ************************************************************************* */
template <int CS>
LiveDemo<CS>::LiveDemo(LiveDemoOptions opts) : opts_(opts), state_(INIT)
{
  if (opts.calib_path.empty())
    throw std::runtime_error("Camera calibration path not specified in LiveDemo options!");
}

/* ************************************************************************* */
template <int CS>
LiveDemo<CS>::~LiveDemo()
{

}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::Run()
{
  Init();
  StartThreads();
  ProcessingLoop();
  JoinThreads();
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::Init()
{
  // configure profiling
  EnableTiming(opts_.enable_timing);

  // create and setup camera
  caminterface_ = df::drivers::CameraInterfaceFactory::Get()->GetInterfaceFromUrl(opts_.source_url);

  // load live camera intrinsics
  if (caminterface_->HasIntrinsics())
  {
    cam_ = caminterface_->GetIntrinsics();
  }
  else
  {
    cam_ = df::PinholeCamera<float>::FromFile(opts_.calib_path);
  }

  // for debugging -- create windows so that they dont pop up after init
  if (opts_.df_opts.debug)
  {
    cv::namedWindow("loop closures", cv::WINDOW_NORMAL);
    cv::namedWindow("reprojection errors", cv::WINDOW_NORMAL);
    cv::namedWindow("features", cv::WINDOW_NORMAL);
    cv::namedWindow("warping lvl 0", cv::WINDOW_NORMAL);
    cv::namedWindow("keyframes lvl 0", cv::WINDOW_NORMAL);
  }

  // create & initialize slam system
  slam_system_ = std::make_unique<df::DeepFactors<float,CS>>();
  slam_system_->Init(cam_, opts_.df_opts);

  if (!opts_.log_dir.empty())
    CreateLogDirs();

  // create & initialize visualizer
  df::VisualizerConfig vis_cfg;
  vis_cfg.vis_mode = opts_.vis_mode;
  vis_cfg.demo_mode = opts_.demo_mode;
  visualizer_ = std::make_unique<df::Visualizer>(vis_cfg);

  // connect callbacks
  slam_system_->SetMapCallback(std::bind(&Visualizer::OnNewMap, &*visualizer_, std::placeholders::_1));
  slam_system_->SetPoseCallback(std::bind(&Visualizer::OnNewCameraPose, &*visualizer_, std::placeholders::_1));
  visualizer_->SetEventCallback(std::bind(&LiveDemo::HandleGuiEvent, this, std::placeholders::_1));
  slam_system_->SetStatsCallback(std::bind(&LiveDemo::StatsCallback, this, std::placeholders::_1));

  // resize camera to network input
  auto netcfg = slam_system_->GetNetworkConfig();
  cam_.ResizeViewport(netcfg.input_width, netcfg.input_height);

  netcam_ = slam_system_->GetNetworkCam();
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::ProcessingLoop()
{
  LOG(INFO) << "Entering processing loop";

  while (opts_.skip_frames > 0)
  {
    caminterface_->GrabFrames(live_timestamp_, &live_frame_);
    LOG(INFO) << "Skipping frame " << opts_.skip_frames;
    opts_.skip_frames--;
  }

  if (opts_.init_on_start)
  {
    LOG(INFO) << "Initializing system on the first frame";
    std::lock_guard<std::mutex> guard(slam_mutex_);
    double timestamp;
    caminterface_->GrabFrames(live_timestamp_, &live_frame_);
    slam_system_->BootstrapOneFrame(live_timestamp_, live_frame_);
    state_ = DemoState::RUNNING;
  }

  int frame_num = 0;
  while (!quit_ && caminterface_->HasMore())
  {
    if (state_ ==  DemoState::PAUSED)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    static int retries = 4;

    // grab new image from camera
    try
    {
      caminterface_->GrabFrames(live_timestamp_, &live_frame_);
      retries = 4; // reset retries after successful grab
    }
    catch (std::exception& e)
    {
      LOG(ERROR) << "Grab frame error " << retries-- << ": " << e.what();
      if(retries <= 0)
      {
        LOG(ERROR) << "Failed to grab too many times";
        quit_ = true;
      }
      continue;
    }

    if (opts_.df_opts.debug)
      LOG(INFO) << "Processing frame " << live_timestamp_ << "(" << frame_num << ")";

    if (opts_.frame_limit != 0 && frame_num > opts_.frame_limit)
    {
      LOG(INFO) << "Exiting because we've hit the frame limit (" << opts_.frame_limit << ")";
      break;
    }

    // record this image if thats requested
    if(opts_.record_input)
    {
      std::stringstream ss;
      ss << dir_input << "/" << std::setfill('0') << std::setw(4) << frame_num << ".png";
      cv::imwrite(ss.str(), live_frame_);
    }

    // give new frame to visualizer
    visualizer_->OnNewFrame(live_frame_);

    // feed the slam system
//    DeepFactorsStatistics stats;
    if (state_ == RUNNING)
    {
      std::lock_guard<std::mutex> guard(slam_mutex_);
      try
      {
        slam_system_->ProcessFrame(live_timestamp_, live_frame_);
      }
      catch(std::exception& e)
      {
        LOG(INFO) << "Exception in slam: " << + e.what();
        SavePostCrashInfo(true);
        break;
      }
    }

    // for debugging -- show the stuff that slam system has imshowed
    cv::waitKey(opts_.pause_step ? 0 : 1);

    frame_num++;
  }

  if (!caminterface_->HasMore())
    LOG(INFO) << "No more frames to process!";

  if (!opts_.log_dir.empty())
  {
    slam_system_->SaveResults(opts_.log_dir);
    SavePostCrashInfo(); // also save detailed info for debugging the system
  }

  LOG(INFO) << "Finished processing loop";

  if (opts_.quit_on_finish)
    quit_ = true;
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::VisualizerLoop()
{
  LOG(INFO) << "Visualization thread started";

  using namespace std;

  // create and init gui
  // NOTE: need to do this here so that opengl is initialized in the correct thread
  // NOTE(2): there's now a way how to fix that in pangolin
  visualizer_->Init(netcam_, opts_.df_opts);

  const float loop_time = 1000.0f / 60.f;
  double elapsed_ms = 0;
  while (!quit_)
  {
    auto loop_start = chrono::steady_clock::now();

    // visualize
    visualizer_->HandleEvents();
    visualizer_->Draw(elapsed_ms);

    // regulate loop frequency to 60 hz
    auto loop_end = chrono::steady_clock::now();
    elapsed_ms = chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();

    int remaining_ms = loop_time - elapsed_ms;
    if (remaining_ms > 0)
      std::this_thread::sleep_for(chrono::milliseconds(remaining_ms));

    if (remaining_ms < 0)
      LOG(WARNING) << "Visualization loop is running slower than 60 Hz (" << (int)elapsed_ms << " ms)";
  }
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::StatsCallback(const DeepFactorsStatistics& stats)
{
  DisplayStats disp_stats =
    { stats.inliers, stats.distance, stats.tracker_error, stats.residual, stats.relin_info };
  visualizer_->OnNewStats(disp_stats);
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::HandleGuiEvent(GuiEvent evt)
{
  switch (evt)
  {
  case GuiEvent::RESET:
    // reset system and enter init state
    {
      LOG(INFO) << "Resetting the SLAM system and entering INIT state";
      state_ = DemoState::INIT;
      std::lock_guard<std::mutex> guard(slam_mutex_);
      slam_system_->Reset();
    }
    break;
  case GuiEvent::PAUSE:
    LOG(INFO) << (state_ == DemoState::PAUSED ? "Unpausing" : "Pausing");
    if (state_ == DemoState::PAUSED)
      state_ = DemoState::RUNNING;
    else if (state_ == DemoState::RUNNING)
      state_ = DemoState::PAUSED;
    break;
  case GuiEvent::INIT:
    if (state_ == DemoState::INIT)
    {
      // if one frame initialization was requested
      if (opts_.init_type == LiveDemoOptions::ONEFRAME)
      {
        LOG(INFO) << "Initializing the system with one frame";
        slam_system_->BootstrapOneFrame(live_timestamp_, live_frame_);
        state_ = DemoState::RUNNING;
        break;
      }

      // two frame initialization
      if (saved_frame_.empty())
      {
        // save frame
        LOG(INFO) << "Grabbed first frame";
        saved_frame_ = live_frame_.clone();
        saved_timestamp_ = live_timestamp_;
      }
      else
      {
        // grab frame, initialize mapper and enter running state
        LOG(INFO) << "Got the second frame, initializing the system";
        slam_system_->BootstrapTwoFrames(saved_timestamp_, live_timestamp_, saved_frame_, live_frame_);

        LOG(INFO) << "Entering RUNNING state";
        state_ = DemoState::RUNNING;
        saved_frame_.release();
      }
    }
    else
    {
      std::lock_guard<std::mutex> guard(slam_mutex_);
      slam_system_->ForceFrame();
    }
    break;
  case GuiEvent::PARAM_CHANGE:
  {
    std::lock_guard<std::mutex> guard(slam_mutex_);
    opts_.df_opts = visualizer_->GetSlamOpts();
    slam_system_->SetOptions(opts_.df_opts);
    break;
  }
  case GuiEvent::NEW_KF:
  {
    std::lock_guard<std::mutex> guard(slam_mutex_);
    slam_system_->ForceKeyframe();
    break;
  }
  case GuiEvent::EXIT:
    LOG(INFO) << "GUI requested exit";
    quit_ = true;
    break;
  default:
    break;
  };
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::StartThreads()
{
  vis_thread_ = std::thread(std::bind(&LiveDemo::VisualizerLoop, this));
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::JoinThreads()
{
  LOG(INFO) << "Waiting for threads to finish";

  // wait for them to finish
  vis_thread_.join();
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::CreateLogDirs()
{
  if (!opts_.log_dir.empty())
    dir_input = opts_.log_dir + "/input";
  else
    dir_input = "input_" + df::GetTimeStamp();

  if (opts_.record_input)
  {
    CreateDirIfNotExists(dir_input);

    std::ofstream file(dir_input + "/cam.txt");
    file << cam_.fx() << " " << cam_.fy() << " " << cam_.u0() << " " << cam_.v0() << " " << cam_.width() << " " << cam_.height();
    file.close();
  }
}

/* ************************************************************************* */
template <int CS>
void LiveDemo<CS>::SavePostCrashInfo(bool crashed)
{
  // if we're logging somewhere, use that
  // if not, create a timestamped crash dir
  std::string out_dir = !opts_.log_dir.empty() ? opts_.log_dir : "crash_" + df::GetTimeStamp();
  df::CreateDirIfNotExists(out_dir);
  slam_system_->SavePostCrashInfo(out_dir);
  LOG(INFO) << "Saved post-crash info in " << out_dir;


  // If the system has actually crashed, then
  // create an empty file named crash for the
  // parsing scripts to detect that
  if (crashed)
    std::ofstream f(out_dir + "/crash");
}

// explicit instantiation
template class LiveDemo<DF_CODE_SIZE>;

} // namespace df
