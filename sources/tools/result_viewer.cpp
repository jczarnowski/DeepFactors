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
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <VisionCore/Image/BufferOps.hpp>

#include "camera_interface_factory.h"
#include "dataset_interface.h"
#include "keyframe_renderer.h"
#include "tum_io.h"

#define WIN_WIDTH 640
#define WIN_HEIGHT 480
#define PANEL_WIDTH 150

using namespace Sophus;
using DisplayData = df::KeyframeRenderer::DisplayData;

std::string source_url_help = "Image source URL" + df::drivers::CameraInterfaceFactory::Get()->GetUrlHelp();
DEFINE_string(source_url, "", source_url_help.c_str());
DEFINE_string(trajectory_file, "", "Path to a file with estimated trajectory");
DEFINE_bool(draw_depths, false, "Draw ground-truth depth images");
DEFINE_string(save_gt, "", "Save the ground truth trajectory to TUM format");
DEFINE_int32(depth_skip, 20, "Skip depth maps for display");
DEFINE_bool(no_display, false, "Quit after loading/saving trajectories");

void RenderTrajectory(const std::vector<SE3f>& poses, df::PinholeCamera<float> cam,
                      bool draw_frames, float frame_col[4], float link_col[4], float scale=1.0f,
                      float frame_width=0.5f, float frame_scale=0.1f, float link_width=2.f)
{
  // render gt poses
  for (uint i = 0; i < poses.size(); ++i)
  {
    if (draw_frames)
    {
      glLineWidth(frame_width);
      glColor4f(frame_col[0], frame_col[1], frame_col[2], frame_col[3]);
      auto pose = poses[i];
      pose.translation() *= scale;
      pangolin::glDrawFrustum<double>(cam.InverseMatrix<double>(),
                                      cam.width(), cam.height(),
                                      pose.matrix().cast<double>(), frame_scale);
    }

    if (i > 0)
    {
      Eigen::Vector3f pos1 = poses[i].translation() * scale;
      Eigen::Vector3f pos2 = poses[i-1].translation() * scale;
      glLineWidth(link_width);
      glColor4f(link_col[0], link_col[1], link_col[2], link_col[3]);
      pangolin::glDrawLine(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]);
    }
  }
}

std::vector<float> split(const std::string &s, char delim)
{
  std::stringstream ss(s);
  std::string item;
  std::vector<float> elems;
  while (std::getline(ss, item, delim))
    elems.push_back(std::stof(item));
  return elems;
}

std::vector<SE3f> LoadTrajectoryTUM(std::string path)
{
  std::ifstream f(path);
  if (not f)
  {
    LOG(WARNING) << "Failed to open trajectory file: '" << path << "'";
    return std::vector<SE3f>{};
  }

  std::string line;
  std::vector<SE3f> est_poses;
  while (std::getline(f, line))
  {
    auto vec = split(line, ' ');
    CHECK_EQ(vec.size(), 8) << "Malformed line in trajectory file: " << line;

    float timestamp = vec[0];
    float tx = vec[1];
    float ty = vec[2];
    float tz = vec[3];
    float qx = vec[4];
    float qy = vec[5];
    float qz = vec[6];
    float qw = vec[7];

    Eigen::Vector3f t(tx, ty, tz);
    Eigen::Quaternionf q(qw, qx, qy, qz);
    est_poses.push_back(SE3f(q, t));
  }
  return est_poses;
}

 void SaveTrajectoryTUM(std::string path, const std::vector<SE3f>& trajectory)
 {
   std::ofstream f(path);
   CHECK(f) << "Failed to open for writing: " << path;

   for (uint i = 0; i < trajectory.size(); ++i)
   {
     auto& pose = trajectory[i];
     TumPose tum_pose{static_cast<double>(i), pose.so3().unit_quaternion(), pose.translation()};
     f << tum_pose << std::endl;
   }
 }

int main(int argc, char *argv[])
{
  // parse command line flags
  google::SetUsageMessage("N-SLAM result viewer");
  google::ParseCommandLineFlags(&argc, &argv, true);

  // init google logging
  FLAGS_colorlogtostderr = true;
  google::LogToStderr();
  google::InitGoogleLogging(argv[0]);

  // create dataset interface
  auto interface = df::drivers::CameraInterfaceFactory::Get()->GetInterfaceFromUrl(FLAGS_source_url);
  auto data_interface = dynamic_cast<df::drivers::DatasetInterface*>(interface.get());
  CHECK(data_interface != nullptr) << "Source url does not represent a dataset";
  CHECK(data_interface->HasIntrinsics()) << "Dataset does not have intrinsics";
  CHECK(data_interface->HasPoses()) << "Dataset does not have poses";

  // load data to display
  auto cam = data_interface->GetIntrinsics();
  auto gt_poses = data_interface->GetPoses();
  std::vector<SE3f> est_poses;

  if (not FLAGS_trajectory_file.empty())
  {
    LOG(INFO) << "Loading estimated trajectory to display";
    est_poses = LoadTrajectoryTUM(FLAGS_trajectory_file);
  }

  if (not FLAGS_save_gt.empty())
  {
    LOG(INFO) << "Saving ground truth trajectory to: " << FLAGS_save_gt;
    SaveTrajectoryTUM(FLAGS_save_gt, gt_poses);
  }

  if (FLAGS_no_display)
  {
    LOG(INFO) << "Exiting as requested with --no_display";
    exit(0);
  }

  // load depth maps if it's needed
  std::vector<DisplayData> display_data;
  if (data_interface->SupportsDepth() && FLAGS_draw_depths)
  {
    LOG(INFO) << "Loading entire dataset...";
    auto frames = data_interface->GetAll();

    LOG(INFO) << "Processing the dataset...";
    for (uint i = 0; i < frames.size(); i += FLAGS_depth_skip)
    {
      display_data.emplace_back(cam.width(), cam.height());
      auto& dd = display_data.back();
      auto& frame = frames[i];
      dd.color_img = frame.img;
      dd.dpt.copyFrom(frame.dpt);
      dd.dpt_orig.copyFrom(frame.dpt);
      dd.pose_wk = frame.pose_wf;
      vc::image::fillBuffer(dd.std, 0);
      vc::image::fillBuffer(dd.vld, 1);
    }
  }

  // opengl & pangolin init
  pangolin::CreateWindowAndBind("Viewer", WIN_WIDTH, WIN_HEIGHT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_BLEND);

  // create and initialize keyframe renderer
  df::KeyframeRenderer kf_render;
  kf_render.Init(cam);
  kf_render.SetCropPix(0);
  kf_render.SetDrawNoisyPixels(false);
  kf_render.SetLightPos(0,0,0);
  kf_render.SetPhong(true);
  kf_render.SetSltThresh(0.111f);
  kf_render.SetStdevThresh(3.0);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrixRDF_TopLeft(cam.width(),cam.height(),cam.fx(),cam.fy(),cam.u0(),cam.v0(),0.1,1000),
    pangolin::ModelViewLookAtRDF(0,0,-1,0,0,0,0,-1,0)
  );

  // Create viewport for map visualization
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH), 1.0, -(float)WIN_WIDTH/WIN_HEIGHT)
    .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));
  pangolin::Var<bool> draw_gt_frames("UI.Draw GT Frames", true, true);
  pangolin::Var<bool> draw_est_frames("UI.Draw Est Frames", true, true);
  pangolin::Var<bool> draw_gt_depths("UI.Draw GT Depths", data_interface->SupportsDepth() && FLAGS_draw_depths, true);
  pangolin::Var<bool> phong("UI.Phong", true, true);
  pangolin::Var<double> scale("UI.Scale", 1.0, 0.01, 1.0);

  while (!pangolin::ShouldQuit())
  {
    // Clear entire screen
    glClearColor(0, 0, 0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate 3d window
    d_cam.Activate(s_cam);

    // render gt poses
    float gt_frame_col[4] = {0.0f, 1.0f, 1.0f, 0.2f};
    float gt_link_col[4] = {0.0, 0.0, 1.0, 1.0f};
    RenderTrajectory(gt_poses, cam, draw_gt_frames, gt_frame_col, gt_link_col);

    // render est poses
    if (not FLAGS_trajectory_file.empty())
    {
      float est_frame_col[4] = {1.0f, 0.0f, 0.0f, 0.2f};
      float est_link_col[4] = {1.0f, 0.0, 0.0, 1.0f};
      RenderTrajectory(est_poses, cam, draw_est_frames, est_frame_col, est_link_col, scale);
    }

    // update renderer settings
    kf_render.SetPhong(phong);

    // render gt depth
    if (draw_gt_depths && FLAGS_draw_depths)
    {
      pangolin::OpenGlMatrix vp = s_cam.GetProjectionModelViewMatrix();
      for (uint i = 0; i < display_data.size(); i++)
        kf_render.RenderKeyframe(vp, display_data[i]);
    }

    // finish frame
    pangolin::FinishFrame();
  }

  // cleanup
  google::ShutdownGoogleLogging();
  gflags::ShutDownCommandLineFlags();

  return 0;
}
