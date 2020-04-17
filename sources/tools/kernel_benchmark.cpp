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
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <VisionCore/Image/BufferOps.hpp>

#include "cu_sfmaligner.h"
#include "cuda_context.h"
#include "keyframe.h"
#include "decoder_network.h"
#include "cu_image_proc.h"
#include "image_sequences.h"
#include "camera_pyramid.h"

using namespace df;

static constexpr int CS = DF_CODE_SIZE;

DEFINE_string(netcfg, "data/nets/scannet256_32.cfg", "Path to a network configuration json");
DEFINE_string(seqname, "scenenet_0", "Name of the sequence to run reconstruction on");
DEFINE_string(seqcfg, "data/sequences.json", "Path to json file containing sequence definitions");
DEFINE_uint32(gpu, 0, "Which gpu to use for SLAM");

df::Keyframe<float>::Ptr
CreateKeyframe(std::string img_path, int des_width,
               int des_height, int pyrlevels, int code_size,
               std::shared_ptr<df::DecoderNetwork> decoder_net)
{
  auto kf = std::make_shared<df::Keyframe<float>>(pyrlevels, des_width, des_height, code_size);

  // load and preprocess images
  cv::Mat img_color = cv::imread(img_path);

  if (img_color.empty())
    LOG(FATAL) << "failed to load file: " << img_path;

  // fill code and pose
  kf->code.setZero();
  kf->pose_wk = Sophus::SE3f();

  // fill color img
  cv::resize(img_color, kf->color_img, {des_width, des_height});

  // create a gray image
  cv::Mat img_gray;
  cv::cvtColor(kf->color_img, img_gray, cv::COLOR_RGB2GRAY);
  img_gray.convertTo(img_gray, CV_32FC1, 1/255.0);

  // create pyramids
  for (int i = 0; i < pyrlevels; ++i)
  {

    vc::image::fillBuffer(kf->pyr_vld.GetGpuLevel(i), 1.0f);

    if (i == 0)
    {
      vc::Image2DView<float, vc::TargetHost> tmp1(img_gray);
      kf->pyr_img.GetGpuLevel(0).copyFrom(tmp1);
      df::SobelGradients(kf->pyr_img.GetGpuLevel(0), kf->pyr_grad.GetGpuLevel(0));
      continue;
    }

    df::GaussianBlurDown(kf->pyr_img.GetGpuLevel(i-1), kf->pyr_img.GetGpuLevel(i));
    df::SobelGradients(kf->pyr_img.GetGpuLevel(i), kf->pyr_grad.GetGpuLevel(i));
  }

  // decode zero code
  auto prx_orig_ptr = kf->pyr_prx_orig.GetCpuMutable();
  auto jac_ptr = kf->pyr_jac.GetCpuMutable();
  auto stdev_ptr = kf->pyr_stdev.GetCpuMutable();
  {
    cuda::ScopedContextPop pop;
    const Eigen::VectorXf zero_code = Eigen::VectorXf::Zero(CS);
    decoder_net->Decode(kf->pyr_img.GetCpuLevel(0), zero_code, prx_orig_ptr.get(), stdev_ptr.get(), jac_ptr.get());
  }

  return kf;
}

struct ConfigInfo
{
  int blocks = 1;
  int threads = 32;
  double time = std::numeric_limits<double>::infinity();
};

int main(int argc, char** argv)
{
  gflags::SetUsageMessage("SfmAligner kernel benchmark");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // init cuda context
  auto devinfo = cuda::Init(FLAGS_gpu);

  // create the decoder network
  std::shared_ptr<df::DecoderNetwork> decoder_net;
  auto netcfg = df::LoadJsonNetworkConfig(FLAGS_netcfg);
  {
    cuda::ScopedContextPop ctx;
    decoder_net = std::make_shared<df::DecoderNetwork>(netcfg);
  }

  // create sfmaligner
  auto aligner = std::make_shared<df::SfmAligner<float,CS>>();

  // shorthand variables
  int width = netcfg.input_width;
  int height = netcfg.input_height;
  int code_size = netcfg.code_size;
  int pyrlevels = netcfg.pyramid_levels;
  CHECK_EQ(code_size, CS) << "trying to load network with wrong code size";

  // get image selection
  df::PinholeCamera<float> cam;
  auto seq = GetImageSequence(FLAGS_seqname, FLAGS_seqcfg, cam);

  // resize cam and create pyramid
  cam.ResizeViewport(width, height);
  df::CameraPyramid<float> cam_pyr(cam, pyrlevels);

  // create two keyframes
  auto kf0 = CreateKeyframe(seq.base_dir + "/" + seq.filenames[0], width, height, pyrlevels, code_size, decoder_net);
  auto kf1 = CreateKeyframe(seq.base_dir + "/" + seq.filenames[1], width, height, pyrlevels, code_size, decoder_net);

  Sophus::SE3f pose0, pose1;
  const Eigen::VectorXf zero_code = Eigen::VectorXf::Zero(CS);

  // ----------------- MEASUREMENT FUNCTION ------------------
  auto RunTests = [&] (std::function<void(int,int)> f, int ntests) -> ConfigInfo
  {
    ConfigInfo best_cfg;
    for (int blocks = 1; blocks < 100; blocks +=5)
    {
      for (int threads = 32; threads < 512; threads+=32)
      {
        double avg_time = 0;
        bool failed = false;
        for (int i = 0; i < ntests; ++i)
        {
          try
          {
            auto start = std::clock();
            f(threads, blocks);
            avg_time += (std::clock() - start) / (double) CLOCKS_PER_SEC * 1000;
          }
          catch (...)
          {
            failed = true;
            break;
          }
        }
        avg_time /= ntests;

        std::cout << blocks << " / " << threads << " / " << (!failed ? std::to_string(avg_time) + " ms" : "invalid") << std::endl;
        if (avg_time < best_cfg.time && !failed)
        {
          best_cfg.blocks = blocks;
          best_cfg.threads = threads;
          best_cfg.time = avg_time;
        }
      } // threads
    } // blocks
    return best_cfg;
  };

  // ----------------- ACTUAL BENCHMARK ----------------------
  // gpu warmup
  int i = 0;
  vc::Image2DView<float,vc::TargetDeviceCUDA> vld = kf0->pyr_vld.GetGpuLevel(i);
  aligner->RunStep(pose0, pose1, zero_code, cam_pyr[i], kf0->pyr_img.GetGpuLevel(i),
                   kf1->pyr_img.GetGpuLevel(i), kf0->pyr_dpt.GetGpuLevel(i),
                   kf0->pyr_stdev.GetGpuLevel(i), vld, kf0->pyr_jac.GetGpuLevel(i),
                   kf1->pyr_grad.GetGpuLevel(i));

  // step function
  auto RunStep = [&] (int threads, int blocks)
  {
    aligner->SetStepThreadsBlocks(threads, blocks);
    for (int i = 0; i < pyrlevels; ++i)
    {
      vc::Image2DView<float,vc::TargetDeviceCUDA> vld = kf0->pyr_vld.GetGpuLevel(i);
      aligner->RunStep(pose0, pose1, zero_code, cam_pyr[i], kf0->pyr_img.GetGpuLevel(i),
                       kf1->pyr_img.GetGpuLevel(i), kf0->pyr_dpt.GetGpuLevel(i),
                       kf0->pyr_stdev.GetGpuLevel(i), vld, kf0->pyr_jac.GetGpuLevel(i),
                       kf1->pyr_grad.GetGpuLevel(i));
    }
  };

  // evaluate error function
  auto EvaluateError = [&] (int threads, int blocks)
  {
    aligner->SetEvalThreadsBlocks(threads, blocks);
    for (int i = 0; i < pyrlevels; ++i)
    {
      vc::Image2DView<float,vc::TargetDeviceCUDA> vld = kf0->pyr_vld.GetGpuLevel(i);
      aligner->EvaluateError(pose0, pose1, cam_pyr[i], kf0->pyr_img.GetGpuLevel(i),
                             kf1->pyr_img.GetGpuLevel(i), kf0->pyr_dpt.GetGpuLevel(i),
                             kf0->pyr_stdev.GetGpuLevel(i), kf1->pyr_grad.GetGpuLevel(i));
    }
  };

  int ntests = 4;

  std::cout << "Running tests on SfmAligner::RunStep" << std::endl;
  auto best_cfg_step = RunTests(RunStep, ntests);

  std::cout << "Running tests on SfmAligner::EvaluateError" << std::endl;
  auto best_cfg_eval = RunTests(EvaluateError, ntests);

  std::cout << "[SfmAligner::RunStep] Best configuration: " << best_cfg_step.blocks << " / " << best_cfg_step.threads << " / " << best_cfg_step.time << " ms" << std::endl;
  std::cout << "[SfmAligner::EvaluateError] Best configuration: " << best_cfg_eval.blocks << " / " << best_cfg_eval.threads << " / " << best_cfg_eval.time << " ms" << std::endl;
}
