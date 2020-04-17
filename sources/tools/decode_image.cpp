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
#include <gflags/gflags.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

#include "decoder_network.h"
#include "intrinsics.h"
#include "timing.h"
#include "display_utils.h"
#include "warping.h"

DEFINE_string(netcfg, "data/nets/scannet256_32.cfg", "Path to protobuf file containing network graph with weights");
DEFINE_string(imgpath, "data/testimg/1052.jpg", "Image to run the network on");
DEFINE_string(gtpath, "", "Path to ground truth depth image");
DEFINE_int32(ntests, 5, "Number of times to run the decoding to average the time");
DEFINE_double(scale, 1, "Scale the jac");
DEFINE_double(gtscale, 1000, "Scale used to convert the gt image to meters");
DEFINE_string(incam, "", "Intrinsic parameters of the input camera image: fx,fy,u0,v0");

using ImagePyramid = vc::RuntimeBufferPyramidManaged<float, vc::TargetHost>;

std::vector<float> split(const std::string &s, char delim)
{
  std::stringstream ss(s);
  std::string item;
  std::vector<float> elems;
  while (std::getline(ss, item, delim))
    elems.push_back(std::stof(item));
  return elems;
}

int main(int argc, char** argv)
{
  gflags::SetUsageMessage("Decode image test");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // create network
  const auto netcfg = df::LoadJsonNetworkConfig(FLAGS_netcfg);
  df::DecoderNetwork decoder(netcfg);

  // shorthand variables
  int pyrlevels = netcfg.pyramid_levels;
  int width = netcfg.input_width;
  int height = netcfg.input_height;

  // load image from disk
  cv::Mat img_orig = cv::imread(FLAGS_imgpath, cv::IMREAD_COLOR);
  CHECK(!img_orig.empty());
  int in_width = img_orig.cols;
  int in_height = img_orig.rows;

  // get the network intrinsics
  df::PinholeCamera<float> netcam(netcfg.camera.fx, netcfg.camera.fy,
                                     netcfg.camera.u0, netcfg.camera.v0,
                                     width, height);

  // parse intrinsics if given in command line
  df::PinholeCamera<float> incam;
  if (not FLAGS_incam.empty())
  {
    auto intr = split(FLAGS_incam, ',');
    CHECK_EQ(intr.size(), 4) << "Malformed intrinsics, should be: fx,fy,u0,v0";
    incam = df::PinholeCamera<float>(intr[0], intr[1], intr[2], intr[3], in_width, in_height);
  }

  // load in ground truth if provided
  cv::Mat gt;
  if (!FLAGS_gtpath.empty())
  {
    gt = cv::imread(FLAGS_gtpath, cv::IMREAD_ANYDEPTH);
    gt.convertTo(gt, CV_32FC1, 1/FLAGS_gtscale);
  }

  // preprocess image
  cv::Mat img;
  cv::cvtColor(img_orig, img, cv::COLOR_RGB2GRAY);
  img.convertTo(img, CV_32FC1, 1/255.0);

  // optionally correct intrinsics
  if (not FLAGS_incam.empty())
  {
    LOG(INFO) << "Correcting intrinsics of the input image";
    img = ChangeIntrinsics(img, incam, netcam);
  }
  else
  {
    LOG(WARNING) << "Input intrinsics not specified in the command "
                    "line arguments, NOT correcting for intrinsics";
    cv::resize(img, img, {width, height});
  }

  // copy it to buffer
  vc::Image2DView<float, vc::TargetHost> img_cpu(img);

  // code
  Eigen::VectorXf code = Eigen::VectorXf::Zero(netcfg.code_size);
  Eigen::MatrixXf pred_code = Eigen::VectorXf::Zero(netcfg.code_size);

  // allocate buffers
  ImagePyramid pyr_prx(pyrlevels, width, height);
  ImagePyramid pyr_prx_pred(pyrlevels, width, height);
  ImagePyramid pyr_stdev(pyrlevels, width, height);
  ImagePyramid pyr_jac(pyrlevels, width, height);

  // predict and decode
  double time_pred_decode = MeasureTimeAverage([&] {
    decoder.PredictAndDecode(img_cpu, code, &pred_code, &pyr_prx, &pyr_stdev, &pyr_jac);
  }, FLAGS_ntests);

  // predict and decode no jacobian
  double time_pred_decode_nojac = MeasureTimeAverage([&] {
    decoder.PredictAndDecode(img_cpu, code, &pred_code, &pyr_prx, &pyr_stdev);
  }, FLAGS_ntests);

  // predict and decode no jacobian
  double time_decode = MeasureTimeAverage([&] {
    decoder.Decode(img_cpu, code, &pyr_prx, &pyr_stdev, &pyr_jac);
  }, FLAGS_ntests);

//  cv::Mat prx_zero = GetOpenCV(pyr_prx[0]).clone();

//  // visualize jacobian
//  for (int i = 0; i < 16; ++i)
//  {

//  }

//  float eps = 10;
//  float all_max = 0;
//  float all_min = 0;
//  float abs_max = 0;
//  std::vector<cv::Mat> jacs;
//  for (int i = 0; i < 16; ++i)
//  {
//    Eigen::VectorXf code_pert = Eigen::VectorXf::Zero(netcfg.code_size);
//    code_pert(i) += eps;

//    // perturb code entry i with eps
//    decoder.Decode(img_cpu, code_pert, &pyr_prx);
//    cv::Mat prx = GetOpenCV(pyr_prx[0]);
//    cv::Mat jac = FLAGS_scale * (prx_zero - prx);

//    double min, max;
//    cv::minMaxIdx(jac, &min, &max);
//    LOG(INFO) << "entry " << i << " min: " << min << " max: " << max;

//    if (min < all_min)
//      all_min = min;

//    if (max > all_max)
//      all_max = max;

//    jac = cv::abs(jac);
//    cv::minMaxIdx(jac, &min, &max);

//    if (max > abs_max)
//      abs_max = max;

//    jacs.push_back(jac);
//  }

//  LOG(INFO) << "all_max = " << all_max;
//  LOG(INFO) << "all_min = " << all_min;

//  cv::Mat img_gray;
//  cv::Mat in[] = {img, img, img};
//  cv::merge(in, 3, img_gray);
//  img_gray.convertTo(img_gray, CV_8UC3, 255, 0);

//  for (int i = 0; i < 16; ++i)
//  {
//    auto jac = jacs[i];

//    jac = jac/abs_max;

//    jac.convertTo(jac_rgb, CV_32FC3);

//    cv::Mat jac_rgb;
//    cv::Mat in[] = {jac, jac, jac};
//    cv::merge(in, 3, jac_rgb);

    // create all colori mage
//    cv::Mat m(height, width, CV_32FC3, cv::Scalar(255.0, 0, 0));

//    LOG(INFO) << m.type();
//    LOG(INFO) << jac_rgb.type();
//    cv::Mat tinted = m.mul(jac_rgb);
//    tinted.convertTo(tinted, CV_8UC3);

//    float alpha = 0.5;
//    cv::Mat combined;

//    cv::addWeighted(tinted, alpha, img_gray, 1-alpha, 0, combined);


//    jac.convertTo(jac, CV_8UC1);

//    cv::Mat jac_disp;
//    cv::applyColorMap(jac, jac_disp, cv::COLORMAP_BONE);
//    jac.convertTo(jac, CV_8UC1, 255.0);

//    cv::imshow("code entry " + std::to_string(i), jac);
//    cv::imwrite("entry_" + std::to_string(i) + ".png", jac);
//  }

  // decode the predicted code
  decoder.Decode(img_cpu, pred_code, &pyr_prx_pred);

  LOG(INFO) << "Number of tests: " << FLAGS_ntests;
  LOG(INFO) << "Average Decode time: " << time_decode << " ms";
  LOG(INFO) << "Average PredictAndDecode time: " << time_pred_decode << " ms";
  LOG(INFO) << "Average PredictAndDecode (No Jacobian) time: " << time_pred_decode_nojac << " ms";

  // recover std from log(std)
  cv::Mat std = GetOpenCV(pyr_stdev[0]);
  cv::exp(std, std);
  std = sqrt(2) * std;

  auto cmap = cv::COLORMAP_JET;

  // display
  cv::resize(img_orig, img_orig, {width, height});
  cv::Mat prx_vis = prx_to_color(pyr_prx[0], cmap);
  cv::Mat prx_pred_vis = prx_to_color(pyr_prx_pred[0], cmap);
  cv::Mat std_vis = apply_colormap(5*std, cv::COLORMAP_JET);
  std::vector<cv::Mat> array = {img_orig, prx_vis, prx_pred_vis};

  // preprocess and display ground truth
  if (!gt.empty())
  {
    if (not FLAGS_incam.empty())
      gt = ChangeIntrinsics(gt, incam, netcam);
    else
      cv::resize(gt, gt, {width, height});

    // convert to prox and display
    cv::Mat gt_prx = df::DepthToProx(gt, netcfg.avg_dpt);
    array.push_back(apply_colormap(gt_prx, cmap));

    // two errors
//    array.push_back(cv::abs(gt_prx-GetOpenCV(pyr_prx[0]))/2.0);
//    array.push_back(cv::abs(gt_prx-GetOpenCV(pyr_prx_pred[0]))/2.0);
  }

  array.push_back(std_vis);

  cv::namedWindow("results", cv::WINDOW_NORMAL);
  cv::imshow("results", CreateMosaic(array, 1, FLAGS_gtpath.empty() ? 3 : 4));

  // wait for user to press q
  LOG(INFO) << "Waiting for 'q'";
  while ('q' != cv::waitKey(0));
}
