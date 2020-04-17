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
#include <fstream>
#include <cstring>

#include <json/json.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "decoder_network.h"
#include "tfwrap.h"

namespace df
{

struct DecoderNetwork::Impl
{
  std::unique_ptr<tf::GraphEvaluator> graph_;
  tf::Tensor image_tensor_;
  tf::Tensor code_tensor_;
};

DecoderNetwork::DecoderNetwork(const NetworkConfig &cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>())
{
  tf::SessionOptions opts;
  opts.gpu_memory_allow_growth(true);
  impl_->graph_ = std::make_unique<tf::GraphEvaluator>(cfg.graph_path, opts);
  /*
   * TODO TensorFlow doesn't allow us to set the visible device list to choose the GPU.
   * By default, GPU 0 is selected so it works for us for now, but this absolutely must be fixed.
   */

  // create tensors
  impl_->image_tensor_= tf::Tensor::fromDims<float>({1, (long long)cfg_.input_height, (long long)cfg_.input_width, 1});
  impl_->code_tensor_ = tf::Tensor::fromDims<float>({1, (long long) cfg_.code_size});
}

DecoderNetwork::~DecoderNetwork()
{

}

void DecoderNetwork::Decode(const vc::Buffer2DView<float, vc::TargetHost>& image,
                            const Eigen::MatrixXf& code,
                            vc::RuntimeBufferPyramidView<float, vc::TargetHost>* prx_out,
                            vc::RuntimeBufferPyramidView<float, vc::TargetHost>* stdev_out,
                            vc::RuntimeBufferPyramidView<float, vc::TargetHost>* jac)
{
  // fill tensors
  impl_->code_tensor_.copyFrom(code.data());
  impl_->image_tensor_.copyFrom(image.ptr());

  // setup feed dict
  std::vector<std::pair<std::string, tf::Tensor>> feed_dict =
    {
      {cfg_.input_image_name, impl_->image_tensor_},
      {cfg_.input_code_name, impl_->code_tensor_}
    };

  // setup fetches from the network
  std::vector<std::string> fetches;

  if (prx_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
      fetches.push_back(cfg_.depth_est_names[i]);
  }

  if (stdev_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
      fetches.push_back(cfg_.depth_std_names[i]);
  }

  if (jac)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
      fetches.push_back(cfg_.depth_jac_names[i]);
  }

  // run the graph
  auto outputs = impl_->graph_->Run(feed_dict, fetches);

  // copy depth back
  if (prx_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
    {
      std::size_t width = prx_out->operator[](i).width();
      std::size_t height = prx_out->operator[](i).height();
      vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
      prx_out->operator[](i).copyFrom(view);
    }
    outputs.erase(outputs.begin(), outputs.begin() + cfg_.pyramid_levels);
  }

  // copy stdev back
  if (stdev_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
    {
      std::size_t width = stdev_out->operator[](i).width();
      std::size_t height = stdev_out->operator[](i).height();
      vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
      stdev_out->operator[](i).copyFrom(view);
    }
    outputs.erase(outputs.begin(), outputs.begin() + cfg_.pyramid_levels);
  }

  // copy jacobian back
  if (jac)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
    {
      std::size_t width = jac->operator[](i).width();
      std::size_t height = jac->operator[](i).height();
      vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
      jac->operator[](i).copyFrom(view);
    }
  }
}

void DecoderNetwork::PredictAndDecode(const vc::Buffer2DView<float, vc::TargetHost>& image,
                                      const Eigen::MatrixXf& code,
                                      Eigen::MatrixXf* pred_code,
                                      vc::RuntimeBufferPyramidView<float, vc::TargetHost>* prx_out,
                                      vc::RuntimeBufferPyramidView<float, vc::TargetHost>* stdev_out,
                                      vc::RuntimeBufferPyramidView<float, vc::TargetHost>* jac)
{
  // fill tensors
  impl_->code_tensor_.copyFrom(code.data());
  impl_->image_tensor_.copyFrom(image.ptr());

  // setup feed dict
  std::vector<std::pair<std::string, tf::Tensor>> feed_dict =
    {
      {cfg_.input_image_name, impl_->image_tensor_},
      {cfg_.input_code_name, impl_->code_tensor_}
    };

  // setup fetches from the network
  std::vector<std::string> fetches;
  if (pred_code)
  {
    fetches.push_back(cfg_.code_pred_name);
  }

  if (prx_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
      fetches.push_back(cfg_.depth_est_names[i]);
  }

  if (stdev_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
      fetches.push_back(cfg_.depth_std_names[i]);
  }

  if (jac)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
      fetches.push_back(cfg_.depth_jac_names[i]);
  }

  // run the graph
  auto outputs = impl_->graph_->Run(feed_dict, fetches);

  // copy predicted code back
  if (pred_code)
  {
    std::memcpy(pred_code->data(), outputs[0].data_ptr(), outputs[0].bytes());
    outputs.erase(outputs.begin(), outputs.begin() + 1);
  }

  // copy depth back
  if (prx_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
    {
      std::size_t width = prx_out->operator[](i).width();
      std::size_t height = prx_out->operator[](i).height();
      vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
      prx_out->operator[](i).copyFrom(view);
    }
    outputs.erase(outputs.begin(), outputs.begin() + cfg_.pyramid_levels);
  }

  // copy stdev back
  if (stdev_out)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
    {
      std::size_t width = stdev_out->operator[](i).width();
      std::size_t height = stdev_out->operator[](i).height();
      vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
      stdev_out->operator[](i).copyFrom(view);
    }
    outputs.erase(outputs.begin(), outputs.begin() + cfg_.pyramid_levels);
  }

  // copy jacobian back
  if (jac)
  {
    for (uint i = 0; i < cfg_.pyramid_levels; ++i)
    {
      std::size_t width = jac->operator[](i).width();
      std::size_t height = jac->operator[](i).height();
      vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
      jac->operator[](i).copyFrom(view);
    }
  }
}

DecoderNetwork::NetworkConfig LoadJsonNetworkConfig(const std::string& cfgpath)
{
  std::ifstream ifs(cfgpath);
  if (ifs.fail())
    LOG(FATAL) << "Could not load network config: " << cfgpath;

  Json::Value root;
  Json::parseFromStream(Json::CharReaderBuilder(), ifs, &root, nullptr);

  std::string graph_path = root["graph_path"].asString();
  if (graph_path[0] != '/')
  {
    // parse graph path as relative to the cfg location
    std::string cfgdir(cfgpath);
    cfgdir.erase(cfgdir.rfind("/"));
    graph_path = cfgdir + "/" + graph_path;
  }

  DecoderNetwork::NetworkConfig cfg;
  cfg.graph_path = graph_path;

  CHECK(root.isMember("input_width"));
  cfg.input_width = root["input_width"].asUInt();

  CHECK(root.isMember("input_height"));
  cfg.input_height = root["input_height"].asUInt();

  CHECK(root.isMember("pyramid_levels"));
  cfg.pyramid_levels = root["pyramid_levels"].asUInt();

  CHECK(root.isMember("code_size"));
  cfg.code_size = root["code_size"].asUInt();

  CHECK(root.isMember("grayscale"));
  cfg.grayscale = root["grayscale"].asBool();

  CHECK(root.isMember("avg_dpt"));
  cfg.avg_dpt = root["avg_dpt"].asDouble();

  CHECK(root.isMember("input_names"));
  const auto& input_names = root["input_names"];

  auto cut_after_colon = [] (const std::string& str) -> std::string {
    return str.substr(0, str.find_last_of(":"));
  };

  CHECK(input_names.isMember("image"));
  cfg.input_image_name = cut_after_colon(input_names["image"].asString());

  CHECK(input_names.isMember("code"));
  cfg.input_code_name = cut_after_colon(input_names["code"].asString());

  auto node2stdvec = [cut_after_colon] (const Json::Value& val) -> std::vector<std::string> {
    std::vector<std::string> vec;
    for (Json::Value::ArrayIndex i = 0; i < val.size(); ++i)
      vec.push_back(cut_after_colon(val[i].asString()));
    return vec;
  };

  CHECK(root.isMember("output_names"));
  const auto& output_names = root["output_names"];

  CHECK(output_names.isMember("depth_est") && output_names["depth_est"].isArray());
  cfg.depth_est_names = node2stdvec(output_names["depth_est"]);

  CHECK(output_names.isMember("depth_stdev") && output_names["depth_stdev"].isArray());
  cfg.depth_std_names = node2stdvec(output_names["depth_stdev"]);

  CHECK(output_names.isMember("depth_jac") && output_names["depth_jac"].isArray());
  cfg.depth_jac_names = node2stdvec(output_names["depth_jac"]);

  if (root.isMember("depth_pred") && root["depth_pred"].asBool())
  {
    cfg.depth_pred = true;
    cfg.depth_pred_names = node2stdvec(output_names["depth_pred"]);
    cfg.code_pred_name = node2stdvec(output_names["code_pred"])[0];
  }
  else
  {
    cfg.depth_pred = false;
  }

  CHECK(root.isMember("camera"));
  const auto& camera = root["camera"];
  CHECK(camera.isMember("fx"));
  CHECK(camera.isMember("fy"));
  CHECK(camera.isMember("u0"));
  CHECK(camera.isMember("v0"));
  cfg.camera.fx = camera["fx"].asDouble();
  cfg.camera.fy = camera["fy"].asDouble();
  cfg.camera.u0 = camera["u0"].asDouble();
  cfg.camera.v0 = camera["v0"].asDouble();

  return cfg;
}

} //namespace df
