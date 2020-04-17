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
#ifndef DECODER_NETWORK_H_
#define DECODER_NETWORK_H_

#include <memory>

#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/BufferPyramid.hpp>

namespace df
{

/*
 * Decoder network class
 */
class DecoderNetwork
{
public:
  typedef std::shared_ptr<DecoderNetwork> Ptr;

  struct NetworkConfig
  {
    struct Cam { double fx, fy, u0, v0; };
    Cam camera;

    std::string graph_path;
    std::size_t input_width;
    std::size_t input_height;
    std::size_t pyramid_levels;
    std::size_t code_size;
    bool grayscale;
    double avg_dpt;

    bool depth_pred;

    // input tensor names
    std::string input_image_name;
    std::string input_code_name;

    // output tensor names
    std::string code_pred_name;
    std::vector<std::string> depth_est_names;
    std::vector<std::string> depth_jac_names;
    std::vector<std::string> depth_std_names;
    std::vector<std::string> depth_pred_names;
  };

  DecoderNetwork(const NetworkConfig &cfg);
  ~DecoderNetwork();

  void Decode(const vc::Buffer2DView<float, vc::TargetHost>& image,
              const Eigen::MatrixXf& code,
              vc::RuntimeBufferPyramidView<float, vc::TargetHost>* prx_out = nullptr,
              vc::RuntimeBufferPyramidView<float, vc::TargetHost>* stdev_out = nullptr,
              vc::RuntimeBufferPyramidView<float, vc::TargetHost>* jac = nullptr);

  void PredictAndDecode(const vc::Buffer2DView<float, vc::TargetHost>& image,
                        const Eigen::MatrixXf& code,
                        Eigen::MatrixXf* pred_code,
                        vc::RuntimeBufferPyramidView<float, vc::TargetHost>* prx_out = nullptr,
                        vc::RuntimeBufferPyramidView<float, vc::TargetHost>* stdev_out = nullptr,
                        vc::RuntimeBufferPyramidView<float, vc::TargetHost>* jac = nullptr);

private:
  NetworkConfig cfg_;
  vc::Buffer2DView<float, vc::TargetHost> image_view;
  vc::Buffer1DView<float, vc::TargetHost> code_view;

  // tensorflow in pimpl
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // create vc wrappers around tensor data
  vc::Buffer2DView<float, vc::TargetHost> image_view_;
  vc::Buffer1DView<float, vc::TargetHost> code_view_;
};

/*
 * Network config loading functions
 */
DecoderNetwork::NetworkConfig LoadJsonNetworkConfig(const std::string& cfgpath);

} // namespace df

#endif // DECODER_NETWORK_H_
