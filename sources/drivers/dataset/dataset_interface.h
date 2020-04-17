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
#ifndef DF_DATASET_INTERFACE_H_
#define DF_DATASET_INTERFACE_H_

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "camera_interface.h"

namespace df
{
namespace drivers
{

struct DatasetFrame
{
  double timestamp;
  cv::Mat img;
  cv::Mat dpt;
  Sophus::SE3f pose_wf;
};

class DatasetInterface : public CameraInterface
{
public:
  DatasetInterface() {}
  virtual ~DatasetInterface() {}

  virtual std::vector<DatasetFrame> GetAll() = 0;
  virtual std::vector<Sophus::SE3f> GetPoses() { return std::vector<Sophus::SE3f>{}; }
  virtual bool HasPoses() = 0;
  virtual bool HasMore() = 0;
};

} // namespace drivers
} // namespace df

#endif // DF_DATASET_INTERFACE_H_

