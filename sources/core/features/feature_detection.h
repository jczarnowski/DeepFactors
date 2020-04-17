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
#ifndef DF_FEATURE_DETECTION_H_
#define DF_FEATURE_DETECTION_H_

#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <glog/logging.h>
#include <brisk/brisk.h>

namespace df
{

struct Features
{
  enum Type { ORB, BRISK };
  typedef std::vector<cv::KeyPoint> Keypoints;

  Features() {}

  Features(Type type)
    : type(type) {}

  Keypoints keypoints;
  cv::Mat descriptors;
  Type type;
};

class FeatureDetector
{
public:
  FeatureDetector() {}
  virtual ~FeatureDetector() {}

  virtual Features DetectAndCompute(const cv::Mat& img) = 0;
};

struct BriskConfig
{
  // detector params
  double uniformity_rad = 5;
  std::size_t octaves = 4;
  double absolute_threshold = 200;
  std::size_t max_num_kpt = 400;

  // extractor params
  bool rotation_invariant = true;
  bool scale_invariant = false;
};

class BriskDetector : public FeatureDetector
{
public:
  typedef brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> Detector;
  typedef brisk::BriskDescriptorExtractor Extractor;

  BriskDetector(BriskConfig cfg = BriskConfig{})
  {
    detector_ = std::make_unique<Detector>(cfg.uniformity_rad, cfg.octaves,
                                           cfg.absolute_threshold,
                                           cfg.max_num_kpt);
    extractor_ = std::make_unique<Extractor>(cfg.rotation_invariant,
                                             cfg.scale_invariant);
  }

  virtual ~BriskDetector() override {}

  virtual Features DetectAndCompute(const cv::Mat& img) override
  {
    CHECK_EQ(img.channels(), 1) << "Non-grayscale image passed to BriskDetector";
    Features f{Features::BRISK};
    detector_->detect(img, f.keypoints);
    extractor_->compute(img, f.keypoints, f.descriptors);
    return f;
  }

private:
  std::unique_ptr<Detector> detector_;
  std::unique_ptr<Extractor> extractor_;
};

class OrbDetector : public FeatureDetector
{
public:
  OrbDetector(int nfeatures, float scale_factor, int nlevels)
  {
    orb_ = cv::ORB::create(nfeatures, scale_factor, nlevels);
  }

  virtual ~OrbDetector() override {}

  virtual Features DetectAndCompute(const cv::Mat& img) override
  {
    CHECK_EQ(img.channels(), 1) << "Non-grayscale image passed to BriskDetector";
    Features f{Features::ORB};
    orb_->detectAndCompute(img, cv::Mat{}, f.keypoints, f.descriptors);
    return f;
  }

private:
  cv::Ptr<cv::ORB> orb_;
};

} // namespace df

#endif // DF_FEATURE_DETECTION_H_
