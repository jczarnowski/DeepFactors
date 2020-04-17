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
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <brisk/brisk.h>
#include <DBoW2.h>

#include "tum_interface.h"
#include "fbrisk.h"

typedef DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>
  FBriskVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>
  FBriskDatabase;

typedef std::vector<std::vector<unsigned char>> DescriptorVec;
typedef brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> BriskDetector;
typedef brisk::BriskDescriptorExtractor BriskExtractor;

DEFINE_string(tum_path, "", "Path to TUM dataset");
DEFINE_string(outname, "tum_voc", "Output vocabulary name");
DEFINE_int32(k, 10, "Branching factor");
DEFINE_int32(L, 6, "Depth levels");

std::vector<std::string> tum_sequences = {
  "rgbd_dataset_freiburg1_360",
  "rgbd_dataset_freiburg1_desk",
  "rgbd_dataset_freiburg1_desk2",
  "rgbd_dataset_freiburg1_floor",
  "rgbd_dataset_freiburg1_room",
//  "rgbd_dataset_freiburg2_360_hemisphere",
//  "rgbd_dataset_freiburg2_360_kidnap",
//  "rgbd_dataset_freiburg2_desk",
//  "rgbd_dataset_freiburg2_desk_with_person",
//  "rgbd_dataset_freiburg2_large_no_loop",
//  "rgbd_dataset_freiburg2_large_with_loop",
//  "rgbd_dataset_freiburg2_pioneer_360",
//  "rgbd_dataset_freiburg2_pioneer_slam",
//  "rgbd_dataset_freiburg2_pioneer_slam2",
//  "rgbd_dataset_freiburg2_pioneer_slam3",
//  "rgbd_dataset_freiburg3_long_office_household",
//  "rgbd_dataset_freiburg3_nostructure_notexture_far",
//  "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
//  "rgbd_dataset_freiburg3_nostructure_texture_far",
//  "rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
//  "rgbd_dataset_freiburg3_structure_notexture_far",
//  "rgbd_dataset_freiburg3_structure_notexture_near",
//  "rgbd_dataset_freiburg3_structure_texture_far",
//  "rgbd_dataset_freiburg3_structure_texture_near",
//  "rgbd_dataset_freiburg3_walking_rpy",
};

DescriptorVec changeStructure(const cv::Mat& mat)
{
  const int L = 48;

  DescriptorVec out(mat.rows);

  int j = 0;
  for(int i = 0; i < mat.rows*mat.cols; i += L, ++j)
  {
    out[j].resize(L);
    std::copy(mat.data + i, mat.data + i + L, out[j].begin());
  }

  return out;
}

int main(int argc, char *argv[])
{
  // parse command line flags
  google::SetUsageMessage("Orb Vocabulary Builder");
  google::ParseCommandLineFlags(&argc, &argv, true);

  // init google logging
  FLAGS_colorlogtostderr = true;
  google::LogToStderr();
  google::InitGoogleLogging(argv[0]);

  BriskDetector detector(20, 2, 200,400);
  BriskExtractor extractor(true, true);

  // create dataset interface
  std::vector<DescriptorVec> features;
  for (uint ii = 0; ii < tum_sequences.size(); ++ii)
  {
    std::string seq = tum_sequences[ii];
    LOG(INFO) << "Processing sequence " << ii+1 << "/" << tum_sequences.size() << ": " << seq;

    std::string seq_dir = FLAGS_tum_path + "/" + seq;
    df::drivers::TumInterface interface(seq_dir);

    while (interface.HasMore())
    {
      cv::Mat img;
      double timestamp;
      interface.GrabFrames(timestamp, &img);

      cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

      cv::Mat descriptors;
      std::vector<cv::KeyPoint> keypoints;
      detector.detect(img, keypoints);
      extractor.compute(img, keypoints, descriptors);

      cv::drawKeypoints(img, keypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::imshow("current_img", img);
      cv::waitKey(2);

      // change structure
      features.push_back(changeStructure(descriptors));
    }
  }

  // build vocabulary
  FBriskVocabulary voc(FLAGS_k, FLAGS_L, DBoW2::TF_IDF, DBoW2::L1_NORM);

  LOG(INFO) << "Creating vocabulary using " << features.size() << " images";
  voc.create(features);
  LOG(INFO) << "Done";

  voc.save(FLAGS_outname + ".yml.gz");
}
