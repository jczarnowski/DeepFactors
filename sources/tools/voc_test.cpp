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
#include <DBoW2.h>
#include <brisk/brisk.h>
#include "fbrisk.h"

typedef DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>
  FBriskVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>
  FBriskDatabase;

typedef std::vector<std::vector<unsigned char>> DescriptorVec;
typedef brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> BriskDetector;
typedef brisk::BriskDescriptorExtractor BriskExtractor;

DescriptorVec changeStructure(const cv::Mat& mat)
{
//  const int L = 48;

//  DescriptorVec out(mat.rows);

//  unsigned int j = 0;
//  for(unsigned int i = 0; i < mat.rows*mat.cols; i += L, ++j)
//  {
//    out[j].resize(L);
//    std::copy(mat.data + i, mat.data + i + L, out[j].begin());
//  }
  int num_keypoints = mat.rows;

  DescriptorVec out(num_keypoints);
  for (int k = 0; k < num_keypoints; ++k)
  {
    out.at(k).resize(mat.cols);
    memcpy(out.at(k).data(), mat.data + mat.cols * k, mat.cols * sizeof(uchar));
  }

  return out;
}

DescriptorVec LoadExtract(std::string file, BriskDetector& detector,
                          BriskExtractor& extractor)
{
  std::cout << "Loading image from: " << file << std::endl;
  cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
  if (img.empty())
    throw std::runtime_error("Failed to open: " + file);

  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keypoints;
  detector.detect(img, keypoints);
  extractor.compute(img, keypoints, descriptors);

  cv::drawKeypoints(img, keypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow(file, img);

  return changeStructure(descriptors);
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "usage: " << argv[0] << " voc_path img0 img1 [img2..]" << std::endl;
    return -1;
  }
  int nimg = argc-2;

  // load voc
  std::cout << "Loading vocabulary from: " << argv[1] << std::endl;
  FBriskVocabulary voc(argv[1]);

  // Brisk detector/extractor
  BriskDetector detector(20, 2, 200, 400);
  BriskExtractor extractor(true, true);

  // load images and extract feautres
  std::vector<DescriptorVec> features(nimg);
  std::vector<DBoW2::BowVector> bow_vecs(nimg);
  for (int i = 2; i < argc; ++i)
  {
    features[i-2] = LoadExtract(argv[i], detector, extractor);
    voc.transform(features[i-2], bow_vecs[i-2]);
  }

  // compute confusion matrix
  std::cout << "Confusion matrix: " << std::endl;
  for (int i = 0; i < nimg; ++i)
  {
    for (int j = 0; j < nimg; ++j)
      std::cout << std::setw(10) << std::setprecision(3) << voc.score(bow_vecs[i], bow_vecs[j]) << " ";
    std::cout << std::endl;
  }

  cv::waitKey(0);
  return 0;
}
