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
#ifndef DF_MATCHING_H_
#define DF_MATCHING_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <opengv/types.hpp>

namespace df
{

std::vector<cv::DMatch> PruneMatchesByThreshold(const std::vector<cv::DMatch>& matches,
                                                float max_dist = 40.f);

void ConvertoToBearingVectors(const std::vector<cv::Point2f>& points0,
                              const std::vector<cv::Point2f>& points1,
                              const cv::Mat& cameraMatrix,
                              opengv::bearingVectors_t* bearingVectors0,
                              opengv::bearingVectors_t* bearingVectors1);

void MatchesToPointVectors(const std::vector<cv::KeyPoint>& keypoints0,
                           const std::vector<cv::KeyPoint>& keypoints1,
                           const std::vector<cv::DMatch>& matches,
                           std::vector<cv::Point2f>* points0,
                           std::vector<cv::Point2f>* points1);

std::vector<cv::DMatch>
PruneMatchesEightPoint(const std::vector<cv::KeyPoint>& keypoints0,
                       const std::vector<cv::KeyPoint>& keypoints1,
                       std::vector<cv::DMatch> matches,
                       cv::Mat cameraMatrix,
                       double threshold = 0.0001,
                       int max_iterations = 1000,
                       double prob = 0.99);

} // namespace df

#endif // DF_MATCHING_H_
