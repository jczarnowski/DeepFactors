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
#include <Eigen/Dense>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>

#include "matching.h"

namespace df
{

std::vector<cv::DMatch> PruneMatchesByThreshold(const std::vector<cv::DMatch>& matches,
                                                float max_dist)
{
  std::vector<cv::DMatch> thresholded_matches(matches);
  std::sort(thresholded_matches.begin(), thresholded_matches.end());
  auto dist_lambda = [&] (cv::DMatch& v) -> bool { return v.distance > max_dist; };
  auto first_wrong = std::find_if(thresholded_matches.begin(), thresholded_matches.end(), dist_lambda);
  return std::vector<cv::DMatch>(thresholded_matches.begin(), first_wrong);
}

void ConvertoToBearingVectors(const std::vector<cv::Point2f>& points0,
                              const std::vector<cv::Point2f>& points1,
                              const cv::Mat& cameraMatrix,
                              opengv::bearingVectors_t* bearingVectors0,
                              opengv::bearingVectors_t* bearingVectors1)
{
  cv::Mat undistortedPoints0, undistortedPoints1;
  cv::undistortPoints(points0, undistortedPoints0, cameraMatrix, cv::noArray());
  cv::undistortPoints(points1, undistortedPoints1, cameraMatrix, cv::noArray());

  for (int i = 0; i < undistortedPoints0.rows; ++i)
  {
    auto p0 = undistortedPoints0.at<cv::Point2f>(i);
    auto p1 = undistortedPoints1.at<cv::Point2f>(i);
    opengv::bearingVector_t bvec0(p0.x, p0.y, 1);
    opengv::bearingVector_t bvec1(p1.x, p1.y, 1);
    bearingVectors0->push_back(bvec0.normalized());
    bearingVectors1->push_back(bvec1.normalized());
  }
}

void MatchesToPointVectors(const std::vector<cv::KeyPoint>& keypoints0,
                           const std::vector<cv::KeyPoint>& keypoints1,
                           const std::vector<cv::DMatch>& matches,
                           std::vector<cv::Point2f>* points0,
                           std::vector<cv::Point2f>* points1)
{
  // find the fr1 keypoints corresponding to fr0 keypoints using matches
  for (std::size_t i = 0; i < matches.size(); ++i)
  {
    auto& m = matches[i];
    points0->push_back(keypoints0[m.queryIdx].pt);
    points1->push_back(keypoints1[m.trainIdx].pt);
  }
}

std::vector<cv::DMatch>
PruneMatchesEightPoint(const std::vector<cv::KeyPoint>& keypoints0,
                       const std::vector<cv::KeyPoint>& keypoints1,
                       std::vector<cv::DMatch> matches,
                       cv::Mat cameraMatrix,
                       double threshold,
                       int max_iterations,
                       double prob)
{
  /*
   * The RANSAC paradigm contains three unspecified parameters:
   * (1) the error tolerance used to determine whether or
   *     not a point is compatible with a model,
   * (2) the number of subsets to try
   * (3) the threshold t, which is the number of compatible points
   *     used to imply that the correct model has been found.
   */

  using opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
  using opengv::relative_pose::CentralRelativeAdapter;

  if (keypoints0.empty() || keypoints1.empty())
    return matches;

  // break up matches into separate lists
  std::vector<cv::Point2f> points0, points1;
  MatchesToPointVectors(keypoints0, keypoints1, matches, &points0, &points1);

  // calculate bearing vectors from points using the camera matrix
  opengv::bearingVectors_t bearingVectors0, bearingVectors1;
  ConvertoToBearingVectors(points0, points1, cameraMatrix, &bearingVectors0, &bearingVectors1);

  // Set up a CentralRelativePoseSacProblem with opengv
  CentralRelativeAdapter adapter(bearingVectors0, bearingVectors1);
  opengv::sac::Ransac<CentralRelativePoseSacProblem> ransac;
  ransac.sac_model_ = std::make_shared<CentralRelativePoseSacProblem>
      (adapter, CentralRelativePoseSacProblem::EIGHTPT);
  ransac.threshold_ = threshold;
  ransac.max_iterations_ = max_iterations;
  ransac.probability_ = prob;

  // get the result
  ransac.computeModel();
  opengv::transformation_t transform = ransac.model_coefficients_;
  Eigen::MatrixXd R = transform.block<3,3>(0,0);
  Eigen::MatrixXd t = transform.block<3,1>(0,3);

  // select inliers from matches
  std::vector<cv::DMatch> pruned_matches;
  for (auto& inlier_idx : ransac.inliers_)
    pruned_matches.push_back(matches[inlier_idx]);

  return pruned_matches;
}

} // namespace df
