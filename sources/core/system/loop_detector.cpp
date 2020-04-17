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

#include "loop_detector.h"

namespace df
{

template <typename Scalar>
LoopDetector<Scalar>::LoopDetector(std::string voc_path, MapPtr map, LoopDetectorConfig cfg, CameraPyr cam)
    : voc_(voc_path), db_(voc_, false, 0), map_(map), cfg_(cfg)
{
  // optimize only on top level
//  for (int i = 0; i < cfg.iterations_per_level.size()-1; ++i)
//    cfg.iterations_per_level[i] = 0;
  cfg.tracker_cfg.iterations_per_level = cfg.iters;
  tracker_ = std::make_shared<CameraTracker>(cam, cfg.tracker_cfg);
}

/* ************************************************************************* */
template <typename Scalar>
void LoopDetector<Scalar>::AddKeyframe(KeyframePtr kf)
{
  auto qf = ChangeStructure(kf->features.descriptors);
  voc_.transform(qf, kf->bow_vec);
  db_.add(kf->bow_vec);
  dmap_[kf->id] = kf->bow_vec;
}

/* ************************************************************************* */
template <typename Scalar>
int LoopDetector<Scalar>::DetectLoop(KeyframePtr kf)
{
  auto qf = ChangeStructure(kf->features.descriptors);
  voc_.transform(qf, kf->bow_vec);

  // find the minimum similarity to your neighbours
  VLOG(1) << "Finding least similar neighbour";
  Scalar min_dist = std::numeric_limits<Scalar>::max();
  auto conns = map_->keyframes.GetConnections(kf->id);
  for (auto& c : conns)
  {
    auto dist = voc_.score(dmap_[c], kf->bow_vec);
    VLOG(1) << "Connection to " << c << " score = " << dist;
    if (dist < min_dist)
      min_dist = dist;
  }
  VLOG(1) << "Sim to worst neighbour = " << min_dist;

  // find max 3 candidates
  DBoW2::QueryResults ret;
  db_.query(kf->bow_vec, ret, 3);

  // for each candidate
  VLOG(1) << "Checking " << ret.size() << " candidates";
  for (auto& res : ret)
  {
    auto kfid = res.Id + 1;

    // print L2 between codes
    Scalar l2diff = (map_->keyframes.Get(kfid)->code - kf->code).norm();
    VLOG(1) << "Candidate " << kfid << " similarity: " << res.Score << " code dist: " << l2diff;

    // check if candidate is our neighbour
    if (std::find(conns.begin(), conns.end(), kfid) != conns.end())
      continue;

    // reject ones that are less similiar that worst neighbour
    if (res.Score < min_dist)
      continue;

    VLOG(1) << "accepted";
  }

  return 0;
}

/* ************************************************************************* */
template <typename Scalar>
typename LoopDetector<Scalar>::LoopInfo
LoopDetector<Scalar>::DetectLoop(const ImagePyramid &pyr_img, const GradPyramid &pyr_grad,
                                 const Features& ft, KeyframePtr curr_kf, SE3T pose_cam)
{
  VLOG(1) << "Checking for global loop";

  DBoW2::BowVector bow_vec;
  auto qf = ChangeStructure(ft.descriptors);
  voc_.transform(qf, bow_vec);

  // find similarity with current keyframe
  auto min_score = voc_.score(curr_kf->bow_vec, bow_vec);
  VLOG(1) << "Similarity with our current keyframe: " << min_score;

  // find max 3 candidates
  DBoW2::QueryResults ret;
  db_.query(bow_vec, ret, cfg_.max_candidates);

  // pre-filter candidates by min_score
  VLOG(1) << "Checking " << ret.size() << " candidates from DBoW";
  std::vector<typename KeyframeT::IdType> candidates;
  for (auto& res : ret)
  {
    auto kfid = res.Id + 1;

    VLOG(1) << "Keyframe " << kfid << " similarity: " << res.Score;

    // check if candidate is our current kf
    if (kfid == curr_kf->id)
      continue;

    if (kfid > curr_kf->id - cfg_.active_window)
      continue;

    // reject ones that are less similiar that worst neighbour
//    if (res.Score < min_score)
//      continue;

    if (res.Score < cfg_.min_similarity)
      continue;

//    VLOG(1) << "accepted";
    candidates.push_back(kfid);
  }

  if (candidates.empty())
    return LoopInfo{};

  // do geometry check
  VLOG(1) << "Performing geometry checks on " << candidates.size() << " candidates";
  Scalar best_dist = std::numeric_limits<Scalar>::infinity();
  IdType best_id = candidates[0];
  SE3T best_pose_cam;
  for (auto& id : candidates)
  {
    auto kf = map_->keyframes.Get(id);
    tracker_->SetKeyframe(kf);
    tracker_->Reset();
    tracker_->TrackFrame(pyr_img, pyr_grad);

    auto pose_cam = tracker_->GetPoseEstimate();
    auto dist = (pose_cam.translation() - kf->pose_wk.translation()).norm();

    if (tracker_->GetInliers() < 0.5f)
      continue;

    if (dist < best_dist)
    {
      best_dist = dist;
      best_id = id;
      best_pose_cam = pose_cam;
    }
  }

  VLOG(1) << "Candidate with best distance: " << best_id << "(" << best_dist << ")";

  // do final check on best candidate
  if (best_dist < cfg_.max_dist)
  {
    LoopInfo info;
    info.detected = true;
    info.pose_wc = best_pose_cam;
    info.loop_id = best_id;
    return info;
  }
  else
    VLOG(1) << "candidate rejected";

  return LoopInfo{};
}


/* ************************************************************************* */
template <typename Scalar>
int LoopDetector<Scalar>::DetectLocalLoop(const ImagePyramid &pyr_img, const GradPyramid &pyr_grad,
                                          const Features& ft, KeyframePtr curr_kf, SE3T pose_cam)
{
  Scalar best_dist = std::numeric_limits<Scalar>::infinity();
  IdType best_id = map_->keyframes.LastId();

  VLOG(2) << "Checking for local loop";
  auto ii = map_->keyframes.Ids().rbegin();
  for (int i = 0; i < cfg_.active_window; ++i)
  {
    if (ii == map_->keyframes.Ids().rend())
      break;

    auto id = *ii;
    auto kf = map_->keyframes.Get(id);
    auto pose_wk = kf->pose_wk;
    auto dist = (pose_cam.translation() - pose_wk.translation()).norm();

    VLOG(2) << "Checking id: " << id;
    if (dist < best_dist && id != curr_kf->id)
    {
      best_dist = dist;
      best_id = id;
    }

    ii++;
  }

  // double check that its not the current keyframe
  // in case the loop has not changed the default value
  if (best_id != curr_kf->id && best_dist < cfg_.max_dist)
    return best_id;

  return 0;
}

/* ************************************************************************* */
template <typename Scalar>
void LoopDetector<Scalar>::Reset()
{
  db_.clear();
  dmap_.clear();
}

/* ************************************************************************* */
template <typename Scalar>
std::vector<std::vector<uchar>> LoopDetector<Scalar>::ChangeStructure(cv::Mat mat)
{
  // TODO: detect feature type and transform accordingly
  // ORB...
//  std::vector<cv::Mat> qf(mat.rows);
//  for (int i = 0; i < mat.rows; ++i)
//    qf[i] = mat.row(i);
//  return qf;

  // BRISK
  std::vector<std::vector<uchar>> out(mat.rows);
  for (int k = 0; k < mat.rows; ++k)
  {
    out.at(k).resize(mat.cols);
    memcpy(out.at(k).data(), mat.data + mat.cols * k, mat.cols * sizeof(uchar));
  }
  return out;
}

/* ************************************************************************* */
template class LoopDetector<float>;

} // namespace df
