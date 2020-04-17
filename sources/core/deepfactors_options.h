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
#ifndef DF_OPTIONS_H_
#define DF_OPTIONS_H_

#include <vector>
#include <string>
#include "mapper.h"

namespace df
{

struct DeepFactorsOptions
{
  typedef std::vector<int> IntVector;
  typedef MapperOptions::ConnectionMode ConnMode;
  enum KeyframeMode { AUTO=0, AUTO_COMBINED, NEVER };
  enum TrackingMode { CLOSEST=0, LAST, FIRST };

  /* required parameters */
  std::size_t     gpu = 0;
  std::string     network_path;
  std::string     vocabulary_path;

  /* iterations */

  /* camera tracking */
  IntVector       tracking_iterations = {4, 6, 5, 10};
  TrackingMode    tracking_mode = CLOSEST;
  float           tracking_huber_delta = 0.3f;
  float           tracking_error_threshold = 0.3f;
  float           tracking_dist_threshold = 2;

  /* misc */
  bool            debug = false;
  std::string     debug_dir;

  /* keyframe connection */
  ConnMode        connection_mode = ConnMode::LAST;
  std::size_t     max_back_connections = 4;

  /* keyframe adding */
  KeyframeMode    keyframe_mode = AUTO;
  float           inlier_threshold = 0.5f;
  float           dist_threshold = 2.f;
  float           frame_dist_threshold = 0.2f;
  float           combined_threshold = 2.0f;

  /* loop closure */
  bool            loop_closure = true;
  float           loop_max_dist = 0.5f;
  int             loop_active_window = 10;
  float           loop_sigma = 1.0f;
  float           loop_min_similarity = 0.35f;
  int             loop_max_candidates = 10;

  /* mapping params */
  bool            interleave_mapping = false;
  float           relinearize_skip = 1;
  float           relinearize_threshold = 0.05f;
  float           partial_relin_check = true;
  float           pose_prior = 0.3f;
  float           code_prior = 1.0f;
  bool            predict_code = true;

  /* photometric error */
  bool            use_photometric = true;
  IntVector       pho_iters = {15, 15, 15, 30};
  float           huber_delta = 0.3f;
  int             sfm_step_threads = 32;
  int             sfm_step_blocks = 11;
  int             sfm_eval_threads = 224;
  int             sfm_eval_blocks = 66;
  bool            normalize_image = false; // normalize image with: (I-mean)/std

  /* reprojection error */
  bool            use_reprojection = false;
  int             rep_nfeatures = 500;
  float           rep_scale_factor = 1.2f;
  int             rep_nlevels = 1;
  float           rep_max_dist = 30.0;
  float           rep_huber = 0.1f;
  int             rep_iters = 15;
  float           rep_sigma = 1.0f;
  int             rep_ransac_maxiters = 1000;
  float           rep_ransac_threshold = 0.0001f;

  /* sparse geometric error */
  bool            use_geometric = false;
  int             geo_npoints = 500;
  bool            geo_stochastic = false;
  float           geo_huber = 0.1f;
  int             geo_iters = 15;

  static KeyframeMode KeyframeModeTranslator(const std::string& s);
  static std::string KeyframeModeTranslator(KeyframeMode mode);
  static TrackingMode TrackingModeTranslator(const std::string& s);
  static std::string TrackingModeTranslator(TrackingMode mode);
  static ConnMode ConnectionModeTranslator(const std::string& s);
  static std::string ConnectionModeTranslator(ConnMode mode);
};

} // namespace df

#endif // DF_OPTIONS_H_
