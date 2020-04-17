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
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "live_demo.h"
#include "logutils.h"
#include "camera_interface_factory.h"

/* demo options */
std::string source_url_help = "Image source URL" + df::drivers::CameraInterfaceFactory::Get()->GetUrlHelp();
DEFINE_string(source_url, "flycap://0", source_url_help.c_str());
DEFINE_string(calib_path, "data/flea.yml", "Path to OpenCV yaml file containing camera calibration");
DEFINE_string(network_path, "data/frozen_graph.pb", "Path to protobuf file containing network graph with weights");
DEFINE_string(vocab_path, "data/ORBvoc.yml.gz", "Path to ORB vocabulary for DBoW2");
DEFINE_uint32(gpu, 0, "Which gpu to use for SLAM");
DEFINE_bool(init_on_start, false, "Initialize the system on the first captured image");
DEFINE_bool(quit_on_finish, false, "Close the program after finishing all frames");
DEFINE_string(init_type, "ONEFRAME", "How to initialize the slam system (ONEFRAME, TWOFRAME)");
DEFINE_string(record_input, "", "Path where to save input images");
DEFINE_bool(pause_step, false, "Pause after each frame");
DEFINE_string(run_log_dir, "", "Directory where the run logs will be saved. The runs are stored in separate dirs named using current timestamp. \n"
                           "If not specified the logs will not be saved. This directory will be created if it doesn't exist.");
DEFINE_string(run_dir_name, "", "Force a specific run directory name. If empty, a timestamp is used");
DEFINE_bool(enable_timing, false, "Enable profiling of certain parts of the algorithm");
DEFINE_uint32(frame_limit, 0, "Limit processing to first <frame_limit> frames");
DEFINE_uint32(skip_frames, 0, "Skip first <skip_frames> frames");

/* SLAM options */
DEFINE_string(keyframe_mode, "AUTO", "New keyframe initialization criteria (AUTO, NEVER)");
DEFINE_string(connection_mode, "FULL", "How new keyframes will be connected to the others (FULL, LASTN, FIRST, LAST)");
DEFINE_string(tracking_mode, "FIRST", "How to select which keyframe to track against (CLOSEST, LAST, FIRST)");
DEFINE_bool(interleave_mapping, true, "Interleave tracking with single mapping steps");
DEFINE_double(inlier_threshold, 0.5, "Inlier threshold used to initialize new keyframes");
DEFINE_double(dist_threshold, 2, "Distance threshold used to initialize new keyframes (used both for pose and distance)");
DEFINE_double(frame_dist_threshold, 0.2, "Distance threshold used to initialize new keyframes (used both for pose and distance)");
DEFINE_double(max_back_connections, 4, "How far to connect back new keyframes");
DEFINE_bool(debug, false, "Display debug images in SLAM");
DEFINE_bool(demo_mode, false, "Hide GUI elements and show only reconstruction");

/* loop closure */
DEFINE_bool(loop_closure, true, "Enable loop closure");
DEFINE_uint32(loop_active_window, 10, "Active window for local/global loop detection");
DEFINE_double(loop_sigma, 1, "Noise stddev of the reprojection factors added for loop closure");
DEFINE_double(loop_max_dist, 0.5f, "Maximum distance to pontential loop closure candidates");
DEFINE_double(loop_min_similarity, 0.4, "Minimum simlarity for loop closure candidates");
DEFINE_uint32(loop_max_candidates, 20, "Maximum number of candidates to get from DBoW2");

/* tracking */
DEFINE_string(tracking_iters, "3,3,5,10", "Comma separated list of number of iterations per pyramid level (tracking)" \
                                          "(e.g. 3,4,5,6, the first being performed on the finest level of detail)");
DEFINE_double(tracking_huber_delta, 0.1, "Huber norm delta used in tracking");
DEFINE_double(tracking_error_threshold, 0.3, "When to consider tracking as lost");
DEFINE_double(tracking_dist_threshold, 2, "When to consider tracking as lost");

/* mapping */
DEFINE_double(pose_prior, 0.1, "Noise of the prior on the first pose");
DEFINE_double(code_prior, 1, "Noise of the prior on the codes");
DEFINE_double(relinearize_skip, 1, "ISAM2 relinarize skip");
DEFINE_double(relinearize_threshold, 0.05, "ISAM2 relinearize threshold");
DEFINE_bool(partial_relin_check, true, "ISAM2 partial relinarization check");
DEFINE_double(huber_delta, 0.1, "Huber norm delta used in mapping");
DEFINE_bool(predict_code, true, "Initialize keyframes with a predicted code");

/* photometric error */
DEFINE_bool(use_photometric, true, "Use photometric error in keyframe-keyframe links");
DEFINE_string(pho_iters, "15,15,15,30", "Comma separated list of number of iterations per pyramid level (mapping)" \
                                        "(e.g. 3,4,5,6, the first being performed on the finest level of detail)");
DEFINE_uint32(sfm_step_blocks, 11, "Number of blocks to use in mapping run optim. step (must be less than 1024)");
DEFINE_uint32(sfm_step_threads, 32, "Number of threads per block to use in mapping to run optim. step (must be a multiple of 32)");
DEFINE_uint32(sfm_eval_blocks, 66, "Number of blocks to use in mapping to evaluate error(must be less than 1024)");
DEFINE_uint32(sfm_eval_threads, 224, "Number of threads per block to use in mapping to evaluate error (must be a multiple of 32)");
DEFINE_bool(normalize_image, false, "Normalize the image before feeding it to the algorithm (subtract mean and divide by stddev)");

/* geometric error */
DEFINE_bool(use_geometric, false, "Use sparse geometric error for mapping");
DEFINE_uint32(geo_npoints, 500, "Number of sparse random points to optimize in geometric factors");
DEFINE_bool(geo_stochastic, false, "Random sparse pattern at every linearization for geometric error");
DEFINE_double(geo_huber, 0.1, "Huber norm used in geometric error");
DEFINE_uint32(geo_iters, 15, "Number of iterations for geometric error");

/* reprojection error */
DEFINE_bool(use_reprojection, false, "Use sparse reprojection error for mapping");
DEFINE_uint32(rep_nfeatures, 500, "Number of keypoints to detect in a keyframe");
DEFINE_double(rep_scale_factor, 1.2, "ORB's scale factor");
DEFINE_uint32(rep_nlevels, 1, "ORB's number of levels");
DEFINE_double(rep_max_dist, 30.0, "Max distance between feature descriptors in a match");
DEFINE_double(rep_huber, 0.1, "Huber norm used in reprojection error factors");
DEFINE_uint32(rep_iters, 15, "Number of iterations for reprojection error");
DEFINE_double(rep_sigma, 1, "Noise stddev of the reprojection factors added to the graph");
DEFINE_double(rep_ransac_maxiters, 1000, "Maximum iterations for the RANSAC outlier rejection");
DEFINE_double(rep_ransac_threshold, 0.00001, "Threshold for detecting outliers in RANSAC outlier rejection");

/* visualization */
DEFINE_string(vis_mode, "ALL", "Which keyframes to draw (FIRST, LAST, ALL)");

std::vector<int> split(const std::string &s, char delim)
{
  std::stringstream ss(s);
  std::string item;
  std::vector<int> elems;
  while (std::getline(ss, item, delim))
    elems.push_back(std::stoi(item));
  return elems;
}

int main(int argc, char* argv[])
{
  // parse command line flags
  google::SetUsageMessage("N-SLAM Live Demo");
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create a logging directory for this run
  // point glog to log there
  // save command line flags
  std::string run_dir;
  if (!FLAGS_run_log_dir.empty())
  {
    run_dir = df::CreateLogDirForRun(FLAGS_run_log_dir, FLAGS_run_dir_name);

    // save flags
    google::AppendFlagsIntoFile(run_dir + "/df.flags", nullptr);
  }

  // Init google logging
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  /* demo options */
  df::LiveDemoOptions opts;
  opts.source_url = FLAGS_source_url;
  opts.calib_path = FLAGS_calib_path;
  opts.df_opts.network_path = FLAGS_network_path;
  opts.df_opts.vocabulary_path = FLAGS_vocab_path;
  opts.df_opts.gpu = FLAGS_gpu;
  opts.init_on_start = FLAGS_init_on_start;
  opts.quit_on_finish = FLAGS_quit_on_finish;
  opts.init_type = df::LiveDemoOptions::InitTypeTranslator(FLAGS_init_type);
  opts.record_input = !FLAGS_record_input.empty();
  opts.record_path = FLAGS_record_input;
  opts.pause_step = FLAGS_pause_step;
  opts.log_dir = run_dir;
  opts.enable_timing = FLAGS_enable_timing;
  opts.frame_limit = FLAGS_frame_limit;
  opts.skip_frames = FLAGS_skip_frames;
  opts.demo_mode = FLAGS_demo_mode;

  /* SLAM options */
  opts.df_opts.keyframe_mode = df::DeepFactorsOptions::KeyframeModeTranslator(FLAGS_keyframe_mode);
  opts.df_opts.connection_mode = df::DeepFactorsOptions::ConnectionModeTranslator(FLAGS_connection_mode);
  opts.df_opts.tracking_mode = df::DeepFactorsOptions::TrackingModeTranslator(FLAGS_tracking_mode);
  opts.df_opts.interleave_mapping = FLAGS_interleave_mapping;
  opts.df_opts.inlier_threshold = FLAGS_inlier_threshold;
  opts.df_opts.dist_threshold = FLAGS_dist_threshold;
  opts.df_opts.frame_dist_threshold = FLAGS_frame_dist_threshold;
  opts.df_opts.max_back_connections = FLAGS_max_back_connections;
  opts.df_opts.debug = FLAGS_debug;

  /* tracking */
  opts.df_opts.tracking_iterations = split(FLAGS_tracking_iters, ',');
  opts.df_opts.tracking_huber_delta = FLAGS_tracking_huber_delta;
  opts.df_opts.tracking_error_threshold = FLAGS_tracking_error_threshold;
  opts.df_opts.tracking_dist_threshold = FLAGS_tracking_dist_threshold;

  /* loop closure */
  opts.df_opts.loop_closure = FLAGS_loop_closure;
  opts.df_opts.loop_active_window = FLAGS_loop_active_window;
  opts.df_opts.loop_sigma = FLAGS_loop_sigma;
  opts.df_opts.loop_max_dist = FLAGS_loop_max_dist;
  opts.df_opts.loop_min_similarity = FLAGS_loop_min_similarity;
  opts.df_opts.loop_max_candidates = FLAGS_loop_max_candidates;

  /* mapping */
  opts.df_opts.pose_prior = FLAGS_pose_prior;
  opts.df_opts.code_prior = FLAGS_code_prior;
  opts.df_opts.relinearize_skip = FLAGS_relinearize_skip;
  opts.df_opts.relinearize_threshold = FLAGS_relinearize_threshold;
  opts.df_opts.partial_relin_check = FLAGS_partial_relin_check;
  opts.df_opts.huber_delta = FLAGS_huber_delta;
  opts.df_opts.predict_code = FLAGS_predict_code;

  /* photometric error */
  opts.df_opts.use_photometric = FLAGS_use_photometric;
  opts.df_opts.pho_iters = split(FLAGS_pho_iters, ',');
  opts.df_opts.sfm_step_blocks = FLAGS_sfm_step_blocks;
  opts.df_opts.sfm_step_threads = FLAGS_sfm_step_threads;
  opts.df_opts.sfm_eval_blocks = FLAGS_sfm_eval_blocks;
  opts.df_opts.sfm_eval_threads = FLAGS_sfm_eval_threads;
  opts.df_opts.normalize_image = FLAGS_normalize_image;

  /* geometric error */
  opts.df_opts.use_geometric = FLAGS_use_geometric;
  opts.df_opts.geo_npoints = FLAGS_geo_npoints;
  opts.df_opts.geo_stochastic = FLAGS_geo_stochastic;
  opts.df_opts.geo_huber = FLAGS_geo_huber;
  opts.df_opts.geo_iters = FLAGS_geo_iters;

  /* reprojection error */
  opts.df_opts.use_reprojection = FLAGS_use_reprojection;
  opts.df_opts.rep_nfeatures = FLAGS_rep_nfeatures;
  opts.df_opts.rep_scale_factor = FLAGS_rep_scale_factor;
  opts.df_opts.rep_nlevels = FLAGS_rep_nlevels;
  opts.df_opts.rep_max_dist = FLAGS_rep_max_dist;
  opts.df_opts.rep_huber = FLAGS_rep_huber;
  opts.df_opts.rep_iters = FLAGS_rep_iters;
  opts.df_opts.rep_sigma = FLAGS_rep_sigma;
  opts.df_opts.rep_ransac_maxiters = FLAGS_rep_ransac_maxiters;
  opts.df_opts.rep_ransac_threshold = FLAGS_rep_ransac_threshold;

  /* visualization */
  opts.vis_mode = df::VisualizerConfig::VisualizeModeTranslator(FLAGS_vis_mode);

  // create and run the live demo
  df::LiveDemo<DF_CODE_SIZE> demo(opts);
  demo.Run();

  // cleanup
  google::ShutdownGoogleLogging();
  gflags::ShutDownCommandLineFlags();

  return 0;
}
