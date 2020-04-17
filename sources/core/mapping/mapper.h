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
#ifndef DF_MAPPER_H_
#define DF_MAPPER_H_

#include <memory>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/base/Vector.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "factor_graph.h"
#include "keyframe_map.h"
#include "cu_image_proc.h"
#include "decoder_network.h"
#include "feature_detection.h"
#include "work_manager.h"
#include "df_work.h"

#include <gtsam/nonlinear/LinearContainerFactor.h>

namespace df
{

struct MapperOptions
{
  enum ConnectionMode { FULL, LASTN, FIRST, LAST };

  double                    code_prior = 1;
  double                    pose_prior = 0.3;
  bool                      predict_code = true;

  /* photometric */
  bool                      use_photometric = true;
  std::vector<int>          pho_iters;
  df::DenseSfmParams     sfm_params;
  int          	            sfm_step_threads = 32;
  int                       sfm_step_blocks = 11;
  int          	            sfm_eval_threads = 66;
  int                       sfm_eval_blocks = 224;

  /* reprojection */
  bool                      use_reprojection = true;
  int                       rep_nfeatures = 500;
  float                     rep_scale_factor = 1.2f;
  int                       rep_nlevels = 1;
  float                     rep_max_dist = 30.0f;
  float                     rep_huber = 0.1f;
  int                       rep_iters = 15;
  float                     rep_sigma = 1.0f;
  int                       rep_ransac_maxiters = 1000;
  float                     rep_ransac_threshold = 0.0001f;

  /* geometric */
  bool                      use_geometric = false;
  int                       geo_npoints = 500;
  bool                      geo_stochastic = false;
  float                     geo_huber = 0.1f;
  int                       geo_iters = 15;

  /* keyframe connections */
  ConnectionMode            connection_mode = LAST;
  int                       max_back_connections = 4;

  /* ISAM2 */
  gtsam::ISAM2Params        isam_params;
};

/*
 * Incremental mapping with ISAM2
 * Containts the keyframe map
 */
template <typename Scalar, int CS>
class Mapper
{
public:
  typedef gtsam::Vector CodeT;
  typedef Sophus::SE3<Scalar> SE3T;
  typedef df::Map<Scalar> MapT;
  typedef df::Keyframe<Scalar> Keyframe;
  typedef df::Frame<Scalar> Frame;
  typedef df::SfmAligner<Scalar,CS> Aligner;
  typedef df::SE3Aligner<Scalar> Se3Aligner;
  typedef typename Aligner::Ptr AlignerPtr;
  typedef typename Se3Aligner::Ptr Se3AlignerPtr;
  typedef typename Keyframe::Ptr KeyframePtr;
  typedef typename Keyframe::IdType KeyframeId;
  typedef typename Frame::Ptr FramePtr;
  typedef typename MapT::Ptr MapPtr;
  typedef typename MapT::FrameId FrameId;
  typedef std::function<void(MapPtr)> MapCallbackT;
  typedef std::vector<int> IterList;

  /*!
   * \brief The only constructor for this class
   * \param opts
   * \param cam_pyr
   * \param network
   */
  Mapper(const MapperOptions& opts,
         const df::CameraPyramid<Scalar>& cam_pyr,
         df::DecoderNetwork::Ptr network);

  /*!
   * \brief Add a new one-way frame to the current keyframe
   * \param img
   * \param col
   * \param pose_init
   * \param kf_id
   */
  void EnqueueFrame(const cv::Mat& img, const cv::Mat& col, const SE3T& pose_init,
                    const Features& ft, FrameId kf_id);

  /*!
   * \brief Add a keyframe and connect it to previous keyframes
   * \param img
   * \param col
   * \param pose_init
   */
  KeyframePtr EnqueueKeyframe(double timestamp, const cv::Mat& img, const cv::Mat& col,
                              const SE3T& pose_init, const Features& ft);

  /*!
   * \brief Add a keyframe with predefined connections to other existing keyframes
   * \param img
   * \param col
   * \param pose_init
   * \param conns
   */
  KeyframePtr EnqueueKeyframe(double timestamp, const cv::Mat& img, const cv::Mat& col, const SE3T& pose_init,
                              const Features& ft, const std::vector<FrameId>& conns);

  void EnqueueLink(FrameId id0, FrameId id, Scalar rep_sigma, bool pho = true, bool rep = true, bool geo = false);

  /*!
   * \brief Marginalize frames that are connected to keyframes
   * \param frames
   */
  void MarginalizeFrames(const std::vector<FrameId>& frames);

  /*!
   * \brief Perform a single mapping step
   */
  void MappingStep();

  /*!
   * \brief Initialize the mapper with two images
   */
  void InitTwoFrames(const cv::Mat& img0, const cv::Mat& img1,
                     const cv::Mat& col0, const cv::Mat& col1,
                     const Features& ft0, const Features& ft1,
                     double ts1, double ts2);

  /*!
   * \brief Initialize the mapper with a single image
   * Relies on the network prediction (decoding a zero code)
   */
  void InitOneFrame(double timestamp, const cv::Mat& img, const cv::Mat& col,
                    const Features& ft);

  /*!
   * \brief Reset the mapper
   */
  void Reset();

  /*!
   * \brief Saves current factor graph and isam2 bayes tree as .dot files
   * \param Filepath + filename prefix e.g. ~/s
   */
  void SaveGraphs(const std::string& prefix);

  /*!
   * \brief Prints all factors in the graph to stdout
   */
  void PrintFactors();

  /*!
   * \brief Prints debug info about the state of the graph
   */
  void PrintDebugInfo();

  /*!
   * \brief Display sparse keypoint matches between keyframes (if any)
   */
  void DisplayMatches();

  /*!
   * \brief Display loop closure reprojection errors
   */
  void DisplayLoopClosures(std::vector<std::pair<int, int>>& link);

  cv::Mat DisplayReprojectionErrors();
  cv::Mat DisplayPhotometricErrors();

  void SavePhotometricDebug(std::string save_dir);
  void SaveReprojectionDebug(std::string save_dir);
  void SaveGeometricDebug(std::string save_dir);

  /*!
   * \brief Set new mapper options
   */
  void SetOptions(const MapperOptions& new_opts);

  std::vector<FrameId> NonmarginalizedFrames();
  std::map<KeyframeId, bool> KeyframeRelinearization();

  /*
   * Setters/getters
   */
  void SetMapCallback(MapCallbackT cb) { map_callback_ = cb; }
  MapPtr GetMap() { return map_; }
  std::size_t NumKeyframes() const { return map_->NumKeyframes(); }
  bool HasWork() const { return !work_manager_.Empty(); }

private:
  /*!
   * \brief Track all changes to the optimization process
   * Determines new factors to be added to or remove from the graph
   * (e.g. when progressing to a new level in coarse to fine)
   *
   * \param New factors to be added to the graph
   * \param Indices of factors to be removed (in isam2 graph)
   * \param New variable initialization (if any)
   */
  void Bookkeeping(gtsam::NonlinearFactorGraph& new_factors,
                   gtsam::FactorIndices& remove_indices,
                   gtsam::Values& var_init);

  /*!
   * \brief Construct a new frame
   * \return Pointer to the new frame
   */
  FramePtr BuildFrame(const cv::Mat& img, const cv::Mat& col,
                      const SE3T& pose_init, const Features& ft);

  /*!
   * \brief Constructs a new keyframe
   */
  KeyframePtr BuildKeyframe(double timestamp, const cv::Mat& img, const cv::Mat& col,
                            const SE3T& pose_init, const Features& ft);

  /*!
   * \brief Create a list of backward connections of a keyframe based on the mapper options
   */
  std::vector<FrameId> BuildBackConnections();

  /*!
   * \brief Update the keyframe map with new isam2 estimate
   * \param Estimated values for poses and codes of keyframes
   */
  void UpdateMap(const gtsam::Values& vals, const gtsam::VectorValues& delta);

  /*!
   * \brief Call all callbacks to inform about new map estimate
   */
  void NotifyMapObservers();

private:
  gtsam::Values estimate_;
  MapPtr map_;

  std::unique_ptr<gtsam::ISAM2> isam_graph_;
  std::shared_ptr<df::DecoderNetwork> network_;
  df::CameraPyramid<Scalar> cam_pyr_;
  Se3AlignerPtr se3aligner_;
  AlignerPtr aligner_;

  // mapping parameters
  MapperOptions opts_;
  MapCallbackT map_callback_;

  // displaying matches
  std::vector<cv::Mat> match_imgs_;
  bool new_match_imgs_ = false;

  work::WorkManager work_manager_;

  // last result of the ISAM2 algorithm step
  gtsam::ISAM2Result isam_res_;
};

} // namespace df

#endif // DF_MAPPER_H_
