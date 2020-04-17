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
#include "image_sequences.h"

#include <fstream>

#include <json/json.h>
#include <glog/logging.h>

ImageSequence SequenceCollection::Get(const std::string& name)
{
  if (!Has(name))
    LOG(FATAL) << "Sequence " << name << " is not in the sequence map";
  return map_[name];
}

bool SequenceCollection::Has(const std::string& name)
{
  return map_.find(name) != map_.end();
}

void SequenceCollection::Add(const std::string& name, const ImageSequence& seq)
{
  map_[name] = seq;
}

ImageSequence GetImageSequence(const std::string& name,
                               const std::string& cfgpath,
                               df::PinholeCamera<float>& cam)
{
  auto col = ParseSequencesFromJson(cfgpath);
  auto seq = col.Get(name);

  // probe first image for width and height
  std::string path = seq.base_dir + "/" + seq.filenames[0];
  cv::Mat first = cv::imread(path);
  if (first.empty())
    LOG(FATAL) << "Could not load the first image in the sequence: " << path;

  // fill the camera with info
  cam = df::PinholeCamera<float>(seq.intrinsics.fx, seq.intrinsics.fy,
                                    seq.intrinsics.ux, seq.intrinsics.uy,
                                    first.cols, first.rows);
  return seq;
}

std::map<std::string,Intrinsics> ParseCameras(const Json::Value& val)
{
  CHECK(val.isObject());

  std::map<std::string,Intrinsics> cams;
  std::vector<std::string> keys = val.getMemberNames();
  for (Json::Value::ArrayIndex i = 0; i < keys.size(); ++i)
  {
    const Json::Value cam_val = val[keys[i]];

    // check if the entry is well formed
    CHECK(cam_val.isArray());
    CHECK_EQ(cam_val.size(), 4);
    for (int i = 0; i < 4; ++i)
      CHECK(cam_val[i].isDouble());

    Intrinsics tmp;
    tmp.fx = cam_val[0].asDouble();
    tmp.fy = cam_val[1].asDouble();
    tmp.ux = cam_val[2].asDouble();
    tmp.uy = cam_val[3].asDouble();
    cams[keys[i]] = tmp;
  }

  return cams;
}

ImageSequence ParseSequence(const Json::Value& val,
                            const std::map<std::string,Intrinsics>& cams)
{
  const char* base_dir_key = "base_dir";
  const char* images_key = "images";
  const char* depth_key = "depth_images";
  const char* camera_key = "camera";

  // check if entry is well formed
  CHECK_EQ(val[base_dir_key].type(), Json::stringValue);
  CHECK_EQ(val[images_key].type(), Json::arrayValue);
  CHECK_EQ(val[camera_key].type(), Json::stringValue);

  ImageSequence seq;

  /* base directory */
  seq.base_dir = val[base_dir_key].asString();

  /* camera */
  std::string camera = val[camera_key].asString();
  if (cams.find(camera) == cams.end())
    LOG(FATAL) << "Camera " << camera << " was not defined in the file";
  seq.intrinsics = cams.at(camera);

  /* images */
  Json::Value images = val[images_key];
  for (Json::Value::ArrayIndex i = 0; i < images.size(); ++i)
  {
    CHECK_EQ(images[i].type(), Json::stringValue);
    seq.filenames.push_back(images[i].asString());
  }

  /* optional gtdepth */
  if (val.isMember(depth_key))
  {
    Json::Value depth_images = val[depth_key];
    CHECK(depth_images.isArray());
    for (Json::Value::ArrayIndex i = 0; i < depth_images.size(); ++i)
    {
      CHECK(depth_images[i].isString());
      seq.depth_filenames.push_back(depth_images[i].asString());
    }
    seq.has_depth = true;
  }
  return seq;
}

SequenceCollection ParseSequencesFromJson(const std::string& path)
{
  const char* cameras_key = "cameras";
  const char* sequences_key = "sequences";

  std::ifstream ifs(path);
  if (ifs.fail())
    LOG(FATAL) << "Could not load file: " << path;

  Json::Value root;
  Json::parseFromStream(Json::CharReaderBuilder(), ifs, &root, nullptr);

  CHECK(root.isMember(cameras_key));
  CHECK(root.isMember(sequences_key));

  // first, load the cameras
  Json::Value cam_val = root["cameras"];
  CHECK(cam_val.isObject());
  auto cams = ParseCameras(cam_val);

  // parse sequences
  SequenceCollection seqcol;
  Json::Value seqs_val = root["sequences"];
  CHECK(seqs_val.isObject());
  auto seq_names = seqs_val.getMemberNames();
  for (auto& name : seq_names)
  {
    CHECK(seqs_val[name].isObject());
    seqcol.Add(name, ParseSequence(seqs_val[name], cams));
  }

  return seqcol;
}
