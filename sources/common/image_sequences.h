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
#ifndef DF_IMAGE_SEQUENCES_H_
#define DF_IMAGE_SEQUENCES_H_

#include <string>
#include <vector>
#include <map>

#include "pinhole_camera.h"

struct Intrinsics
{
  float fx;
  float fy;
  float ux;
  float uy;
};

struct ImageSequence
{
  std::string base_dir;
  std::vector<std::string> filenames;
  std::vector<std::string> depth_filenames;
  Intrinsics intrinsics;
  bool has_depth = false;
};

class SequenceCollection
{
public:
  ImageSequence Get(const std::string& name);
  bool Has(const std::string& name);
  void Add(const std::string& name, const ImageSequence& seq);

private:
  std::map<std::string, ImageSequence> map_;
};

ImageSequence GetImageSequence(const std::string& name,
                               const std::string& cfgpath,
                               df::PinholeCamera<float>& cam);


SequenceCollection ParseSequencesFromJson(const std::string& path);


#endif //  DF_IMAGE_SEQUENCES_H_
