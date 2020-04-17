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
#include <sstream>
#include <iomanip>

#include "logutils.h"

namespace fs = boost::filesystem;

namespace df
{

std::string GetTimeStamp(std::string format)
{
  std::time_t t = std::time(nullptr);
  auto localTime = std::localtime(&t);
  std::stringstream ss;
  ss << std::put_time(localTime, format.c_str());
  return ss.str();
}

void CreateDirIfNotExists(std::string dir)
{
  if (!fs::exists(dir))
    fs::create_directory(dir);
}

std::string CreateLogDirForRun(std::string logdir, std::string run_dir_name)
{
  // check if the logging dir exists. If not, create it
  CreateDirIfNotExists(logdir);

  // create a timestamped rundir inside log_dir or use whatever the user specified in
  // run_dir_name
  std::string dir_name = run_dir_name.empty() ? GetTimeStamp() : run_dir_name;
  std::string rundir = (fs::path(logdir) / dir_name).string();
  CreateDirIfNotExists(rundir); // shouldn't exist really

  return rundir;
}

} // namespace df
