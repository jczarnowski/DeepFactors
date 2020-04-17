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
#ifndef DF_LOGUTILS_H_
#define DF_LOGUTILS_H_

#include <ctime>
#include <string>
#include <boost/filesystem.hpp>

namespace df
{

std::string GetTimeStamp(std::string format = "%y%m%d%H%M%S");
void CreateDirIfNotExists(std::string dir);
std::string CreateLogDirForRun(std::string logdir, std::string run_dir_name);

} // namespace df

#endif // DF_LOGUTILS_H_
