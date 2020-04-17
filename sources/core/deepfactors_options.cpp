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
#include "deepfactors_options.h"
#include "glog/logging.h"

namespace df
{

DeepFactorsOptions::KeyframeMode DeepFactorsOptions::KeyframeModeTranslator(const std::string& s)
{
  if (s == "AUTO")
    return KeyframeMode::AUTO;
  else if (s == "NEVER")
    return KeyframeMode::NEVER;
  else if (s == "AUTO_COMBINED")
    return KeyframeMode::AUTO_COMBINED;
  else
    LOG(FATAL) << "Unknown KeyframeMode " << s;
}

std::string DeepFactorsOptions::KeyframeModeTranslator(KeyframeMode mode)
{
  std::string name;
  switch (mode)
  {
  case KeyframeMode::AUTO:
    name = "AUTO";
    break;
  case KeyframeMode::NEVER:
    name = "NEVER";
    break;
  case KeyframeMode::AUTO_COMBINED:
    name = "AUTO_COMBINED";
    break;
  default:
    name = "UNKNOWN";
    break;
  }
  return name;
}

DeepFactorsOptions::TrackingMode DeepFactorsOptions::TrackingModeTranslator(const std::string& s)
{
  if (s == "FIRST")
    return TrackingMode::FIRST;
  else if (s == "LAST")
    return TrackingMode::LAST;
  else if (s == "CLOSEST")
    return TrackingMode::CLOSEST;
  else
    LOG(FATAL) << "Invalid tracking mode: " << s;
}

std::string DeepFactorsOptions::TrackingModeTranslator(TrackingMode mode)
{
  std::string name;
  switch (mode)
  {
  case TrackingMode::FIRST:
    name = "FIRST";
    break;
  case TrackingMode::LAST:
    name = "LAST";
    break;
  case TrackingMode::CLOSEST:
    name = "CLOSEST";
    break;
  default:
    name = "UNKNOWN";
    break;
  }
  return name;
}

DeepFactorsOptions::ConnMode DeepFactorsOptions::ConnectionModeTranslator(const std::string& s)
{
  if (s == "FIRST")
    return ConnMode::FIRST;
  else if (s == "LAST")
    return ConnMode::LAST;
  else if (s == "LASTN")
    return ConnMode::LASTN;
  else if (s == "FULL")
    return ConnMode::FULL;
  else
    LOG(FATAL) << "Invalid connection mode: " << s;
}

std::string DeepFactorsOptions::ConnectionModeTranslator(DeepFactorsOptions::ConnMode mode)
{
  std::string name;
  switch (mode)
  {
  case ConnMode::FIRST:
    name = "FIRST";
    break;
  case ConnMode::LAST:
    name = "LAST";
    break;
  case ConnMode::LASTN:
    name = "LASTN";
    break;
  case ConnMode::FULL:
    name = "FULL";
    break;
  default:
    name = "UNKNOWN";
    break;
  }
  return name;
}

} // namespace df
