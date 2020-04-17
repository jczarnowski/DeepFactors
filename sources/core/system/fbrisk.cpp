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
#include <string>
#include <vector>

#include <fbrisk.h>
#include <brisk/brisk.h>

namespace DBoW2 {

void FBrisk::meanValue(const std::vector<pDescriptor>& descriptors,
                       TDescriptor& mean)
{
  mean.resize(0);
  mean.resize(L, 0);

  uint64_t s = descriptors.size() / 2;
  std::vector<uint64_t> sum(L*8, 0);

  for(auto it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const auto& desc = **it;
    for(int i = 0; i < L; i++)
    {
      for(int bit = 0; bit < 8; ++bit)
      {
        sum[i*8 + bit] += (desc[i] & (0x01 << bit)) ? 1 : 0;
      }
    }
  }

  for(int i = 0; i < L; i++)
  {
    for(int bit = 0; bit < 8; ++bit)
    {
      if(sum[i*8 + bit] > s)
        mean[i] |= (0x01 << bit);
    }
  }
}

/* ************************************************************************* */
double FBrisk::distance(const FBrisk::TDescriptor &a,
                        const FBrisk::TDescriptor &b)
{
  return (double)brisk::Hamming::PopcntofXORed(&a.front(),&b.front(),L/16);
}

/* ************************************************************************* */
std::string FBrisk::toString(const FBrisk::TDescriptor &a)
{
  std::stringstream ss;
  for(int i = 0; i < L; ++i)
    ss << int(a[i]) << " ";
  return ss.str();
}

/* ************************************************************************* */
void FBrisk::fromString(FBrisk::TDescriptor &a, const std::string &s)
{
  a.resize(L);

  std::stringstream ss(s);
  for(int i = 0; i < L; ++i)
  {
    int v;
    ss >> v;
    a[i] = v;
  }
}

/* ************************************************************************* */
void FBrisk::toMat32F(const std::vector<TDescriptor> &descriptors,
                      cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const int N = descriptors.size();
  mat.create(N, L*8, CV_32F);
  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < L*8; j += 8, p += 8)
    {
      for(int bit = 0; bit < 8; ++bit)
        *(p + bit) = (desc[j] & (0x01 << bit));
    }
  }
}

} // namespace DBoW2


