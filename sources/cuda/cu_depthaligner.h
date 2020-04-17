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
#ifndef DF_CU_DEPTHALIGNER_H_
#define DF_CU_DEPTHALIGNER_H_

#include <Eigen/Core>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>

#include "reduction_items.h"

namespace df
{

// forward declarations
template <typename Scalar>
class PinholeCamera;

template <typename Scalar, int CS>
class DepthAligner
{
public:
  typedef std::shared_ptr<DepthAligner<Scalar, CS>> Ptr;
  typedef Eigen::Matrix<Scalar,CS,1> CodeT;
  typedef vc::Image2DView<Scalar,vc::TargetDeviceCUDA> ImageBuffer;
  typedef JTJJrReductionItem<Scalar, CS> ReductionItem;

  DepthAligner();
  virtual ~DepthAligner();

  ReductionItem RunStep(const CodeT& code,
                        const ImageBuffer& target_dpt,
                        const ImageBuffer& prx_orig,
                        const ImageBuffer& prx_jac);

private:
  vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> bscratch_;
};

} // namespace df

#endif // DF_CU_DEPTHALIGNER_H_
