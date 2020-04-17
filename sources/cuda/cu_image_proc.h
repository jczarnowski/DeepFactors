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
#ifndef DF_CU_IMAGE_PROC_H_
#define DF_CU_IMAGE_PROC_H_

#include <VisionCore/Buffers/Image2D.hpp>
#include <Eigen/Dense>

namespace df
{

template <typename T>
void GaussianBlurDown(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& in,
                      vc::Buffer2DView<T, vc::TargetDeviceCUDA>& out);

template <typename T, typename TG>
void SobelGradients(const vc::Buffer2DView<T,vc::TargetDeviceCUDA>& img,
                    vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA>& grad);

template <typename T>
T SquaredError(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf1,
               const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf2);

template <typename T, int CS, typename ImageBuf=vc::Buffer2DView<T, vc::TargetDeviceCUDA>>
void UpdateDepth(const Eigen::Matrix<T,CS,1>& code,
                 const ImageBuf& prx_orig,
                 const ImageBuf& prx_jac,
                 T avg_dpt,
                 ImageBuf& dpt_out);

} // namespace df

#endif // DF_CU_IMAGE_PROC_H_
