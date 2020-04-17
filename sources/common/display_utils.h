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
#ifndef DF_DISPLAY_UTILS_H_
#define DF_DISPLAY_UTILS_H_

#include <opencv2/opencv.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>

cv::Mat apply_colormap(cv::Mat mat, cv::ColormapTypes cmap = cv::COLORMAP_JET);

cv::Mat GetOpenCV(const vc::Buffer2DView<float, vc::TargetHost>& buf);

cv::Mat prx_to_color(vc::Buffer2DView<float, vc::TargetHost> prxbuf, cv::ColormapTypes cmap=cv::COLORMAP_JET);

cv::Mat CreateMosaic(const std::vector<cv::Mat>& array, int rows, int cols);

cv::Mat CreateMosaic(const std::vector<cv::Mat>& array);

#endif // DF_DISPLAY_UTILS_H_
