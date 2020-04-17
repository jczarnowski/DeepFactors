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
#ifndef DF_NEAREST_PSD_H_
#define DF_NEAREST_PSD_H_

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <gtsam/base/cholesky.h>

template <typename T>
bool IsPsd(const Eigen::MatrixBase<T>& M)
{
//  size_t maxrank;
//  bool success;
//  gtsam::Matrix mat = M.template cast<double>();
//  boost::tie(maxrank, success) = gtsam::choleskyCareful(mat);
//  return success;

//  Eigen::LLT<T> llt(M);
//  if (llt.info() == Eigen::NumericalIssue)
//    return false;
//  return true;

  Eigen::LDLT<T> ldlt(M);
  return ldlt.isPositive();
}

/*
 * Find nearest positive semi-definite matrix to M
 *
 * [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
 *
 * [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
 * matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
 *
 */
template <typename T>
T NearestPsd(const Eigen::MatrixBase<T>& M)
{
  typedef typename T::value_type Scalar;

  T B = (M + M.transpose()) / 2;

  Eigen::JacobiSVD<T> svd(B, Eigen::ComputeThinV);
  T H = svd.matrixV().transpose() * svd.singularValues().asDiagonal() * svd.matrixV();

  T A2 = (B + H) / 2;
  T A3 = (A2 + A2.transpose()) / 2;

  int k = 1;
  Scalar spacing = 1e-15;
  T I = T::Identity(M.rows(), M.cols());
  while (!IsPsd(A3))
  {
    Eigen::SelfAdjointEigenSolver<T> es(A3);
    Scalar min_eig = es.eigenvalues().minCoeff();
    A3 += I * (-min_eig * k + spacing);
    k *= 2;
  }
  std::cout << "had to do " << log2(k) << " iterations" << std::endl;
  return A3;
}

// correct JtJ to be PSD by zeroing out negative eigenvalues
template <typename T>
T CorrectPSD(const Eigen::MatrixBase<T>& M)
{
  typedef Eigen::SelfAdjointEigenSolver<T> Solver;
  typedef typename Solver::EigenvectorsType EigenVecs;
  typedef typename Solver::RealVectorType RealVec;

  Solver es;
  es.compute(M);

  RealVec D = es.eigenvalues();
  EigenVecs V = es.eigenvectors();

  D = (D.array() < 0).select(0, D); // zero out negative elems
  return V * D.asDiagonal() * V.inverse(); // reconstruct matrix
}

#endif // DF_NEAREST_PSD_H_
