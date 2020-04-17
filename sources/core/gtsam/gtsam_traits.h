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
#ifndef DF_GTSAM_TRAITS_H_
#define DF_GTSAM_TRAITS_H_

#include <sophus/se3.hpp>

namespace gtsam
{

template<typename Scalar>
struct traits<Sophus::SE3<Scalar>>
{
  typedef Sophus::SE3<Scalar> SE3T;

  /** Return the dimensionality of the tangent space of this value */
  static size_t GetDimension(const SE3T& pose)
  {
    return SE3T::DoF;
  }

  static void Print(const SE3T& pose, const std::string& str)
  {
    std::cout << str << pose.log().transpose();
  }

  /** Increment the value, by mapping3 from the vector delta in the tangent
    * space of the current value back to the manifold to produce a new,
    * incremented value.
    * @param delta The delta vector in the tangent space of this value, by
    * which to increment this value.
    */
  static SE3T Retract(const SE3T& pose, const gtsam::Vector& delta)
  {
    Eigen::Matrix<Scalar,6,1> update = delta.cast<Scalar>();
    Eigen::Matrix<Scalar,3,1> trs_update = update.template head<3>();
    Eigen::Matrix<Scalar,3,1> rot_update = update.template tail<3>();

    SE3T pose_new;
    pose_new.translation() = pose.translation() + trs_update;
    pose_new.so3() = SE3T::SO3Type::exp(rot_update) * pose.so3();
    return pose_new;
  }

  /** Compute the coordinates in the tangent space of this value that
    * retract() would map to \c value.
    * @param value The value whose coordinates should be determined in the
    * tangent space of the value on which this function is called.
    * @return The coordinates of \c value in the tangent space of \c this.
    */
  static gtsam::Vector Local(const SE3T& first, const SE3T& second)
  {
    typename SE3T::Tangent tangent;
    tangent.template head<3>() = second.translation() - first.translation();
    tangent.template tail<3>() = (second.so3() * first.so3().inverse()).log();
    return tangent.template cast<double>();
  }

  /** Compare this Value with another for equality. */
  static bool Equals(const SE3T& first, const SE3T& second, Scalar tol)
  {
    return Local(first, second).norm() < tol;
  }
};

/*
 * instantiate SE3 traits for double and float
 */
template struct traits<Sophus::SE3<float>>;
template struct traits<Sophus::SE3<double>>;

} // namespace gtsam

#endif // DF_GTSAM_TRAITS_H_
