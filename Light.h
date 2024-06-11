
#ifndef Light_h
#define Light_h

#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_DONT_VECTORIZE

#include <Eigen/Dense>

struct AreaLight {
    Eigen::Vector3d pos;
    Eigen::Vector3d arm_u;
    Eigen::Vector3d arm_v;
    Eigen::Vector3d color;
    double intensity;
};


#endif /* Light_h */
