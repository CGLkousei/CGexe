//
// Created by KouseiNakayama on 2024/05/28.
//

#ifndef CGEXE_PARTICIPATINGMEDIA_H
#define CGEXE_PARTICIPATINGMEDIA_H

#include <Eigen/Dense>

struct ParticipatingMedia {
    Eigen::Vector3d pos;
    Eigen::Vector3d color;

    double radius;
    double extinction;
    double albedo;
    double hg_g;
};

#endif //CGEXE_PARTICIPATINGMEDIA_H
