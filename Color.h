//
// Created by a5820 on 2024/05/10.
//

#ifndef CGEXE_COLOR_H
#define CGEXE_COLOR_H

#include <Eigen/Dense>

using Color = Eigen::Vector3d;

double getLuminance(const Color &c);

Color changeLuminance(const Color &c, const double &l_out);

Color codeToColor(const std::string &colorCode);

#endif //CGEXE_COLOR_H
