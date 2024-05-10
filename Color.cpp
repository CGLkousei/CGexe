//
// Created by a5820 on 2024/05/10.
//

#include "Color.h"

Color codeToColor(const std::string &colorCode) {
    const auto rgb = (uint32_t) strtol((const char *) &colorCode[1], nullptr, 16);

    const auto red = (rgb >> 16) & 0xFF;
    const auto green = (rgb >> 8) & 0xFF;
    const auto blue = (rgb >> 0) & 0xFF;

    return Color{(double)red / 255, (double)green / 255, (double)blue / 255};
}

double getLuminance(const Color &c) {
    return c.dot(Eigen::Vector3d(0.2126, 0.7152, 0.0722));
}

Color changeLuminance(const Color &c, const double &l_out) {
    return c * (l_out / getLuminance(c));
}