//
// Created by a5820 on 2024/05/10.
//

#include "Image.h"
#include <iostream>
#include <fstream>

Image::Image(const int &width, const int &height) : width(width), height(height) {
    pixels = new Color[width * height];
    for(int i=0;i<width * height;i++){
        pixels[i] = Eigen::Vector3d::Zero();
    }
}
Image::Image(const int &width, const int &height, const GLubyte* buffer) : width(width), height(height) {
    pixels = new Color[width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int index_p = y * width + x;
            const int index_b = (height - y - 1) * width + x;
            pixels[index_p].x() = buffer[index_b * 3];
            pixels[index_p].y() = buffer[index_b * 3 + 1];
            pixels[index_p].z() = buffer[index_b * 3 + 2];
        }
    }
}

Image::~Image() {
    delete[] pixels;
}

void Image::save(const std::string &fname) const {
    const auto mat = toCvMat8UC3();

    /// 画像を保存します
    bool success = cv::imwrite(fname + ".png", mat);

    /// 保存に成功したかどうかを確認する
    if (success) std::cout << "save a image is success." << std::endl;
    else std::cerr << "save a image is failed." << std::endl;
}

void Image::generateCSV(const std::string &fname) const {
    std::ofstream ofs(fname + ".csv");

    if(ofs.is_open()){
        ofs << width << "," << height << std::endl;
        for(int i = 0; i< width * height; i++){
            ofs << pixels[i].x() << "," << pixels[i].y() << "," << pixels[i].z() << std::endl;
        }

        ofs.close();
    }
}

Image Image::apply_gamma_correction() const {
    Image img(width, height);

    for(int i = 0; i < width * height; i++) {
        for(int j = 0; j < 3; j++) {
            if(pixels[i][j] <= 0.0031308) img.pixels[i][j] = pixels[i][j] * 12.92;
            else img.pixels[i][j] = pow(pixels[i][j], 1.0 / 2.4) * 1.055 - 0.055;
        }
    }

    return img;
}

void Image::reinhard_tone_mapping() const {
    /**
     * @ref https://64.github.io/tonemapping
     */
    for(int i = 0; i < width * height; i++) {
        for(int j = 0; j < 3; j++) {
            pixels[i][j] = pixels[i][j] / (pixels[i][j] + 1.0);
        }
    }
}

Image Image::apply_reinhard_extended_tone_mapping() const {
    Image img(width, height);

    for(int i = 0; i < width * height; i++) {
        img.pixels[i] = pixels[i] / (1.0 + getLuminance(pixels[i]));
    }

    return img;
}

double Image::getMaxLuminance() const {
    double maxLuminance = 0.0;
    for(int i = 0; i < width * height; i++) {
        const auto luminance = getLuminance(pixels[i]);
        if(luminance > maxLuminance) maxLuminance = luminance;
    }
    return maxLuminance;
}

cv::Mat Image::toCvMat() const {
    cv::Mat mat(height, width, CV_64FC3);  // 64-bit float, 3 channels

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            auto& pixel = mat.at<cv::Vec3d>(i, j);
            Eigen::Vector3d& imgPixel = pixels[i * width + j];

            pixel[0] = imgPixel[2] / 255.0;  // B
            pixel[1] = imgPixel[1] / 255.0;  // G
            pixel[2] = imgPixel[0] / 255.0;  // R

        }
    }

    return mat;
}

cv::Mat Image::toCvMat8UC3() const {
    auto mat = toCvMat();
    cv::Mat mat8uc3;
    mat.convertTo(mat8uc3, CV_8UC3, 255);
    return mat8uc3;
}

Image Image::cvMatToImage(const cv::Mat &mat) {
    const auto width = mat.cols;
    const auto height = mat.rows;

    auto image = Image(width, height);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            const auto p_idx = i * width + j;
            auto &vec = mat.at<cv::Vec3f>(i, j);
            image.pixels[p_idx] = Color(vec[2], vec[1], vec[0]);
        }
    }

    return image;
}

Image Image::cvMat8U3CToImage(const cv::Mat &mat) {
    const auto width = mat.cols;
    const auto height = mat.rows;

    auto image = Image(width, height);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            const auto p_idx = i * width + j;
            auto &vec = mat.at<cv::Vec3b>(i, j);
            image.pixels[p_idx] = Color(vec[2] / 255.0, vec[1] / 255.0, vec[0] / 255.0);
        }
    }

    return image;
}

Image Image::loadImage(const std::string &filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);

    if(img.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        // エラーハンドリングを行います。
    }

    // OpenCVでは画像はBGR順に格納されているため、RGB順に変換します。
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Imageオブジェクトを作成します。
    Image image(img.cols, img.rows);

    // OpenCVの画像データをImageオブジェクトにコピーします。
    for (int i = 0; i < image.height; i++) {
        for (int j = 0; j < image.width; j++) {
            cv::Vec3b color = img.at<cv::Vec3b>(i, j);
            image.pixels[i * image.width + j] = Eigen::Vector3d(color[0], color[1], color[2]) / 255.0;
        }
    }

    return image;
}

void Image::show(const std::string &name) const {
    const auto mat = toCvMat();
    cv::imshow(name, mat);
    cv::waitKey(0);
}

Image Image::apply_gaussian_blur(const unsigned int &k_width, const unsigned int &k_height, const double &sigma) const {
    auto mat = toCvMat8UC3();
    // ガウシアンフィルタを適用
    cv::Mat blurred;
    cv::Size kernelSize(k_width, k_height);  // カーネルサイズを指定します
    cv::GaussianBlur(mat, blurred, kernelSize, sigma);

    return cvMat8U3CToImage(blurred);
}

Image Image::apply_bilateral_filter(const unsigned int &d, const unsigned int &sigma_color, const unsigned int &sigma_space) const {
    auto mat = toCvMat8UC3();
    cv::Mat bilateral;
    cv::bilateralFilter(mat, bilateral, d, sigma_color, sigma_space);

    return cvMat8U3CToImage(bilateral);
}

Image Image::resize(const double &scaleRatio) const {
    // 出力画像
    cv::Mat img;
    // 画像リサイズ
    cv::resize(toCvMat8UC3(), img, cv::Size(), scaleRatio, scaleRatio, cv::INTER_LANCZOS4);
    return cvMat8U3CToImage(img);
}