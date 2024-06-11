
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include "Jpeg.h"

// based on
// http://soratobi96.hatenablog.com/entry/20180923/1537690266

void allocJpeg(JpegData &io_JpegData) {
    io_JpegData.data = nullptr;
    io_JpegData.data = (uint8_t *) malloc(sizeof(uint8_t) * io_JpegData.width * io_JpegData.height * io_JpegData.ch);
}

bool readJpegData(JpegData &out_JpegData, const char *in_FileName) {
    struct jpeg_decompress_struct cinfo;
    jpeg_create_decompress(&cinfo);

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    FILE *fp = fopen(in_FileName, "rb");
    if (fp == NULL) {
        printf("Error: failed to open %s\n", in_FileName);
        return false;
    }

    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    out_JpegData.width = cinfo.image_width;
    out_JpegData.height = cinfo.image_height;
    out_JpegData.ch = cinfo.num_components;

    allocJpeg(out_JpegData);

    uint8_t *row = out_JpegData.data;
    const uint32_t stride = out_JpegData.width * out_JpegData.ch;
    for (int y = 0; y < out_JpegData.height; y++) {
        jpeg_read_scanlines(&cinfo, &row, 1);
        row += stride;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return true;
}

bool
writeJpegData(const float *in_FilmBuffer, const char *out_FileName, const int width, const int height, const int ch,
              const int quality) {
    uint8_t *data = (uint8_t *) malloc(sizeof(uint8_t) * width * height * ch);
    for (int i = 0; i < width * height * ch; i++) {
        data[i] = static_cast<uint8_t>( std::round(std::max(0.0f, std::min(255.0f, in_FilmBuffer[i]))));
    }

    struct jpeg_compress_struct cinfo;
    jpeg_create_compress(&cinfo);

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    FILE *fp = fopen(out_FileName, "wb");
    if (fp == NULL) {
        printf("Error: failed to open %s\n", out_FileName);
        return false;
    }

    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = ch;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    uint8_t *row = data;
    const uint32_t stride = width * ch;
    for (int y = 0; y < height; y++) {
        jpeg_write_scanlines(&cinfo, &row, 1);
        row += stride;
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
    memset(data, 0, sizeof(uint8_t) * width * height * ch);

    return true;
}

void finalizeJpegData(JpegData &io_JpegData) {
    if (io_JpegData.data != nullptr) {
        free(io_JpegData.data);
        io_JpegData.data = nullptr;
    }
}
