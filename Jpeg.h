//
//  Jpeg.h
//  gl3d_mesh_vertex_shading_texture
//
//  Created by Yonghao Yue on 2019/10/22.
//  Copyright © 2019 Yonghao Yue. All rights reserved.
//

#ifndef Jpeg_h
#define Jpeg_h

// based on
// http://soratobi96.hatenablog.com/entry/20180923/1537690266

#include <stdint.h>
#include <jpeglib.h>

struct JpegData
{
  JpegData(): data( nullptr ), width(0), height(0), ch(0) {}
  
  uint8_t *data;
  uint32_t width;
  uint32_t height;
  uint32_t ch;
};

bool readJpegData( JpegData& out_JpegData, const char* in_FileName );
void finalizeJpegData( JpegData& io_JpegData );

#endif /* Jpeg_h */
