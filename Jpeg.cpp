//
//  Jpeg.cpp
//  gl3d_mesh_vertex_shading_texture
//
//  Created by Yonghao Yue on 2019/10/22.
//  Copyright © 2019 Yonghao Yue. All rights reserved.
//

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "Jpeg.h"

// based on
// http://soratobi96.hatenablog.com/entry/20180923/1537690266

void allocJpeg( JpegData& io_JpegData )
{
  io_JpegData.data = nullptr;
  io_JpegData.data = (uint8_t*)malloc( sizeof(uint8_t) * io_JpegData.width * io_JpegData.height * io_JpegData.ch);
}

bool readJpegData( JpegData& out_JpegData, const char* in_FileName )
{
  struct jpeg_decompress_struct cinfo;
  jpeg_create_decompress( &cinfo );
  
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error( &jerr );
  
  FILE *fp = fopen( in_FileName, "rb" );
  if (fp == NULL) {
    printf( "Error: failed to open %s\n", in_FileName );
    return false;
  }

  jpeg_stdio_src( &cinfo, fp );
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);
  
  out_JpegData.width  = cinfo.image_width;
  out_JpegData.height = cinfo.image_height;
  out_JpegData.ch     = cinfo.num_components;
  
  allocJpeg( out_JpegData );
  
  uint8_t *row = out_JpegData.data;
  const uint32_t stride = out_JpegData.width * out_JpegData.ch;
  for ( int y = 0; y < out_JpegData.height; y++ )
  {
    jpeg_read_scanlines( &cinfo, &row, 1 );
    row += stride;
  }
  
  jpeg_finish_decompress( &cinfo );
  jpeg_destroy_decompress( &cinfo );
  fclose( fp );
  
  return true;
}

void finalizeJpegData( JpegData& io_JpegData )
{
  if ( io_JpegData.data != nullptr )
  {
    free( io_JpegData.data );
    io_JpegData.data = nullptr;
  }
}
