#pragma once

#include <string>

// MuJoCo's distributed simulate sources use one small part of the LodePNG
// API for screenshots, but the Linux binary archive does not ship LodePNG.
// Keep the compatible surface deliberately narrow: 8-bit RGB file encoding.
enum LodePNGColorType {
  LCT_GREY = 0,
  LCT_RGB = 2,
  LCT_PALETTE = 3,
  LCT_GREY_ALPHA = 4,
  LCT_RGBA = 6,
};

namespace lodepng {

unsigned encode(const std::string& filename, const unsigned char* image,
                unsigned width, unsigned height,
                LodePNGColorType color_type = LCT_RGBA,
                unsigned bit_depth = 8);

}  // namespace lodepng
