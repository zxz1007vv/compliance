#include "lodepng.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

namespace {

void append_u32(std::vector<unsigned char>& output, std::uint32_t value) {
  output.push_back(static_cast<unsigned char>(value >> 24));
  output.push_back(static_cast<unsigned char>(value >> 16));
  output.push_back(static_cast<unsigned char>(value >> 8));
  output.push_back(static_cast<unsigned char>(value));
}

std::uint32_t crc32(const unsigned char* data, std::size_t size) {
  std::uint32_t crc = 0xffffffffU;
  for (std::size_t index = 0; index < size; ++index) {
    crc ^= data[index];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
  }
  return ~crc;
}

void append_chunk(std::vector<unsigned char>& png, const char type[4],
                  const std::vector<unsigned char>& data) {
  append_u32(png, static_cast<std::uint32_t>(data.size()));
  const std::size_t crc_start = png.size();
  png.insert(png.end(), type, type + 4);
  png.insert(png.end(), data.begin(), data.end());
  append_u32(png, crc32(png.data() + crc_start, png.size() - crc_start));
}

std::uint32_t adler32(const std::vector<unsigned char>& data) {
  constexpr std::uint32_t modulus = 65521;
  std::uint32_t a = 1;
  std::uint32_t b = 0;
  for (unsigned char value : data) {
    a = (a + value) % modulus;
    b = (b + a) % modulus;
  }
  return (b << 16) | a;
}

std::vector<unsigned char> store_deflate(const std::vector<unsigned char>& raw) {
  std::vector<unsigned char> stream;
  stream.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
  // zlib header: deflate, 32 KiB window, fastest/no compression.
  stream.push_back(0x78);
  stream.push_back(0x01);
  std::size_t offset = 0;
  do {
    const std::size_t count = std::min<std::size_t>(65535, raw.size() - offset);
    const bool final = offset + count == raw.size();
    stream.push_back(final ? 0x01 : 0x00);  // byte-aligned stored DEFLATE block
    const auto length = static_cast<std::uint16_t>(count);
    const auto inverse = static_cast<std::uint16_t>(~length);
    stream.push_back(static_cast<unsigned char>(length));
    stream.push_back(static_cast<unsigned char>(length >> 8));
    stream.push_back(static_cast<unsigned char>(inverse));
    stream.push_back(static_cast<unsigned char>(inverse >> 8));
    stream.insert(stream.end(), raw.begin() + offset, raw.begin() + offset + count);
    offset += count;
  } while (offset < raw.size());
  append_u32(stream, adler32(raw));
  return stream;
}

}  // namespace

namespace lodepng {

unsigned encode(const std::string& filename, const unsigned char* image,
                unsigned width, unsigned height, LodePNGColorType color_type,
                unsigned bit_depth) {
  if (!image || !width || !height || color_type != LCT_RGB || bit_depth != 8) return 1;
  const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
  if (row_bytes / 3 != width || height >
      (std::numeric_limits<std::size_t>::max() / (row_bytes + 1))) return 1;

  std::vector<unsigned char> raw;
  raw.reserve((row_bytes + 1) * height);
  for (unsigned row = 0; row < height; ++row) {
    raw.push_back(0);  // PNG filter: None
    const unsigned char* begin = image + static_cast<std::size_t>(row) * row_bytes;
    raw.insert(raw.end(), begin, begin + row_bytes);
  }

  std::vector<unsigned char> png{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<unsigned char> header;
  append_u32(header, width);
  append_u32(header, height);
  header.insert(header.end(), {8, 2, 0, 0, 0});
  append_chunk(png, "IHDR", header);
  append_chunk(png, "IDAT", store_deflate(raw));
  append_chunk(png, "IEND", {});

  std::ofstream output(filename, std::ios::binary);
  if (!output) return 2;
  output.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
  return output ? 0 : 2;
}

}  // namespace lodepng
