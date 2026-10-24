#pragma once

// RT diagnostic capture: helpers to turn mapped D3D11 staging textures into small images and write them out.
// main.cpp ("RT diagnostic capture" in the Luma panel, Ctrl+Shift+F9) copies each stage of a frame into staging
// textures at present, maps them a frame later, and calls these. Output per capture N in the capture folder:
//   cNNNN_<stage>.bin  quarter resolution, row major, no header: u8 gray (gray stages), u8 RGB (rgb stages), f32 (depth)
//   cNNNN_<stage>.png  the same for viewing (depth normalized)
//   cNNNN.txt          key=value metadata (camera, light, scene counts), read by Tests/rt_diag_analyze.cpp
// See RTAO_NOTES.md "Diagnostic capture".

#include <d3d11.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace RTDiag
{
   constexpr uint32_t kDownsample = 4;

   inline float HalfToFloat(uint16_t h)
   {
      const uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
      float value;
      if (exponent == 0)
         value = std::ldexp(float(mantissa), -24);
      else if (exponent == 31)
         value = mantissa ? NAN : INFINITY;
      else
         value = std::ldexp(float(mantissa | 1024), int(exponent) - 25);
      return sign ? -value : value;
   }

   inline float Float11or10(uint32_t bits, int mantissa_bits)
   {
      const uint32_t exponent = bits >> mantissa_bits, mantissa = bits & ((1u << mantissa_bits) - 1);
      if (exponent == 0)
         return std::ldexp(float(mantissa), -14 - mantissa_bits);
      if (exponent == 31)
         return mantissa ? NAN : INFINITY;
      return std::ldexp(float(mantissa | (1u << mantissa_bits)), int(exponent) - 15 - mantissa_bits);
   }

   inline uint32_t BytesPerTexel(DXGI_FORMAT format)
   {
      switch (format)
      {
      case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_TYPELESS: return 1;
      case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_TYPELESS: return 2;
      case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R8G8B8A8_TYPELESS:
      case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8A8_TYPELESS:
      case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R11G11B10_FLOAT:
      case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_TYPELESS:
      case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT:
         return 4;
      case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R32G32_FLOAT:
      case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
         return 8;
      default: return 0;
      }
   }

   // One texel as 4 floats (missing channels 0, alpha 1). False if the format isn't handled.
   inline bool DecodeTexel(DXGI_FORMAT format, const uint8_t* p, float out[4])
   {
      out[0] = out[1] = out[2] = 0.f;
      out[3] = 1.f;
      uint32_t u32;
      uint16_t u16[4];
      switch (format)
      {
      case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_TYPELESS: out[0] = p[0] / 255.f; return true;
      case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_TYPELESS: out[0] = p[0] / 255.f; out[1] = p[1] / 255.f; return true;
      case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: std::memcpy(u16, p, 2); out[0] = u16[0] / 65535.f; return true;
      case DXGI_FORMAT_R16_FLOAT: std::memcpy(u16, p, 2); out[0] = HalfToFloat(u16[0]); return true;
      case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R8G8B8A8_TYPELESS:
         for (int c = 0; c < 4; c++) out[c] = p[c] / 255.f;
         return true;
      case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8A8_TYPELESS:
         out[0] = p[2] / 255.f; out[1] = p[1] / 255.f; out[2] = p[0] / 255.f; out[3] = p[3] / 255.f;
         return true;
      case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS:
         std::memcpy(&u32, p, 4);
         out[0] = (u32 & 1023) / 1023.f; out[1] = ((u32 >> 10) & 1023) / 1023.f; out[2] = ((u32 >> 20) & 1023) / 1023.f; out[3] = (u32 >> 30) / 3.f;
         return true;
      case DXGI_FORMAT_R11G11B10_FLOAT:
         std::memcpy(&u32, p, 4);
         out[0] = Float11or10(u32 & 2047, 6); out[1] = Float11or10((u32 >> 11) & 2047, 6); out[2] = Float11or10(u32 >> 22, 5);
         return true;
      case DXGI_FORMAT_R16G16_FLOAT: std::memcpy(u16, p, 4); out[0] = HalfToFloat(u16[0]); out[1] = HalfToFloat(u16[1]); return true;
      case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_TYPELESS: std::memcpy(u16, p, 4); out[0] = u16[0] / 65535.f; out[1] = u16[1] / 65535.f; return true;
      case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: std::memcpy(&out[0], p, 4); return true;
      case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: std::memcpy(&u32, p, 4); out[0] = (u32 & 0xFFFFFF) / 16777215.f; return true;
      case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS:
         std::memcpy(u16, p, 8);
         for (int c = 0; c < 4; c++) out[c] = HalfToFloat(u16[c]);
         return true;
      case DXGI_FORMAT_R16G16B16A16_UNORM: std::memcpy(u16, p, 8); for (int c = 0; c < 4; c++) out[c] = u16[c] / 65535.f; return true;
      case DXGI_FORMAT_R32G32_FLOAT: std::memcpy(out, p, 8); return true;
      case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: std::memcpy(&out[0], p, 4); return true;
      default: return false;
      }
   }

   enum class Kind
   {
      Gray, // One channel ("channel") as 0..1, u8
      Rgb565, // R16 holding RGB565 (RT sun scene view), u8 RGB
      DepthPacked, // RGBA8 holding the 4 bytes of a float (RTAO's packed depth copy), f32
      DepthFloat, // Channel 0 as is, f32
      Color, // RGB, scRGB (linear, 1 = 80 nits) when "scrgb", else already display encoded, u8 RGB sRGB
      Id16, // R16_UNORM holding an id: written as u16 (.bin), the png shows id != 0 as white
   };

   struct Image
   {
      uint32_t width = 0, height = 0, channels = 0; // channels: 1 gray, 3 rgb, 0 float
      std::vector<uint8_t> u8;
      std::vector<float> f32;
      std::vector<uint16_t> u16; // Id16
   };

   inline float LinearToSrgb(float v)
   {
      v = v <= 0.f ? 0.f : (v >= 1.f ? 1.f : v);
      return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
   }

   // From a mapped staging texture, point sampled every kDownsample texels (the block center). Empty if unsupported.
   inline Image Convert(const D3D11_MAPPED_SUBRESOURCE& mapped, DXGI_FORMAT format, uint32_t width, uint32_t height, Kind kind, int channel = 0, bool scrgb = false)
   {
      Image image;
      const uint32_t bpp = BytesPerTexel(format);
      if (!bpp || !mapped.pData)
         return image;
      image.width = width / kDownsample;
      image.height = height / kDownsample;
      if (kind == Kind::Id16)
      {
         if (format != DXGI_FORMAT_R16_UNORM && format != DXGI_FORMAT_R16_TYPELESS)
            return Image();
         image.channels = 1;
         image.u16.resize(size_t(image.width) * image.height);
         image.u8.resize(image.u16.size());
         for (uint32_t y = 0; y < image.height; y++)
            for (uint32_t x = 0; x < image.width; x++)
            {
               uint16_t id;
               std::memcpy(&id, static_cast<const uint8_t*>(mapped.pData) + size_t(y * kDownsample + kDownsample / 2) * mapped.RowPitch + size_t(x * kDownsample + kDownsample / 2) * 2, 2);
               image.u16[size_t(y) * image.width + x] = id;
               image.u8[size_t(y) * image.width + x] = id ? 255 : 0;
            }
         return image;
      }
      const bool float_out = kind == Kind::DepthPacked || kind == Kind::DepthFloat;
      const bool rgb_out = kind == Kind::Rgb565 || kind == Kind::Color;
      image.channels = float_out ? 0 : (rgb_out ? 3 : 1);
      if (float_out)
         image.f32.resize(size_t(image.width) * image.height);
      else
         image.u8.resize(size_t(image.width) * image.height * image.channels);
      const uint8_t* base = static_cast<const uint8_t*>(mapped.pData);
      for (uint32_t y = 0; y < image.height; y++)
         for (uint32_t x = 0; x < image.width; x++)
         {
            const uint32_t sx = x * kDownsample + kDownsample / 2, sy = y * kDownsample + kDownsample / 2;
            const uint8_t* p = base + size_t(sy) * mapped.RowPitch + size_t(sx) * bpp;
            const size_t i = size_t(y) * image.width + x;
            float t[4];
            if (kind == Kind::DepthPacked)
            {
               float d;
               std::memcpy(&d, p, 4);
               image.f32[i] = d;
               continue;
            }
            if (!DecodeTexel(format, p, t))
               return Image();
            if (kind == Kind::DepthFloat)
               image.f32[i] = t[0];
            else if (kind == Kind::Gray)
               image.u8[i] = uint8_t(std::lround((t[channel] <= 0.f ? 0.f : (t[channel] >= 1.f ? 1.f : t[channel])) * 255.f));
            else if (kind == Kind::Rgb565)
            {
               const uint32_t v = uint32_t(std::lround(t[0] * 65535.f));
               image.u8[i * 3 + 0] = uint8_t(((v >> 11) & 31) * 255 / 31);
               image.u8[i * 3 + 1] = uint8_t(((v >> 5) & 63) * 255 / 63);
               image.u8[i * 3 + 2] = uint8_t((v & 31) * 255 / 31);
            }
            else
            {
               for (int c = 0; c < 3; c++)
               {
                  const float v = scrgb ? LinearToSrgb(t[c] * (80.f / 203.f)) : (t[c] <= 0.f ? 0.f : (t[c] >= 1.f ? 1.f : t[c]));
                  image.u8[i * 3 + c] = uint8_t(std::lround(v * 255.f));
               }
            }
         }
      return image;
   }
}
