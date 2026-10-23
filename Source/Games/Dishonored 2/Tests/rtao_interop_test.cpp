// Standalone smoke test for rtao_interop.{h,cpp} (no game, no ReShade). Exercises, on the local GPU:
// the D3D12 device on the D3D11 adapter (LUID), DXR 1.1 check, shared textures, the shared fence, the D3D11
// depth copy shaders from Shaders/Dishonored 2/Luma_RTAO.hlsl, the D3D12 test kernel and the readback on D3D11.
//
// Build (x64 Native Tools prompt, from this folder):
//   cl /nologo /std:c++20 /EHsc /O2 /I.. rtao_interop_test.cpp ..\rtao_interop.cpp d3d11.lib d3dcompiler.lib
// Run from this folder (it reads the shaders by relative path): rtao_interop_test.exe

#include "rtao_interop.h"

#include <d3dcompiler.h>
#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, ...)                         \
   do                                            \
   {                                             \
      if (!(cond))                               \
      {                                          \
         printf("FAIL %s:%d: ", __FILE__, __LINE__); \
         printf(__VA_ARGS__);                    \
         printf("\n");                           \
         g_failures++;                           \
      }                                          \
   } while (0)

static ID3DBlob* Compile(const wchar_t* path, const char* entry, const char* target)
{
   ID3DBlob* code = nullptr;
   ID3DBlob* errors = nullptr;
   HRESULT hr = D3DCompileFromFile(path, nullptr, nullptr, entry, target, 0, 0, &code, &errors);
   if (FAILED(hr))
   {
      printf("compile %s failed 0x%08lX %s\n", entry, (unsigned long)hr, errors ? (const char*)errors->GetBufferPointer() : "");
      exit(2);
   }
   if (errors)
      errors->Release();
   return code;
}

static std::vector<float> ReadAO(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11ShaderResourceView* srv, UINT& w, UINT& h)
{
   ID3D11Resource* resource = nullptr;
   srv->GetResource(&resource);
   ID3D11Texture2D* texture = nullptr;
   resource->QueryInterface(IID_PPV_ARGS(&texture));
   resource->Release();
   D3D11_TEXTURE2D_DESC desc;
   texture->GetDesc(&desc);
   w = desc.Width;
   h = desc.Height;
   desc.Usage = D3D11_USAGE_STAGING;
   desc.BindFlags = 0;
   desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
   desc.MiscFlags = 0;
   ID3D11Texture2D* staging = nullptr;
   device->CreateTexture2D(&desc, nullptr, &staging);
   context->CopyResource(staging, texture);
   D3D11_MAPPED_SUBRESOURCE mapped;
   context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
   std::vector<float> out(w * h);
   for (UINT y = 0; y < h; y++)
      for (UINT x = 0; x < w; x++)
      {
         const uint8_t* row = (const uint8_t*)mapped.pData + y * mapped.RowPitch;
         out[y * w + x] = desc.Format == DXGI_FORMAT_R16_UNORM ? ((const uint16_t*)row)[x] / 65535.f : row[x] / 255.f;
      }
   context->Unmap(staging, 0);
   staging->Release();
   texture->Release();
   return out;
}

int main()
{
   ID3D11Device* device = nullptr;
   ID3D11DeviceContext* context = nullptr;
   D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_1;
   HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &device, nullptr, &context);
   if (FAILED(hr))
   {
      printf("D3D11CreateDevice failed 0x%08lX\n", (unsigned long)hr);
      return 2;
   }

   RTAO::Interop interop;
   std::string error;
   if (!interop.Init(device, error))
   {
      printf("Init failed (RTAO would fall back to XeGTAO): %s\n", error.c_str());
      return 1;
   }

   const UINT W = 320, H = 180;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());
   printf("interop: %s\n", interop.GetDescription().c_str());

   // Source "game" depth: device depth 0.01 (view depth 100) with a closer square at 0.0125 (view depth 80)
   std::vector<float> depth(W * H, 0.01f);
   for (UINT y = 60; y < 120; y++)
      for (UINT x = 100; x < 200; x++)
         depth[y * W + x] = 0.0125f;
   D3D11_TEXTURE2D_DESC depth_desc = { W, H, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   D3D11_SUBRESOURCE_DATA depth_data = { depth.data(), W * 4, 0 };
   ID3D11Texture2D* depth_texture = nullptr;
   device->CreateTexture2D(&depth_desc, &depth_data, &depth_texture);
   ID3D11ShaderResourceView* depth_srv = nullptr;
   device->CreateShaderResourceView(depth_texture, nullptr, &depth_srv);

   ID3DBlob* vs_code = Compile(L"..\\..\\..\\..\\Shaders\\Global\\Luma_Copy_VS.hlsl", "main", "vs_5_0");
   ID3DBlob* ps_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", interop.IsDepthPacked() ? "depth_pack_ps" : "depth_copy_ps", "ps_5_0");
   ID3D11VertexShader* vs = nullptr;
   ID3D11PixelShader* ps = nullptr;
   device->CreateVertexShader(vs_code->GetBufferPointer(), vs_code->GetBufferSize(), nullptr, &vs);
   device->CreatePixelShader(ps_code->GetBufferPointer(), ps_code->GetBufferSize(), nullptr, &ps);

   // InvProj such that the kernel's view depth is 1 / device depth (column major: _m23 = [14], _m32 = [11], _m33 = [15])
   RTAO::Constants constants = {};
   constants.inv_projection[14] = -1.f;
   constants.inv_projection[11] = 1.f;
   constants.inv_projection[15] = 0.f;
   constants.strength = 1.f;

   for (int mode = 0; mode < 2; mode++)
   {
      // Depth copy (what main.cpp's DrawRTAO does through DrawCustomPixelShader)
      ID3D11RenderTargetView* rtv = interop.GetDepthRTV();
      D3D11_VIEWPORT viewport = { 0, 0, (float)W, (float)H, 0, 1 };
      context->RSSetViewports(1, &viewport);
      context->OMSetRenderTargets(1, &rtv, nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
      context->VSSetShader(vs, nullptr, 0);
      context->PSSetShader(ps, nullptr, 0);
      context->PSSetShaderResources(0, 1, &depth_srv);
      context->Draw(4, 0);
      ID3D11RenderTargetView* null_rtv = nullptr;
      context->OMSetRenderTargets(1, &null_rtv, nullptr);
      rtv->Release();

      constants.debug_mode = mode;
      constants.frame = 0;
      CHECK(interop.Run(context, constants, error), "Run: %s", error.c_str());

      ID3D11ShaderResourceView* ao_srv = interop.GetAOSRV();
      UINT w = 0, h = 0;
      std::vector<float> ao = ReadAO(device, context, ao_srv, w, h);
      ao_srv->Release();
      auto at = [&](UINT x, UINT y) { return ao[y * w + x]; };
      if (mode == 0)
      {
         printf("depth test AO: far %.3f, behind the square's edge %.3f, square %.3f\n", at(20, 20), at(203, 90), at(150, 90));
         CHECK(std::fabs(at(20, 20) - 1.f) < 0.01f, "open area should be unoccluded");
         CHECK(at(203, 90) < 0.95f, "pixels next to the closer square should be occluded");
         CHECK(std::fabs(at(150, 90) - 1.f) < 0.01f, "the closer square itself should be unoccluded");
      }
      else
      {
         // Bar sits at x 0..15 on frame 0; cells are 32 px
         printf("test pattern: cell(1,0) %.3f, cell(1,1) %.3f, bar %.3f\n", at(40, 5), at(40, 40), at(5, 5));
         CHECK(std::fabs(at(40, 5) - 1.f) < 0.01f, "cell (1,0) should be 1");
         CHECK(std::fabs(at(40, 40) - 0.35f) < 0.01f, "cell (1,1) should be 0.35");
         CHECK(at(5, 5) < 0.01f, "bar should be 0");
      }
   }

   // Resize path
   CHECK(interop.EnsureTargets(device, W / 2, H / 2, W / 2, H / 2, error), "EnsureTargets resize: %s", error.c_str());
   interop.Shutdown();

   printf(g_failures ? "FAILED (%d)\n" : "PASSED\n", g_failures);
   return g_failures ? 1 : 0;
}
