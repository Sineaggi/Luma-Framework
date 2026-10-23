// Standalone smoke test for rtao_interop.{h,cpp} (no game, no ReShade). Exercises, on the local GPU:
// the D3D12 device on the D3D11 adapter (LUID), DXR 1.1 check, shared textures, the shared fence, the D3D11
// depth copy shaders from Shaders/Dishonored 2/Luma_RTAO.hlsl, the D3D12 test kernel and the readback on D3D11.
// M1: TLAS retention, D3D11 -> D3D12 shared buffers, BLAS (CPU and shared-buffer input) + TLAS of a box scene,
// ray traced AO (RayQuery) and the denoiser, checked against a CPU ray cast of the same scene.
//
// Build (x64 Native Tools prompt, from this folder):
//   cl /nologo /std:c++20 /EHsc /O2 /I.. rtao_interop_test.cpp ..\rtao_interop.cpp ..\rtao_accel.cpp ..\rtao_pass.cpp ..\rtao_capture.cpp d3d11.lib d3dcompiler.lib
// Run from this folder (it reads the shaders by relative path): rtao_interop_test.exe

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rtao_interop.h"
#include "rtao_accel.h"
#include "rtao_capture.h"
#include "rtao_pass.h"

#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// ---- M1: acceleration structures, ray traced AO, denoiser

// Row-vector 4x4 helpers (clip = float4(world, 1) * M), the convention of rtao_pass.h
struct Mat4
{
   float m[16] = {};
   float& operator()(int r, int c) { return m[r * 4 + c]; }
   float operator()(int r, int c) const { return m[r * 4 + c]; }
};
static Mat4 Mul(const Mat4& a, const Mat4& b)
{
   Mat4 o;
   for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++)
         for (int k = 0; k < 4; k++)
            o(r, c) += a(r, k) * b(k, c);
   return o;
}
static Mat4 Inverse(const Mat4& in)
{
   double a[4][8];
   for (int r = 0; r < 4; r++)
      for (int c = 0; c < 8; c++)
         a[r][c] = c < 4 ? in(r, c) : (c - 4 == r ? 1.0 : 0.0);
   for (int c = 0; c < 4; c++)
   {
      int pivot = c;
      for (int r = c + 1; r < 4; r++)
         if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
            pivot = r;
      for (int k = 0; k < 8; k++)
         std::swap(a[c][k], a[pivot][k]);
      const double d = a[c][c];
      for (int k = 0; k < 8; k++)
         a[c][k] /= d;
      for (int r = 0; r < 4; r++)
         if (r != c)
         {
            const double f = a[r][c];
            for (int k = 0; k < 8; k++)
               a[r][k] -= f * a[c][k];
         }
   }
   Mat4 o;
   for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++)
         o(r, c) = float(a[r][c + 4]);
   return o;
}
struct V3
{
   float x, y, z;
};
static V3 Sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 Cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 Normalize(V3 a)
{
   float l = std::sqrt(Dot(a, a));
   return { a.x / l, a.y / l, a.z / l };
}

struct Box
{
   V3 min, max;
};
// 0 is the ground; the floating box (3) goes through the shared-buffer BLAS (the game capture route)
static const Box kBoxes[] = {
   { { -20, -1, -20 }, { 20, 0, 20 } },
   { { -1, 0, -1 }, { 1, 2, 1 } },
   { { 3, 0, 2 }, { 4, 3, 3 } },
   { { -4.5f, 0.6f, -1 }, { -3.5f, 1.6f, 0 } }, // 0.6 above the ground, inside the 1.5 AO radius
   { { 4.5f, 0, -4 }, { 5.5f, 1, -3 } }, // Captured through the MVP-only family
};
static constexpr int kBoxCount = int(sizeof(kBoxes) / sizeof(kBoxes[0]));

// Unit cube [0,1]^3
static const float kCubeVertices[8 * 3] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1 };
static const uint32_t kCubeIndices[36] = { 0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5 };

static bool RayBox(V3 o, V3 d, const Box& b, float& t_hit)
{
   float t0 = 0.f, t1 = 1e30f;
   const float* oo = &o.x;
   const float* dd = &d.x;
   const float* mn = &b.min.x;
   const float* mx = &b.max.x;
   for (int a = 0; a < 3; a++)
   {
      if (std::fabs(dd[a]) < 1e-12f)
      {
         if (oo[a] < mn[a] || oo[a] > mx[a])
            return false;
         continue;
      }
      float ta = (mn[a] - oo[a]) / dd[a], tb = (mx[a] - oo[a]) / dd[a];
      if (ta > tb)
         std::swap(ta, tb);
      t0 = std::max(t0, ta);
      t1 = std::min(t1, tb);
      if (t0 > t1)
         return false;
   }
   t_hit = t0;
   return true;
}

static float BoxDistance(V3 p, const Box& b)
{
   float dx = std::max({ b.min.x - p.x, 0.f, p.x - b.max.x });
   float dy = std::max({ b.min.y - p.y, 0.f, p.y - b.max.y });
   float dz = std::max({ b.min.z - p.z, 0.f, p.z - b.max.z });
   return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static void TestTlasRetention()
{
   RTAO::TlasBuilder tlas;
   const float identity[3][4] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
   // 1 and 2 are static and in view long enough to be promoted
   for (uint32_t i = 0; i < RTAO::TlasBuilder::kPromoteFrames; i++)
   {
      tlas.BeginFrame();
      tlas.AddInstance(1, 1, identity);
      tlas.AddInstance(2, 1, identity);
   }
   // A moving object: a new identity (key) every frame
   for (uint64_t key = 100; key < 110; key++)
   {
      tlas.BeginFrame();
      tlas.AddInstance(2, 1, identity); // 2 stays in view, 1 left the view
      tlas.AddInstance(key, 1, identity);
   }
   CHECK(tlas.RetainedCount() == 2 + RTAO::TlasBuilder::kYoungRetainFrames + 1, "a moving object keeps only its last few copies (got %zu)", tlas.RetainedCount());
   for (uint32_t i = 0; i < RTAO::TlasBuilder::kRetainFrames - 10; i++)
   {
      tlas.BeginFrame();
      tlas.AddInstance(2, 1, identity);
   }
   CHECK(tlas.RetainedCount() == 2, "a promoted instance must be kept for kRetainFrames after it left the view, the moving copies gone (got %zu)", tlas.RetainedCount());
   tlas.BeginFrame();
   tlas.AddInstance(2, 1, identity);
   CHECK(tlas.RetainedCount() == 1, "a promoted instance must be pruned after kRetainFrames (got %zu)", tlas.RetainedCount());
   printf("TLAS retention: ok\n");
}

// The game capture route: D3D11 CopySubresourceRegion from a (non shared) D3D11 buffer into a D3D12 shared buffer
static void TestSharedBuffer(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop)
{
   std::string error;
   std::vector<uint32_t> pattern(256);
   for (uint32_t i = 0; i < pattern.size(); i++)
      pattern[i] = 0xA5000000u | i * 7919u;
   D3D11_BUFFER_DESC desc = { UINT(pattern.size() * 4), D3D11_USAGE_IMMUTABLE, D3D11_BIND_VERTEX_BUFFER, 0, 0, 0 };
   D3D11_SUBRESOURCE_DATA data = { pattern.data(), 0, 0 };
   ID3D11Buffer* source = nullptr;
   device->CreateBuffer(&desc, &data, &source);
   ID3D12Resource* shared12 = nullptr;
   ID3D11Buffer* shared11 = nullptr;
   if (!interop.CreateSharedBuffer(device, pattern.size() * 4, shared12, shared11, error))
   {
      CHECK(false, "CreateSharedBuffer: %s", error.c_str());
      source->Release();
      return;
   }
   context->CopySubresourceRegion(shared11, 0, 0, 0, 0, source, 0, nullptr);

   D3D12_RESOURCE_DESC readback_desc = shared12->GetDesc();
   readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE; // The shared one may allow UAVs, a readback buffer can't
   D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_READBACK };
   ID3D12Resource* readback = nullptr;
   interop.GetDevice()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
   CHECK(interop.Run(context, [&](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo&, std::string&) {
      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition.pResource = shared12;
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
      list->ResourceBarrier(1, &barrier);
      list->CopyBufferRegion(readback, 0, shared12, 0, pattern.size() * 4);
      std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
      list->ResourceBarrier(1, &barrier);
      return true;
   }, error), "Run (shared buffer): %s", error.c_str());
   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   const uint32_t* mapped = nullptr;
   readback->Map(0, nullptr, (void**)&mapped);
   bool same = mapped && std::memcmp(mapped, pattern.data(), pattern.size() * 4) == 0;
   readback->Unmap(0, nullptr);
   CHECK(same, "D3D11 copy into the shared buffer must be visible on D3D12");
   printf("shared buffer (D3D11 CopySubresourceRegion -> D3D12): %s\n", same ? "ok" : "MISMATCH");
   readback->Release();
   shared11->Release();
   shared12->Release();
   source->Release();
}

static void TestRayTracedScene(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps)
{
   std::string error;
   const UINT W = 320, H = 180;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());

   // Camera: LH, row vectors, reversed Z
   const V3 eye = { 0, 4, -9 }, target = { 0, 0.5f, 0 }, up = { 0, 1, 0 };
   const V3 f = Normalize(Sub(target, eye)), r = Normalize(Cross(up, f)), u = Cross(f, r);
   Mat4 view;
   view(0, 0) = r.x; view(0, 1) = u.x; view(0, 2) = f.x;
   view(1, 0) = r.y; view(1, 1) = u.y; view(1, 2) = f.y;
   view(2, 0) = r.z; view(2, 1) = u.z; view(2, 2) = f.z;
   view(3, 0) = -Dot(r, eye); view(3, 1) = -Dot(u, eye); view(3, 2) = -Dot(f, eye); view(3, 3) = 1;
   const float n = 0.1f, fr = 1000.f, sy = 1.f / std::tan(3.14159265f / 6.f), sx = sy / (float(W) / float(H));
   Mat4 proj;
   proj(0, 0) = sx;
   proj(1, 1) = sy;
   proj(2, 2) = -n / (fr - n);
   proj(2, 3) = 1;
   proj(3, 2) = n * fr / (fr - n);
   const Mat4 view_proj = Mul(view, proj), inv_view_proj = Inverse(view_proj);

   // Reference "rasterization" on the CPU: depth, hit object and world position per pixel
   std::vector<float> depth(W * H, 0.f);
   std::vector<int> hit_id(W * H, -1);
   std::vector<V3> hit_pos(W * H);
   for (UINT y = 0; y < H; y++)
      for (UINT x = 0; x < W; x++)
      {
         const float ndc_x = (x + 0.5f) / W * 2.f - 1.f, ndc_y = 1.f - (y + 0.5f) / H * 2.f;
         float v[4] = { ndc_x, ndc_y, 0.5f, 1.f }, w[4] = {};
         for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
               w[c] += v[k] * inv_view_proj(k, c);
         const V3 dir = Normalize(Sub({ w[0] / w[3], w[1] / w[3], w[2] / w[3] }, eye));
         float best = 1e30f;
         for (int b = 0; b < kBoxCount; b++)
         {
            float t;
            if (RayBox(eye, dir, kBoxes[b], t) && t < best)
            {
               best = t;
               hit_id[y * W + x] = b;
            }
         }
         if (hit_id[y * W + x] < 0)
            continue;
         const V3 p = { eye.x + dir.x * best, eye.y + dir.y * best, eye.z + dir.z * best };
         hit_pos[y * W + x] = p;
         float clip[4] = {};
         const float pv[4] = { p.x, p.y, p.z, 1.f };
         for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
               clip[c] += pv[k] * view_proj(k, c);
         depth[y * W + x] = clip[2] / clip[3];
      }
   D3D11_TEXTURE2D_DESC depth_desc = { W, H, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   D3D11_SUBRESOURCE_DATA depth_data = { depth.data(), W * 4, 0 };
   ID3D11Texture2D* depth_texture = nullptr;
   device->CreateTexture2D(&depth_desc, &depth_data, &depth_texture);
   ID3D11ShaderResourceView* depth_srv = nullptr;
   device->CreateShaderResourceView(depth_texture, nullptr, &depth_srv);

   // The floating box's mesh crosses the way game geometry will: D3D11 buffers copied into D3D12 shared buffers
   ID3D11Buffer* vb11 = nullptr;
   ID3D11Buffer* ib11 = nullptr;
   {
      D3D11_BUFFER_DESC desc = { sizeof(kCubeVertices), D3D11_USAGE_IMMUTABLE, D3D11_BIND_VERTEX_BUFFER, 0, 0, 0 };
      D3D11_SUBRESOURCE_DATA data = { kCubeVertices, 0, 0 };
      device->CreateBuffer(&desc, &data, &vb11);
      desc = { sizeof(kCubeIndices), D3D11_USAGE_IMMUTABLE, D3D11_BIND_INDEX_BUFFER, 0, 0, 0 };
      data = { kCubeIndices, 0, 0 };
      device->CreateBuffer(&desc, &data, &ib11);
   }
   ID3D12Resource* vb12 = nullptr;
   ID3D12Resource* ib12 = nullptr;
   ID3D11Buffer* vb_shared11 = nullptr;
   ID3D11Buffer* ib_shared11 = nullptr;
   CHECK(interop.CreateSharedBuffer(device, sizeof(kCubeVertices), vb12, vb_shared11, error), "shared VB: %s", error.c_str());
   CHECK(interop.CreateSharedBuffer(device, sizeof(kCubeIndices), ib12, ib_shared11, error), "shared IB: %s", error.c_str());
   if (!vb12 || !ib12)
      return;
   context->CopySubresourceRegion(vb_shared11, 0, 0, 0, 0, vb11, 0, nullptr);
   context->CopySubresourceRegion(ib_shared11, 0, 0, 0, 0, ib11, 0, nullptr);

   RTAO::Retirer retirer;
   RTAO::BlasCache blases;
   RTAO::TlasBuilder tlas;
   RTAO::AOPass pass;
   CHECK(pass.Init(interop.GetDevice(), error), "AOPass::Init: %s", error.c_str());

   RTAO::PassSettings settings;
   std::memcpy(settings.view_proj, view_proj.m, sizeof(settings.view_proj));
   std::memcpy(settings.inv_view_proj, inv_view_proj.m, sizeof(settings.inv_view_proj));
   settings.cam_pos[0] = eye.x; settings.cam_pos[1] = eye.y; settings.cam_pos[2] = eye.z;
   settings.cam_fwd[0] = f.x; settings.cam_fwd[1] = f.y; settings.cam_fwd[2] = f.z;
   settings.reversed_z = true;
   settings.radius = 1.5f;
   settings.ray_count = 4;

   uint32_t last_instance_count = 0;
   auto run_frame = [&](RTAO::FilterMode filter_mode) {
      ID3D11RenderTargetView* rtv = interop.GetDepthRTV();
      D3D11_VIEWPORT viewport = { 0, 0, (float)W, (float)H, 0, 1 };
      context->RSSetViewports(1, &viewport);
      context->OMSetRenderTargets(1, &rtv, nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
      context->VSSetShader(copy_vs, nullptr, 0);
      context->PSSetShader(depth_ps, nullptr, 0);
      context->PSSetShaderResources(0, 1, &depth_srv);
      context->Draw(4, 0);
      ID3D11RenderTargetView* null_rtv = nullptr;
      context->OMSetRenderTargets(1, &null_rtv, nullptr);
      rtv->Release();

      settings.filter_mode = filter_mode;
      CHECK(interop.Run(context, [&](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& err) {
         retirer.Collect(info.completed_fence_value);
         const RTAO::BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         RTAO::CpuMesh cube = { kCubeVertices, 8, 12, kCubeIndices, 36 };
         if (!blases.BuildFromCpu(ctx, 1, cube, err))
            return false;
         if (!blases.Has(2))
         {
            D3D12_RESOURCE_BARRIER barriers[2] = {};
            ID3D12Resource* buffers[2] = { vb12, ib12 };
            for (int i = 0; i < 2; i++)
            {
               barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
               barriers[i].Transition.pResource = buffers[i];
               barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
               barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            list->ResourceBarrier(2, barriers);
            RTAO::GpuMesh gpu_cube;
            gpu_cube.vertices = vb12->GetGPUVirtualAddress();
            gpu_cube.vertex_count = 8;
            gpu_cube.indices = ib12->GetGPUVirtualAddress();
            gpu_cube.index_count = 36;
            if (!blases.BuildFromGpu(ctx, 2, gpu_cube, err))
               return false;
            for (auto& barrier : barriers)
               std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
            list->ResourceBarrier(2, barriers);
         }
         blases.FlushBarrier(list);

         tlas.BeginFrame();
         for (int b = 0; b < kBoxCount; b++)
         {
            const Box& box = kBoxes[b];
            const float xform[3][4] = { { box.max.x - box.min.x, 0, 0, box.min.x }, { 0, box.max.y - box.min.y, 0, box.min.y }, { 0, 0, box.max.z - box.min.z, box.min.z } };
            tlas.AddInstance(uint64_t(b), b == 3 ? 2 : 1, xform);
         }
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         if (!tlas.Build(ctx, blases, info.frame_slot, top, last_instance_count, err))
            return false;
         return pass.Record(list, info, top, settings, retirer, err);
      }, error), "Run (RTAO): %s", error.c_str());
      ID3D11ShaderResourceView* ao_srv = interop.GetAOSRV();
      UINT w = 0, h = 0;
      std::vector<float> ao = ReadAO(device, context, ao_srv, w, h);
      ao_srv->Release();
      return ao;
   };

   // Pixel classes, from the CPU reference
   auto average = [&](const std::vector<float>& ao, auto predicate, int& count) {
      double sum = 0;
      count = 0;
      for (UINT i = 0; i < W * H; i++)
         if (predicate(i))
         {
            sum += ao[i];
            count++;
         }
      return count ? float(sum / count) : -1.f;
   };
   auto is_sky = [&](UINT i) { return hit_id[i] < 0; };
   auto is_open_ground = [&](UINT i) {
      if (hit_id[i] != 0)
         return false;
      for (int b = 1; b < kBoxCount; b++)
         if (BoxDistance(hit_pos[i], kBoxes[b]) < 2.f)
            return false;
      return true;
   };
   auto is_corner = [&](UINT i) { // Ground right next to box 1's walls
      return hit_id[i] == 0 && BoxDistance(hit_pos[i], kBoxes[1]) < 0.15f;
   };
   auto is_box_top = [&](UINT i) {
      const V3 p = hit_pos[i];
      return hit_id[i] == 1 && p.y > 1.99f && std::fabs(p.x) < 0.5f && std::fabs(p.z) < 0.5f;
   };
   auto is_under_floating = [&](UINT i) {
      const V3 p = hit_pos[i];
      return hit_id[i] == 0 && p.x > -4.4f && p.x < -3.6f && p.z > -0.9f && p.z < -0.1f;
   };

   int counts[5];
   std::vector<float> raw = run_frame(RTAO::FilterMode::Off);
   const float sky = average(raw, is_sky, counts[0]);
   const float open = average(raw, is_open_ground, counts[1]);
   const float corner = average(raw, is_corner, counts[2]);
   const float top = average(raw, is_box_top, counts[3]);
   const float under = average(raw, is_under_floating, counts[4]);
   printf("RTAO raw (%u TLAS instances, %zu BLAS): sky %.3f (%d px), open ground %.3f (%d), wall corner %.3f (%d), box top %.3f (%d), under the floating box %.3f (%d)\n",
      last_instance_count, blases.Count(), sky, counts[0], open, counts[1], corner, counts[2], top, counts[3], under, counts[4]);
   CHECK(last_instance_count == kBoxCount, "all instances in the TLAS");
   CHECK(counts[0] > 0 && sky > 0.999f, "sky must be unoccluded");
   CHECK(counts[1] > 50 && open > 0.97f, "open ground must be (nearly) unoccluded");
   CHECK(counts[2] > 10 && corner < 0.8f && corner > 0.2f, "ground at a wall must be partly occluded");
   CHECK(counts[3] > 10 && top > 0.97f, "the top of a box must be unoccluded");
   CHECK(counts[4] > 10 && under < 0.9f, "ground under the floating box (shared-buffer BLAS) must see it");

   // Temporal: converges, stays sane
   auto stddev = [&](const std::vector<float>& ao, auto predicate) {
      int count = 0;
      float mean = average(ao, predicate, count);
      double s = 0;
      for (UINT i = 0; i < W * H; i++)
         if (predicate(i))
            s += (ao[i] - mean) * (ao[i] - mean);
      return count ? float(std::sqrt(s / count)) : 0.f;
   };
   const float raw_noise = stddev(raw, is_corner);
   std::vector<float> filtered;
   for (int i = 0; i < 24; i++)
      filtered = run_frame(RTAO::FilterMode::Temporal);
   int c;
   const float filtered_corner = average(filtered, is_corner, c);
   const float filtered_open = average(filtered, is_open_ground, c);
   const float filtered_noise = stddev(filtered, is_corner);
   bool finite = true;
   for (float v : filtered)
      finite = finite && std::isfinite(v) && v >= 0.f && v <= 1.f;
   printf("RTAO temporal (24 frames): open ground %.3f, wall corner %.3f, corner noise %.4f (raw %.4f)\n", filtered_open, filtered_corner, filtered_noise, raw_noise);
   CHECK(finite, "filtered AO must be finite and in [0, 1]");
   CHECK(filtered_open > 0.97f, "open ground must stay unoccluded after filtering");
   CHECK(std::fabs(filtered_corner - corner) < 0.15f, "filtering must not shift the corner AO much");
   CHECK(filtered_noise < raw_noise, "the denoiser must reduce noise");

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   vb12->Release();
   ib12->Release();
   vb_shared11->Release();
   ib_shared11->Release();
   vb11->Release();
   ib11->Release();
   depth_srv->Release();
   depth_texture->Release();
}

// ---- M1: geometry capture (game draws -> BLAS + instance matrices), with game-like buffers for all 3 families

// Reflects some of the game's own vertex shaders, if Luma's dump of them is on this machine
static void TestReflection()
{
   const char* dump = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Dishonored2\\Luma\\Dishonored 2\\Dump\\";
   struct Case
   {
      const char* file;
      RTAO::MatrixSource source;
      const char* skip;
   };
   const Case cases[] = {
      { "0x004D8839.vs_5_0.cso", RTAO::MatrixSource::DirectInstanceBuffer, nullptr }, // G-buffer, direct instance buffer
      { "0x00603116.vs_5_0.cso", RTAO::MatrixSource::DirectInstanceBuffer, nullptr }, // depth only, direct instance buffer
      { "0x000209DB.vs_5_0.cso", RTAO::MatrixSource::IndexedInstanceBuffer, nullptr },
      { "0x0008106B.vs_5_0.cso", RTAO::MatrixSource::ConstantBuffer, nullptr },
      { "0x0236F519.vs_5_0.cso", RTAO::MatrixSource::None, "skinned" },
      { "0x0275C69D.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr }, // MVP only, declares PerViewCB
      { "0x00E0ECFF.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr }, // MVP only, PerViewCB assumed at b1
   };
   int tested = 0;
   for (const Case& c : cases)
   {
      const std::string path = std::string(dump) + c.file;
      FILE* f = fopen(path.c_str(), "rb");
      if (!f)
         continue;
      std::vector<uint8_t> code;
      fseek(f, 0, SEEK_END);
      code.resize(size_t(ftell(f)));
      fseek(f, 0, SEEK_SET);
      fread(code.data(), 1, code.size(), f);
      fclose(f);
      RTAO::VertexShaderInfo info;
      RTAO::ReflectVertexShader(code.data(), code.size(), info);
      printf("reflect %s: source %u, skip %s, PerInstanceCB b%d, instance offset @%d, rows @%d/%d/%d, depth hack @%d, instance t%d (stride %u), index t%d\n", c.file, unsigned(info.source), info.skip_reason ? info.skip_reason : "-",
         int(info.instance_cb_slot), int(info.instance_offset_cb), int(info.row_cb[0]), int(info.row_cb[1]), int(info.row_cb[2]), int(info.depth_hack_cb), int(info.instance_buffer_slot), info.instance_buffer_stride, int(info.instance_index_slot));
      CHECK(info.source == c.source, "%s: wrong matrix source", c.file);
      CHECK(!c.skip || (info.skip_reason && strcmp(info.skip_reason, c.skip) == 0), "%s: wrong skip reason", c.file);
      tested++;
   }
   if (tested == 0)
      printf("reflect: game shader dump not found, skipped\n");
}

static ID3D11Buffer* MakeBuffer(ID3D11Device* device, UINT bind, const void* data, UINT size, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, UINT misc = 0, UINT stride = 0)
{
   D3D11_BUFFER_DESC desc = { size, usage, bind, 0, misc, stride };
   D3D11_SUBRESOURCE_DATA init = { data, 0, 0 };
   ID3D11Buffer* buffer = nullptr;
   HRESULT hr = device->CreateBuffer(&desc, &init, &buffer);
   CHECK(SUCCEEDED(hr), "CreateBuffer 0x%08lX", (unsigned long)hr);
   return buffer;
}

static void BoxTransform(const Box& box, float out[12])
{
   const float t[12] = { box.max.x - box.min.x, 0, 0, box.min.x, 0, box.max.y - box.min.y, 0, box.min.y, 0, 0, box.max.z - box.min.z, box.min.z };
   std::memcpy(out, t, sizeof(t));
}

static void TestGeometryCapture(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps)
{
   std::string error;
   const UINT W = 320, H = 180;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());

   // Same camera and CPU reference as TestRayTracedScene
   const V3 eye = { 0, 4, -9 }, target = { 0, 0.5f, 0 }, up = { 0, 1, 0 };
   const V3 f = Normalize(Sub(target, eye)), r = Normalize(Cross(up, f)), u = Cross(f, r);
   Mat4 view;
   view(0, 0) = r.x; view(0, 1) = u.x; view(0, 2) = f.x;
   view(1, 0) = r.y; view(1, 1) = u.y; view(1, 2) = f.y;
   view(2, 0) = r.z; view(2, 1) = u.z; view(2, 2) = f.z;
   view(3, 0) = -Dot(r, eye); view(3, 1) = -Dot(u, eye); view(3, 2) = -Dot(f, eye); view(3, 3) = 1;
   const float n = 0.1f, fr = 1000.f, sy = 1.f / std::tan(3.14159265f / 6.f), sx = sy / (float(W) / float(H));
   Mat4 proj;
   proj(0, 0) = sx;
   proj(1, 1) = sy;
   proj(2, 2) = -n / (fr - n);
   proj(2, 3) = 1;
   proj(3, 2) = n * fr / (fr - n);
   const Mat4 view_proj = Mul(view, proj), inv_view_proj = Inverse(view_proj);
   std::vector<float> depth(W * H, 0.f);
   std::vector<int> hit_id(W * H, -1);
   std::vector<V3> hit_pos(W * H);
   for (UINT y = 0; y < H; y++)
      for (UINT x = 0; x < W; x++)
      {
         float v[4] = { (x + 0.5f) / W * 2.f - 1.f, 1.f - (y + 0.5f) / H * 2.f, 0.5f, 1.f }, w[4] = {};
         for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
               w[c] += v[k] * inv_view_proj(k, c);
         const V3 dir = Normalize(Sub({ w[0] / w[3], w[1] / w[3], w[2] / w[3] }, eye));
         float best = 1e30f;
         for (int b = 0; b < kBoxCount; b++)
         {
            float t;
            if (RayBox(eye, dir, kBoxes[b], t) && t < best)
            {
               best = t;
               hit_id[y * W + x] = b;
            }
         }
         if (hit_id[y * W + x] < 0)
            continue;
         const V3 p = { eye.x + dir.x * best, eye.y + dir.y * best, eye.z + dir.z * best };
         hit_pos[y * W + x] = p;
         float clip[4] = {};
         const float pv[4] = { p.x, p.y, p.z, 1.f };
         for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
               clip[c] += pv[k] * view_proj(k, c);
         depth[y * W + x] = clip[2] / clip[3];
      }
   D3D11_TEXTURE2D_DESC depth_desc = { W, H, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   D3D11_SUBRESOURCE_DATA depth_data = { depth.data(), W * 4, 0 };
   ID3D11Texture2D* depth_texture = nullptr;
   device->CreateTexture2D(&depth_desc, &depth_data, &depth_texture);
   ID3D11ShaderResourceView* depth_srv = nullptr;
   device->CreateShaderResourceView(depth_texture, nullptr, &depth_srv);

   // Game-like mesh: stride 20 (uv, then position at offset 8), 4 unrelated vertices and 6 indices in front of it
   struct Vertex
   {
      float uv[2];
      float pos[3];
   };
   std::vector<Vertex> vertices(4 + 8, Vertex{ { 9, 9 }, { 99, 99, 99 } });
   for (int i = 0; i < 8; i++)
      std::memcpy(vertices[4 + i].pos, &kCubeVertices[i * 3], 12);
   std::vector<uint16_t> indices(6 + 36, 3);
   for (int i = 0; i < 36; i++)
      indices[6 + i] = uint16_t(kCubeIndices[i]);
   ID3D11Buffer* vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, vertices.data(), UINT(vertices.size() * sizeof(Vertex)));
   ID3D11Buffer* ib = MakeBuffer(device, D3D11_BIND_INDEX_BUFFER, indices.data(), UINT(indices.size() * 2));

   // Family 1: direct instance buffer (stride 96, rows at 48), index = SV_InstanceID + cb_instancebufferoffset (@560 = 1)
   std::vector<float> direct(3 * 24, 7.f);
   BoxTransform(kBoxes[0], &direct[1 * 24 + 12]);
   BoxTransform(kBoxes[1], &direct[2 * 24 + 12]);
   ID3D11Buffer* direct_buffer = MakeBuffer(device, D3D11_BIND_SHADER_RESOURCE, direct.data(), UINT(direct.size() * 4), D3D11_USAGE_IMMUTABLE, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 96);
   ID3D11ShaderResourceView* direct_srv = nullptr;
   device->CreateShaderResourceView(direct_buffer, nullptr, &direct_srv);
   std::vector<uint32_t> direct_cb(640 / 4, 0);
   direct_cb[560 / 4] = 1;
   reinterpret_cast<float*>(direct_cb.data())[556 / 4] = 1.f;
   ID3D11Buffer* direct_cb_buffer = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, direct_cb.data(), 640);

   // Family 2: indexed instance buffer (stride 48), index buffer entry 1 = 0xAB000005 -> record 5 (low 24 bits)
   std::vector<float> indexed(8 * 12, 5.f);
   BoxTransform(kBoxes[2], &indexed[5 * 12]);
   ID3D11Buffer* indexed_buffer = MakeBuffer(device, D3D11_BIND_SHADER_RESOURCE, indexed.data(), UINT(indexed.size() * 4), D3D11_USAGE_IMMUTABLE, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 48);
   ID3D11ShaderResourceView* indexed_srv = nullptr;
   device->CreateShaderResourceView(indexed_buffer, nullptr, &indexed_srv);
   const uint32_t index_values[4] = { 7, 0xAB000005u, 0, 0 };
   ID3D11Buffer* index_buffer = MakeBuffer(device, D3D11_BIND_SHADER_RESOURCE, index_values, sizeof(index_values));
   D3D11_SHADER_RESOURCE_VIEW_DESC index_srv_desc = {};
   index_srv_desc.Format = DXGI_FORMAT_R32_UINT;
   index_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
   index_srv_desc.Buffer.NumElements = 4;
   ID3D11ShaderResourceView* index_srv = nullptr;
   device->CreateShaderResourceView(index_buffer, &index_srv_desc, &index_srv);
   std::vector<uint32_t> indexed_cb(64 / 4, 0);
   indexed_cb[16 / 4] = 1;
   ID3D11Buffer* indexed_cb_buffer = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, indexed_cb.data(), 64);

   // Family 3: PerInstanceCB rows (z @160, y @176, x @192, as in the game), depth hack @464. Plus a "view model":
   // a huge box around the camera, depth hack 0.25, which must be left out (it would occlude everything).
   auto make_cb_family = [&](const Box& box, float depth_hack) {
      std::vector<float> cb(480 / 4, 0.f);
      float t[12];
      BoxTransform(box, t);
      std::memcpy(&cb[192 / 4], &t[0], 16);
      std::memcpy(&cb[176 / 4], &t[4], 16);
      std::memcpy(&cb[160 / 4], &t[8], 16);
      cb[464 / 4] = depth_hack;
      return MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb.data(), 480);
   };
   ID3D11Buffer* model_cb = make_cb_family(kBoxes[3], 1.f);

   // Family 4: MVP only (cb_mvpmatrixz @16, y @32, x @48, w @64, as in the game's skinned shaders; depth hack @84),
   // with the draw's PerViewCB holding cb_viewprojectionmatrix @32
   ID3D11Buffer* mvp_cb = nullptr;
   ID3D11Buffer* per_view_cb = nullptr;
   {
      Mat4 model;
      float t[12];
      BoxTransform(kBoxes[4], t);
      model(0, 0) = t[0];
      model(1, 1) = t[5];
      model(2, 2) = t[10];
      model(3, 0) = t[3];
      model(3, 1) = t[7];
      model(3, 2) = t[11];
      model(3, 3) = 1.f;
      const Mat4 mvp = Mul(model, view_proj);
      std::vector<float> cb(96 / 4, 0.f);
      const uint32_t column_offsets[4] = { 48, 32, 16, 64 }; // x, y, z, w
      for (int c = 0; c < 4; c++)
         for (int r = 0; r < 4; r++)
            cb[column_offsets[c] / 4 + r] = mvp(r, c);
      cb[84 / 4] = 1.f;
      mvp_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb.data(), 96);
      std::vector<float> view(112 / 4, 0.f);
      std::memcpy(&view[32 / 4], view_proj.m, 64);
      per_view_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, view.data(), 112);
   }
   ID3D11Buffer* view_model_cb = make_cb_family({ { -50, -50, -50 }, { 50, 50, 50 } }, 0.25f);

   RTAO::VertexShaderInfo direct_vs, indexed_vs, cb_vs, mvp_vs;
   mvp_vs.source = RTAO::MatrixSource::MvpConstantBuffer;
   mvp_vs.skip_reason = nullptr;
   mvp_vs.instance_cb_slot = 2;
   mvp_vs.mvp_cb[0] = 48;
   mvp_vs.mvp_cb[1] = 32;
   mvp_vs.mvp_cb[2] = 16;
   mvp_vs.mvp_cb[3] = 64;
   mvp_vs.depth_hack_cb = 84;
   mvp_vs.per_view_cb_slot = 1;
   mvp_vs.view_proj_cb = 32;
   direct_vs.source = RTAO::MatrixSource::DirectInstanceBuffer;
   direct_vs.skip_reason = nullptr;
   direct_vs.instance_cb_slot = 2;
   direct_vs.instance_offset_cb = 560;
   direct_vs.depth_hack_cb = 556;
   direct_vs.instance_buffer_slot = 3;
   direct_vs.instance_buffer_stride = 96;
   direct_vs.record_row0 = 48;
   indexed_vs.source = RTAO::MatrixSource::IndexedInstanceBuffer;
   indexed_vs.skip_reason = nullptr;
   indexed_vs.instance_cb_slot = 2;
   indexed_vs.instance_offset_cb = 16;
   indexed_vs.instance_buffer_slot = 1;
   indexed_vs.instance_buffer_stride = 48;
   indexed_vs.instance_index_slot = 0;
   cb_vs.source = RTAO::MatrixSource::ConstantBuffer;
   cb_vs.skip_reason = nullptr;
   cb_vs.instance_cb_slot = 2;
   cb_vs.row_cb[0] = 192;
   cb_vs.row_cb[1] = 176;
   cb_vs.row_cb[2] = 160;
   cb_vs.depth_hack_cb = 464;

   auto base_draw = [&]() {
      RTAO::DrawInfo d;
      d.vertex_buffer = vb;
      d.vertex_stride = sizeof(Vertex);
      d.position_format = DXGI_FORMAT_R32G32B32_FLOAT;
      d.position_offset = 8;
      d.index_buffer = ib;
      d.index_format = DXGI_FORMAT_R16_UINT;
      d.index_count = 36;
      d.first_index = 6;
      d.base_vertex = 4;
      return d;
   };
   RTAO::DrawInfo draw_direct = base_draw();
   draw_direct.vs = &direct_vs;
   draw_direct.instance_count = 2;
   draw_direct.instance_cb = direct_cb_buffer;
   draw_direct.instance_buffer = direct_srv;
   RTAO::DrawInfo draw_indexed = base_draw();
   draw_indexed.vs = &indexed_vs;
   draw_indexed.instance_cb = indexed_cb_buffer;
   draw_indexed.instance_buffer = indexed_srv;
   draw_indexed.instance_index = index_srv;
   RTAO::DrawInfo draw_cb = base_draw();
   draw_cb.vs = &cb_vs;
   draw_cb.instance_cb = model_cb;
   RTAO::DrawInfo draw_view_model = draw_cb;
   draw_view_model.instance_cb = view_model_cb;
   RTAO::DrawInfo draw_mvp = base_draw();
   draw_mvp.vs = &mvp_vs;
   draw_mvp.instance_cb = mvp_cb;
   draw_mvp.per_view_cb = per_view_cb;

   ID3DBlob* cs_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "capture_instances_cs", "cs_5_0");
   ID3D11ComputeShader* capture_cs = nullptr;
   device->CreateComputeShader(cs_code->GetBufferPointer(), cs_code->GetBufferSize(), nullptr, &capture_cs);
   cs_code->Release();

   RTAO::GeometryCapture capture;
   CHECK(capture.Init(device, interop, capture_cs, error), "GeometryCapture::Init: %s", error.c_str());
   capture_cs->Release();
   ID3D11DeviceContext* deferred = nullptr;
   device->CreateDeferredContext(0, &deferred);

   RTAO::Retirer retirer;
   RTAO::BlasCache blases;
   RTAO::TlasBuilder tlas;
   RTAO::AOPass pass;
   CHECK(pass.Init(interop.GetDevice(), error), "AOPass::Init: %s", error.c_str());
   RTAO::GpuTimer timer;
   CHECK(timer.Init(interop.GetDevice(), interop.GetTimestampFrequency(), error), "GpuTimer::Init: %s", error.c_str());
   RTAO::PassSettings settings;
   std::memcpy(settings.view_proj, view_proj.m, sizeof(settings.view_proj));
   std::memcpy(settings.inv_view_proj, inv_view_proj.m, sizeof(settings.inv_view_proj));
   settings.cam_pos[0] = eye.x; settings.cam_pos[1] = eye.y; settings.cam_pos[2] = eye.z;
   settings.cam_fwd[0] = f.x; settings.cam_fwd[1] = f.y; settings.cam_fwd[2] = f.z;

   auto run_frame = [&](RTAO::FilterMode filter_mode, uint32_t debug_mode) {
      // The "game" draws: two families on the immediate context, the third (and the view model) on a deferred one
      CHECK(capture.CaptureDraw(context, draw_direct), "capture (direct)");
      CHECK(capture.CaptureDraw(context, draw_indexed), "capture (indexed)");
      CHECK(capture.CaptureDraw(context, draw_mvp), "capture (MVP only)");
      CHECK(capture.CaptureDraw(deferred, draw_cb), "capture (cbuffer)");
      CHECK(capture.CaptureDraw(deferred, draw_view_model), "capture (view model)");
      ID3D11CommandList* command_list = nullptr;
      deferred->FinishCommandList(FALSE, &command_list);
      capture.OnFinishCommandList(deferred, command_list);
      capture.OnExecuteCommandList(command_list);
      context->ExecuteCommandList(command_list, TRUE);
      command_list->Release();

      ID3D11RenderTargetView* rtv = interop.GetDepthRTV();
      D3D11_VIEWPORT viewport = { 0, 0, (float)W, (float)H, 0, 1 };
      context->RSSetViewports(1, &viewport);
      context->OMSetRenderTargets(1, &rtv, nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
      context->VSSetShader(copy_vs, nullptr, 0);
      context->PSSetShader(depth_ps, nullptr, 0);
      context->PSSetShaderResources(0, 1, &depth_srv);
      context->Draw(4, 0);
      ID3D11RenderTargetView* null_rtv = nullptr;
      context->OMSetRenderTargets(1, &null_rtv, nullptr);
      rtv->Release();

      settings.filter_mode = filter_mode;
      settings.debug_mode = debug_mode;
      CHECK(interop.Run(context, [&](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& err) {
         retirer.Collect(info.completed_fence_value);
         timer.Collect(info.frame_slot);
         timer.Begin(list, info.frame_slot);
         if (!capture.Record(list, info, blases, tlas, retirer, 64, err))
            return false;
         const RTAO::BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         uint32_t count = 0;
         if (!tlas.Build(ctx, blases, info.frame_slot, top, count, err))
            return false;
         if (!pass.Record(list, info, top, settings, retirer, err))
            return false;
         timer.End(list, info.frame_slot);
         return true;
      }, error), "Run (capture): %s", error.c_str());
      capture.OnPresent();
      ID3D11ShaderResourceView* ao_srv = interop.GetAOSRV();
      UINT w = 0, h = 0;
      std::vector<float> ao = ReadAO(device, context, ao_srv, w, h);
      ao_srv->Release();
      return ao;
   };

   std::vector<float> raw;
   for (int i = 0; i < 6; i++) // Matrices are read back 3 frames late
      raw = run_frame(RTAO::FilterMode::Off, 0);
   const RTAO::CaptureStats stats = capture.GetStats();
   printf("capture: draws %llu seen / %llu captured, instances %llu, meshes %llu (ready %llu), buffers copied %llu (%llu KB arena), BLAS built %llu, TLAS instances %llu, view model instances %llu\n",
      (unsigned long long)stats.draws_seen, (unsigned long long)stats.draws_captured, (unsigned long long)stats.instances_captured, (unsigned long long)stats.meshes, (unsigned long long)stats.meshes_ready,
      (unsigned long long)stats.buffers_copied, (unsigned long long)(stats.arena_vertex_bytes / 1024), (unsigned long long)stats.blas_built, (unsigned long long)stats.tlas_instances, (unsigned long long)stats.view_model_instances);
   CHECK(stats.meshes == 1 && stats.blas_built == 1, "all families draw the same mesh: one BLAS");
   CHECK(stats.buffers_copied == 2, "the vertex and index buffers are copied once");
   CHECK(stats.tlas_instances == kBoxCount, "%d boxes in the TLAS (the view model left out), got %llu", kBoxCount, (unsigned long long)stats.tlas_instances);
   CHECK(stats.view_model_instances == 1, "the view model draw must be flagged");

   auto average = [&](const std::vector<float>& ao, auto predicate, int& count) {
      double sum = 0;
      count = 0;
      for (UINT i = 0; i < W * H; i++)
         if (predicate(i))
         {
            sum += ao[i];
            count++;
         }
      return count ? float(sum / count) : -1.f;
   };
   auto is_sky = [&](UINT i) { return hit_id[i] < 0; };
   auto is_open_ground = [&](UINT i) {
      if (hit_id[i] != 0)
         return false;
      for (int b = 1; b < kBoxCount; b++)
         if (BoxDistance(hit_pos[i], kBoxes[b]) < 2.f)
            return false;
      return true;
   };
   auto is_corner = [&](UINT i) { return hit_id[i] == 0 && BoxDistance(hit_pos[i], kBoxes[1]) < 0.15f; };
   auto is_near_b = [&](UINT i) { return hit_id[i] == 0 && BoxDistance(hit_pos[i], kBoxes[2]) < 0.15f; };
   auto is_under_floating = [&](UINT i) {
      const V3 p = hit_pos[i];
      return hit_id[i] == 0 && p.x > -4.4f && p.x < -3.6f && p.z > -0.9f && p.z < -0.1f;
   };
   auto is_near_mvp_box = [&](UINT i) { return hit_id[i] == 0 && BoxDistance(hit_pos[i], kBoxes[4]) < 0.15f; };
   int c[6];
   const float sky = average(raw, is_sky, c[0]), open = average(raw, is_open_ground, c[1]), corner = average(raw, is_corner, c[2]);
   const float near_b = average(raw, is_near_b, c[3]), under = average(raw, is_under_floating, c[4]);
   printf("captured scene AO (raw): sky %.3f, open ground %.3f, wall corner (direct family) %.3f, next to box B (indexed family) %.3f, under the floating box (cbuffer family, deferred) %.3f\n", sky, open, corner, near_b, under);
   CHECK(sky > 0.999f, "sky must be unoccluded");
   CHECK(open > 0.97f, "open ground must be unoccluded (the view model must not be in the TLAS)");
   CHECK(c[2] > 10 && corner < 0.8f, "direct family box occludes");
   CHECK(c[3] > 5 && near_b < 0.8f, "indexed family box occludes");
   CHECK(c[4] > 10 && under < 0.9f, "cbuffer family box occludes");
   const float near_mvp = average(raw, is_near_mvp_box, c[5]);
   printf("next to the MVP family box: %.3f (%d px)\n", near_mvp, c[5]);
   CHECK(c[5] > 5 && near_mvp < 0.8f, "MVP family box occludes");

   // BLAS coverage debug: the captured geometry must line up with the depth
   std::vector<float> coverage = run_frame(RTAO::FilterMode::Off, 3);
   int hits = 0, matched = 0;
   for (UINT i = 0; i < W * H; i++)
      if (hit_id[i] >= 0)
      {
         hits++;
         matched += coverage[i] > 0.99f;
      }
   printf("BLAS coverage: %d of %d geometry pixels match the depth (%.1f%%), RT pass GPU time %.3f ms\n", matched, hits, 100.0 * matched / std::max(hits, 1), timer.LastMs());
   CHECK(matched > hits * 0.98, "captured geometry must line up with the depth");

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   timer.Shutdown();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   capture.Shutdown();
   deferred->Release();
   for (IUnknown* p : std::initializer_list<IUnknown*>{ vb, ib, direct_buffer, direct_srv, direct_cb_buffer, indexed_buffer, indexed_srv, index_buffer, index_srv, indexed_cb_buffer, model_cb, view_model_cb, mvp_cb, per_view_cb, depth_srv, depth_texture })
      p->Release();
}

int main()
{
   setvbuf(stdout, nullptr, _IONBF, 0);
   // RTAO_D3D12_DEBUG=1: D3D12 debug layer + GPU based validation (messages go to the debugger output)
   if (GetEnvironmentVariableA("RTAO_D3D12_DEBUG", nullptr, 0) != 0)
   {
      auto get_debug = reinterpret_cast<PFN_D3D12_GET_DEBUG_INTERFACE>(GetProcAddress(LoadLibraryW(L"d3d12.dll"), "D3D12GetDebugInterface"));
      ID3D12Debug1* debug = nullptr;
      if (get_debug && SUCCEEDED(get_debug(IID_PPV_ARGS(&debug))))
      {
         debug->EnableDebugLayer();
         debug->SetEnableGPUBasedValidation(TRUE);
         debug->Release();
         printf("D3D12 debug layer on\n");
      }
   }
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

   TestTlasRetention();
   TestSharedBuffer(device, context, interop);
   TestRayTracedScene(device, context, interop, vs, ps);
   TestReflection();
   TestGeometryCapture(device, context, interop, vs, ps);

   // With RTAO_D3D12_DEBUG set: any D3D12 validation error fails the test
   ID3D12InfoQueue* info_queue = nullptr;
   if (SUCCEEDED(interop.GetDevice()->QueryInterface(IID_PPV_ARGS(&info_queue))))
   {
      interop.WaitForIdle(5000);
      UINT64 errors = 0;
      const UINT64 count = info_queue->GetNumStoredMessages();
      for (UINT64 i = 0; i < count; i++)
      {
         SIZE_T length = 0;
         info_queue->GetMessage(i, nullptr, &length);
         std::vector<char> storage(length);
         auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
         if (FAILED(info_queue->GetMessage(i, message, &length)))
            continue;
         if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
         {
            printf("D3D12 validation error: %s\n", message->pDescription);
            errors++;
         }
         else if (message->Severity == D3D12_MESSAGE_SEVERITY_WARNING)
         {
            printf("D3D12 validation warning: %s\n", message->pDescription);
         }
      }
      printf("D3D12 debug layer: %llu messages, %llu errors\n", (unsigned long long)count, (unsigned long long)errors);
      CHECK(errors == 0, "D3D12 validation errors");
      info_queue->Release();
   }

   // Resize path
   CHECK(interop.EnsureTargets(device, W / 2, H / 2, W / 2, H / 2, error), "EnsureTargets resize: %s", error.c_str());
   interop.Shutdown();

   printf(g_failures ? "FAILED (%d)\n" : "PASSED\n", g_failures);
   return g_failures ? 1 : 0;
}
