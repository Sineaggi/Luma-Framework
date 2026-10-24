// Standalone smoke test for rtao_interop.{h,cpp} (no game, no ReShade). Exercises, on the local GPU:
// the D3D12 device on the D3D11 adapter (LUID), DXR 1.1 check, shared textures, the shared fence, the D3D11
// depth copy shaders from Shaders/Dishonored 2/Luma_RTAO.hlsl, the D3D12 test kernel and the readback on D3D11.
// M1: TLAS retention, D3D11 -> D3D12 shared buffers, BLAS (CPU and shared-buffer input) + TLAS of a box scene,
// ray traced AO (RayQuery) and the denoiser, checked against a CPU ray cast of the same scene.
// RT sun shadows (rtsun_test.inl): the sun pass on the same scene, the light capture, the screen shadows shaders.
//
// Build (x64 Native Tools prompt, from this folder):
//   cl /nologo /std:c++20 /EHsc /O2 /I.. rtao_interop_test.cpp ..\rtao_interop.cpp ..\rtao_accel.cpp ..\rtao_pass.cpp ..\rtao_capture.cpp ..\rtao_alpha.cpp ..\rtsun_pass.cpp d3d11.lib d3dcompiler.lib
// Run from this folder (it reads the shaders by relative path): rtao_interop_test.exe
// Benchmark (no debug layer): rtao_interop_test.exe bench [folder for .pgm images], RTAO_BENCH_FILTER=<name part> to
// run only some configurations (see RunBenchmark / BenchConfigs).

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rtao_interop.h"
#include "rtao_accel.h"
#include "rtao_capture.h"
#include "rtao_pass.h"
#include "rtsun_pass.h"

#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Where Luma dumped the game's shaders (RTAO_GAME_SHADER_DUMP, or the default Steam install). The reflection checks
// read the game's own shaders from there at test time; no game data is part of the repository.
static std::string GameShaderDumpDir()
{
   char buffer[1024];
   const DWORD length = GetEnvironmentVariableA("RTAO_GAME_SHADER_DUMP", buffer, sizeof(buffer));
   if (length > 0 && length < sizeof(buffer))
   {
      std::string dir(buffer, length);
      if (dir.back() != '\\' && dir.back() != '/')
         dir += '\\';
      return dir;
   }
   return "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Dishonored2\\Luma\\Dishonored 2\\Dump\\";
}

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
   { { -2.5f, 0, -4.5f }, { -1.5f, 1.5f, -3.5f } }, // Captured as an alpha tested (cutout) wind mesh; opaque elsewhere
};
static constexpr int kCutoutBox = 5;
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

   // Shadow map instances, current rule: static world drawn once (a shadow map region the game keeps) stays until its
   // mesh goes; a moving object (shadow only, or main view then shadow map a frame later) leaves no trail; two redraw
   // batches of the same mesh side by side in consecutive frames are not taken for moves
   {
      RTAO::TlasBuilder shadow;
      shadow.BeginFrame();
      for (uint64_t k = 0; k < 6; k++)
      {
         float xform[3][4] = { { 1, 0, 0, 10.f + 0.3f * k }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
         shadow.AddInstance(500 + k, 3, xform, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
      }
      // Moving, seen by the main view then the shadow map a frame later
      for (uint64_t f = 0; f < 40; f++)
      {
         shadow.BeginFrame();
         float now[3][4] = { { 1, 0, 0, 0.1f * f }, { 0, 1, 0, 5 }, { 0, 0, 1, 0 } };
         float before[3][4] = { { 1, 0, 0, 0.1f * (f - 1) }, { 0, 1, 0, 5 }, { 0, 0, 1, 0 } };
         shadow.AddInstance(1000 + f, 2, now, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceMain);
         if (f > 0)
            shadow.AddInstance(1000 + f - 1, 2, before, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
      }
      // Moving, shadow map only
      for (uint64_t f = 0; f < 40; f++)
      {
         shadow.BeginFrame();
         float now[3][4] = { { 1, 0, 0, 0.1f * f }, { 0, 1, 0, 9 }, { 0, 0, 1, 0 } };
         shadow.AddInstance(2000 + f, 4, now, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
      }
      // Two redraw batches (40 each, the same mesh 0.3 apart), one frame apart
      for (int b = 0; b < 2; b++)
      {
         shadow.BeginFrame();
         for (uint64_t k = 0; k < 40; k++)
         {
            float xform[3][4] = { { 1, 0, 0, 100.f + 0.3f * k + 0.15f * b }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
            shadow.AddInstance(3000 + b * 100 + k, 5, xform, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
         }
      }
      for (uint32_t i = 0; i < 10; i++)
         shadow.BeginFrame();
      // 6 static + the two movers' last positions (+ the main view mover's last main-only copy, young) + 80 batch copies
      CHECK(shadow.RetainedCount() >= 6 + 2 + 80 && shadow.RetainedCount() <= 6 + 3 + 80, "moving objects must leave no trail, batches must stay (retained %zu)", shadow.RetainedCount());
      for (uint32_t i = 0; i < 20000; i++)
         shadow.BeginFrame();
      CHECK(shadow.RetainedCount() >= 6 + 2 + 80, "static shadow map world must stay without sightings (retained %zu)", shadow.RetainedCount());
   }

   // The old rule (test toggle): static casters need kShadowPromoteFrames sightings
   {
      RTAO::TlasBuilder shadow;
      shadow.shadow_mode = 1;
      for (uint32_t f = 0; f < RTAO::TlasBuilder::kShadowPromoteFrames; f++)
      {
         shadow.BeginFrame();
         for (uint64_t k = 0; k < 6; k++)
         {
            float xform[3][4] = { { 1, 0, 0, 10.f + 0.3f * k }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
            shadow.AddInstance(500 + k, 3, xform, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
         }
      }
      for (uint64_t f = 0; f < 40; f++)
      {
         shadow.BeginFrame();
         float now[3][4] = { { 1, 0, 0, 0.1f * f }, { 0, 1, 0, 5 }, { 0, 0, 1, 0 } };
         float before[3][4] = { { 1, 0, 0, 0.1f * (f - 1) }, { 0, 1, 0, 5 }, { 0, 0, 1, 0 } };
         shadow.AddInstance(1000 + f, 2, now, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceMain);
         if (f > 0)
            shadow.AddInstance(1000 + f - 1, 2, before, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
      }
      CHECK(shadow.RetainedCount() <= 6 + RTAO::TlasBuilder::kShadowYoungRetainFrames + 2, "a moving object must not leave a trail (retained %zu)", shadow.RetainedCount());
      for (uint32_t i = 0; i < RTAO::TlasBuilder::kRetainFrames + 50; i++)
         shadow.BeginFrame();
      CHECK(shadow.RetainedCount() == 6, "static shadow map casters must be kept past kRetainFrames (retained %zu)", shadow.RetainedCount());
      for (uint32_t i = 0; i < RTAO::TlasBuilder::kShadowRetainFrames; i++)
         shadow.BeginFrame();
      CHECK(shadow.RetainedCount() == 0, "shadow map casters must go after kShadowRetainFrames (retained %zu)", shadow.RetainedCount());
   }
   // Particles in the shadow maps: a mesh bringing 40 new copies in random places every frame (more than a redraw batch)
   // keeps only its latest frame of copies; a one off static redraw burst of another mesh (the game filling its shadow
   // map) stays
   {
      RTAO::TlasBuilder shadow;
      shadow.BeginFrame();
      for (uint64_t k = 0; k < 50; k++)
      {
         float xform[3][4] = { { 1, 0, 0, 3.f * k }, { 0, 1, 0, 100 }, { 0, 0, 1, 0 } };
         shadow.AddInstance(90000 + k, 7, xform, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
      }
      uint64_t next = 100000;
      for (uint32_t f = 0; f < 200; f++)
      {
         shadow.BeginFrame();
         for (uint32_t k = 0; k < 40; k++)
         {
            float xform[3][4] = { { 1, 0, 0, float((next * 7919) % 1000) }, { 0, 1, 0, float((next * 104729) % 1000) }, { 0, 0, 1, 0 } };
            shadow.AddInstance(next++, 8, xform, RTAO::kInstanceMaskOpaque, RTAO::kNoCutoutRecord, RTAO::kInstanceSourceShadow);
         }
      }
      shadow.BeginFrame();
      const RTAO::RetentionStats r = shadow.GetRetentionStats();
      printf("TLAS retention, particles: retained %zu after 200 frames of 40 new copies (dynamic meshes %u, static %u, bursts %u)\n", shadow.RetainedCount(), r.dynamic_meshes, r.shadow_static, r.bursts);
      CHECK(shadow.RetainedCount() <= 50 + 40 * (RTAO::TlasBuilder::kDynamicRetainFrames + 2), "particle copies must not pile up (retained %zu)", shadow.RetainedCount());
      CHECK(r.dynamic_meshes == 1 && r.shadow_static >= 50, "only the particle mesh is dynamic, the static burst stays (dynamic %u, static %u)", r.dynamic_meshes, r.shadow_static);
      for (uint32_t f = 0; f < 1000; f++)
         shadow.BeginFrame();
      CHECK(shadow.RetainedCount() >= 50 && shadow.RetainedCount() <= 50 + 40 * (RTAO::TlasBuilder::kDynamicRetainFrames + 2), "after the particles stop, the static burst stays (retained %zu)", shadow.RetainedCount());
   }
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
   const std::string dump_dir = GameShaderDumpDir(); // Luma's shader dump of the game (read at test time, not committed)
   const char* dump = dump_dir.c_str();
   struct Case
   {
      const char* file;
      RTAO::MatrixSource source;
      const char* skip;
      bool wind = false;
      uint32_t skin_sets = 0; // Skinned: 1 or 2
      uint32_t bones_slot = 0;
      uint32_t mtx_offset_cb = 0;
   };
   const Case cases[] = {
      { "0x004D8839.vs_5_0.cso", RTAO::MatrixSource::DirectInstanceBuffer, nullptr }, // G-buffer, direct instance buffer
      { "0x00603116.vs_5_0.cso", RTAO::MatrixSource::DirectInstanceBuffer, nullptr }, // depth only, direct instance buffer
      { "0x000209DB.vs_5_0.cso", RTAO::MatrixSource::IndexedInstanceBuffer, nullptr },
      { "0x0008106B.vs_5_0.cso", RTAO::MatrixSource::ConstantBuffer, nullptr },
      { "0x0236F519.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr, false, 1, 0, 0 }, // skinned, MVP only, depth only
      { "0x00299934.vs_5_0.cso", RTAO::MatrixSource::DirectInstanceBuffer, nullptr, false, 2, 0, 16 }, // skinned (8 bones), direct instance buffer
      { "0x00C40689.vs_5_0.cso", RTAO::MatrixSource::IndexedInstanceBuffer, nullptr, false, 1, 2, 96 }, // skinned, indexed instance buffer, bones at t2
      { "0x02973411.vs_5_0.cso", RTAO::MatrixSource::ConstantBuffer, nullptr, false, 2, 0, 16 }, // skinned, cb_modelmatrix rows
      // Seen with DYNAMIC vertex buffers in game (movable objects: doors, windows)
      { "0xBDE400DD.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr }, // VERTEX_POSITION0/1 (previous), MVP + previous MVP
      { "0x129571C1.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr },
      { "0x722BBAC1.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr },
      { "0xC81598D1.vs_5_0.cso", RTAO::MatrixSource::ConstantBuffer, nullptr }, // cb_modelmatrix rows + MVP
      { "0x0275C69D.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr }, // MVP only, declares PerViewCB
      { "0x00E0ECFF.vs_5_0.cso", RTAO::MatrixSource::MvpConstantBuffer, nullptr }, // MVP only, PerViewCB assumed at b1
      { "0x65471A10.vs_5_0.cso", RTAO::MatrixSource::IndexedInstanceBuffer, nullptr, true }, // vegetation bending, ro_windbuffer
      { "0x24159CF5.vs_5_0.cso", RTAO::MatrixSource::IndexedInstanceBuffer, nullptr, true }, // spline wind (ro_splineinfos)
      { "0x09B3E8ED.vs_5_0.cso", RTAO::MatrixSource::IndexedInstanceBuffer, nullptr, true }, // vegetation bending, constant wind
      { "0x0E19777D.vs_5_0.cso", RTAO::MatrixSource::DirectInstanceBuffer, nullptr, true }, // vegetation bending, direct family
   };
   auto load = [&](const char* file, std::vector<uint8_t>& code) {
      const std::string path = std::string(dump) + file;
      FILE* f = fopen(path.c_str(), "rb");
      if (!f)
         return false;
      fseek(f, 0, SEEK_END);
      code.resize(size_t(ftell(f)));
      fseek(f, 0, SEEK_SET);
      fread(code.data(), 1, code.size(), f);
      fclose(f);
      return true;
   };
   int tested = 0;
   for (const Case& c : cases)
   {
      std::vector<uint8_t> code;
      if (!load(c.file, code))
         continue;
      RTAO::VertexShaderInfo info;
      RTAO::ReflectVertexShader(code.data(), code.size(), info);
      printf("reflect %s: source %u, skip %s, wind %d, PerInstanceCB b%d, instance offset @%d, rows @%d/%d/%d, depth hack @%d, instance t%d (stride %u), index t%d\n", c.file, unsigned(info.source), info.skip_reason ? info.skip_reason : "-",
         int(info.wind), int(info.instance_cb_slot), int(info.instance_offset_cb), int(info.row_cb[0]), int(info.row_cb[1]), int(info.row_cb[2]), int(info.depth_hack_cb), int(info.instance_buffer_slot), info.instance_buffer_stride, int(info.instance_index_slot));
      CHECK(info.source == c.source, "%s: wrong matrix source", c.file);
      CHECK(!c.skip || (info.skip_reason && strcmp(info.skip_reason, c.skip) == 0), "%s: wrong skip reason", c.file);
      if (c.skin_sets)
         printf("   skinned: %u sets, ro_matrices t%d, cb_mtxoffset @%d\n", info.skin_sets, int(info.bones_slot), int(info.mtx_offset_cb));
      CHECK(info.skinned == (c.skin_sets != 0) && (!c.skin_sets || (info.skin_sets == c.skin_sets && info.bones_slot == c.bones_slot && info.mtx_offset_cb == c.mtx_offset_cb)), "%s: wrong skinning reflection", c.file);
      CHECK(info.wind == c.wind, "%s: wind %d expected", c.file, int(c.wind));
      tested++;
   }

   // Alpha test detection on the game's pixel shaders
   struct PsCase
   {
      const char* file;
      bool discard;
      bool alpha_tested;
      // Texture exact (alpha tested only): alpha slot, reference (cbuffer offset or kNoOffset + immediate), uv scale/bias
      bool exact = false;
      uint32_t slot = 0;
      uint32_t ref_cb = RTAO::kNoOffset;
      float ref = 0.5f;
      uint32_t scalebias_cb = RTAO::kNoOffset;
   };
   const PsCase ps_cases[] = {
      { "0x048591C8.ps_5_0.cso", true, true, true, 0, 32, 0.5f, 16 }, // depth only: diffuse alpha vs cb_defaultalphatest, uv scale/bias
      { "0x00376C8C.ps_5_0.cso", true, true, true, 7, 364, 0.5f, 128 }, // forward lit, cb_defaultalphatest
      { "0x009040D6.ps_5_0.cso", true, true, true, 7, RTAO::kNoOffset, 0.5f, 560 }, // vegetation (cb_vegetanim_*): diffuse alpha vs a hardcoded 0.5
      { "0x00C30E15.ps_5_0.cso", true, true, true, 0, RTAO::kNoOffset, 0.5f, 0 }, // vegetation depth only: alpha test 0.5, then LOD dithering
      { "0x116EF527.ps_5_0.cso", true, true, true, 0, 64, 0.5f, RTAO::kNoOffset }, // hair (cb_hairalphatest), raw TEXCOORD
      { "0x00C4313E.ps_5_0.cso", true, true, true, 8, 628, 0.5f, RTAO::kNoOffset }, // hair, forward lit: scale/bias used elsewhere, not for the alpha
      { "0x013962FA.ps_5_0.cso", true, false }, // LOD dithering only (pixel position): opaque
      { "0x008D47EA.ps_5_0.cso", true, false }, // discard on a DOF blur map: not a material alpha
      { "0x0293C36F.ps_5_0.cso", true, false }, // dissolve effect only (noise textures vs threshold): opaque
      { "0x0001F8D9.ps_5_0.cso", false, false }, // no discard
   };
   for (const PsCase& c : ps_cases)
   {
      std::vector<uint8_t> code;
      if (!load(c.file, code))
         continue;
      RTAO::PixelShaderInfo info;
      RTAO::ReflectPixelShader(code.data(), code.size(), info);
      printf("reflect %s: discard %d, alpha tested %d, texture exact %d (alpha t%d.%c, cb%d, reference @%d / %.3f, scale/bias @%d)\n", c.file, int(info.discard), int(info.alpha_tested), int(info.texture_exact),
         int(info.alpha_slot), "xyzw"[info.alpha_channel & 3], int(info.material_cb_slot), int(info.alpha_ref_cb), info.alpha_ref, int(info.uv_scalebias_cb));
      CHECK(info.discard == c.discard && info.alpha_tested == c.alpha_tested, "%s: expected discard %d, alpha tested %d", c.file, int(c.discard), int(c.alpha_tested));
      if (c.alpha_tested)
         CHECK(info.texture_exact == c.exact && info.alpha_slot == c.slot && info.alpha_channel == 3 && info.alpha_ref_cb == c.ref_cb && std::fabs(info.alpha_ref - c.ref) < 1e-6f && info.uv_scalebias_cb == c.scalebias_cb &&
                  (c.ref_cb == RTAO::kNoOffset && c.scalebias_cb == RTAO::kNoOffset ? true : info.material_cb_slot == 2),
            "%s: texture exact reflection", c.file);
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
   ID3D11Buffer* cutout_cb = make_cb_family(kBoxes[kCutoutBox], 1.f);

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
   // A wind animated, alpha tested plant: captured in rest pose, as a cutout
   RTAO::VertexShaderInfo wind_vs = cb_vs;
   wind_vs.wind = true;
   RTAO::DrawInfo draw_cutout = base_draw();
   draw_cutout.vs = &wind_vs;
   draw_cutout.instance_cb = cutout_cb;
   draw_cutout.cutout = true;

   ID3DBlob* cs_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "capture_instances_cs", "cs_5_0");
   ID3D11ComputeShader* capture_cs = nullptr;
   device->CreateComputeShader(cs_code->GetBufferPointer(), cs_code->GetBufferSize(), nullptr, &capture_cs);
   cs_code->Release();

   RTAO::GeometryCapture capture;
   CHECK(capture.Init(device, interop, capture_cs, nullptr, error), "GeometryCapture::Init: %s", error.c_str());
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

   uint32_t last_tlas_cutouts = 0;
   bool repeat_draws = false; // Another pass drawing the direct family again, once as is and once with rewritten bindings
   auto run_frame = [&](RTAO::FilterMode filter_mode, uint32_t debug_mode) {
      // The "game" draws: two families on the immediate context, the third (and the view model) on a deferred one
      CHECK(capture.CaptureDraw(context, draw_direct), "capture (direct)");
      if (repeat_draws)
      {
         CHECK(capture.CaptureDraw(context, draw_direct), "repeated draw");
         RTAO::DrawInfo rewritten = draw_direct;
         rewritten.binding_version = 7;
         CHECK(capture.CaptureDraw(deferred, rewritten), "draw with rewritten per instance data");
      }
      CHECK(capture.CaptureDraw(context, draw_indexed), "capture (indexed)");
      CHECK(capture.CaptureDraw(context, draw_mvp), "capture (MVP only)");
      CHECK(capture.CaptureDraw(context, draw_cutout), "capture (cutout, wind)");
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
         last_tlas_cutouts = tlas.LastCutoutCount();
         if (!pass.Record(list, info, top, settings, retirer, err, &timer))
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
   printf("foliage: wind draws %llu (%llu instances), cutout draws %llu (%llu instances), cutouts read back last frame %llu, in the TLAS %u\n", (unsigned long long)stats.wind_draws,
      (unsigned long long)stats.wind_instances, (unsigned long long)stats.cutout_draws, (unsigned long long)stats.cutout_instances, (unsigned long long)stats.frame_cutout_instances, last_tlas_cutouts);
   CHECK(stats.wind_draws == 6 && stats.cutout_draws == 6, "one wind/cutout draw per frame must be counted");
   CHECK(stats.frame_cutout_instances == 1 && last_tlas_cutouts == 1, "the cutout box must be in the TLAS as a cutout");

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
   CHECK(timer.LastSpanCount() == 2 && timer.LastSpanMs(0) > 0.0 && std::fabs(timer.LastSpanMs(0) + timer.LastSpanMs(1) - timer.LastMs()) < 1e-6,
      "GPU timer: the pass marks the end of the trace (2 spans that add up, got %u)", timer.LastSpanCount());

   // Dedup: a repeat of a draw in the same frame (same mesh and bindings) is skipped, one whose per instance data was
   // rewritten in between (another binding version) is captured
   {
      const RTAO::CaptureStats before = capture.GetStats();
      repeat_draws = true;
      for (int i = 0; i < 4; i++)
         raw = run_frame(RTAO::FilterMode::Off, 0);
      repeat_draws = false;
      const RTAO::CaptureStats after = capture.GetStats();
      printf("dedup: %llu repeated draws skipped in 4 frames, %llu captured, TLAS instances %llu\n", (unsigned long long)(after.dedup_draws - before.dedup_draws),
         (unsigned long long)(after.draws_captured - before.draws_captured), (unsigned long long)after.tlas_instances);
      CHECK(after.dedup_draws - before.dedup_draws == 4, "one repeated draw per frame skipped");
      CHECK(after.draws_captured - before.draws_captured == 4 * 7, "6 draws + the rewritten one captured per frame");
      CHECK(after.tlas_instances == kBoxCount, "same TLAS (got %llu)", (unsigned long long)after.tlas_instances);
   }

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

   // Cutouts: the ground next to the alpha tested box at opacity 0 (as if the box weren't there), 0.5 and 1 (solid).
   // Averaged over frames (the per crossing coin flip changes every frame, like the ray directions).
   auto is_near_cutout = [&](UINT i) { return hit_id[i] == 0 && BoxDistance(hit_pos[i], kBoxes[kCutoutBox]) < 0.15f; };
   const float opacities[3] = { 0.f, 0.5f, 1.f };
   float near_cutout[3] = {}, gpu_ms[3] = {};
   for (int o = 0; o < 3; o++)
   {
      settings.cutout_opacity = opacities[o];
      double sum = 0, ms = 0;
      int count = 0, timed = 0;
      for (int frame = 0; frame < 16; frame++)
      {
         std::vector<float> ao = run_frame(RTAO::FilterMode::Off, 0);
         sum += average(ao, is_near_cutout, count);
         if (frame >= 4) // The timer reads back a few frames late
         {
            ms += timer.LastMs();
            timed++;
         }
      }
      near_cutout[o] = float(sum / 16);
      gpu_ms[o] = float(ms / timed);
      CHECK(count > 5, "pixels next to the cutout box (%d)", count);
   }
   settings.cutout_opacity = 0.5f;
   printf("next to the cutout box (raw AO, 16 frames): opacity 0 %.3f, 0.5 %.3f, 1 %.3f; RT pass GPU time %.3f / %.3f / %.3f ms\n", near_cutout[0], near_cutout[1], near_cutout[2], gpu_ms[0], gpu_ms[1], gpu_ms[2]);
   CHECK(near_cutout[0] > 0.97f, "opacity 0 must leave the cutout box out");
   CHECK(near_cutout[2] < 0.8f, "opacity 1 must make the cutout box solid");
   CHECK(near_cutout[1] < near_cutout[0] - 0.03f && near_cutout[1] > near_cutout[2] + 0.03f, "opacity 0.5 must occlude between no box and a solid box");

   // Fast cutouts (one coin per ray, no candidate loop): also partial
   {
      settings.cutout_mode = 2;
      double sum = 0;
      int count = 0;
      for (int frame = 0; frame < 16; frame++)
         sum += average(run_frame(RTAO::FilterMode::Off, 0), is_near_cutout, count);
      settings.cutout_mode = 0;
      const float fast = float(sum / 16);
      printf("fast cutouts, opacity 0.5: next to the cutout box %.3f\n", fast);
      CHECK(fast < near_cutout[0] - 0.03f && fast > near_cutout[2] + 0.03f, "fast cutouts at 0.5 must occlude between no box and a solid box");
   }

   // Quality Low (half resolution trace + upsample in the denoiser) converges to about the full resolution result
   {
      settings.half_res = false;
      settings.reset_history = true;
      std::vector<float> full, half;
      for (int frame = 0; frame < 32; frame++)
      {
         full = run_frame(RTAO::FilterMode::Temporal, 0);
         settings.reset_history = false;
      }
      settings.half_res = true;
      settings.reset_history = true;
      for (int frame = 0; frame < 32; frame++)
      {
         half = run_frame(RTAO::FilterMode::Temporal, 0);
         settings.reset_history = false;
      }
      settings.half_res = false;
      int count = 0;
      const float full_corner = average(full, is_corner, count), half_corner = average(half, is_corner, count);
      const float full_under = average(full, is_under_floating, count), half_under = average(half, is_under_floating, count);
      const float half_open = average(half, is_open_ground, count), half_sky = average(half, is_sky, count);
      double diff = 0;
      int geometry = 0;
      bool finite = true;
      for (UINT i = 0; i < W * H; i++)
      {
         finite = finite && std::isfinite(half[i]) && half[i] >= 0.f && half[i] <= 1.f;
         if (hit_id[i] >= 0)
         {
            diff += std::fabs(half[i] - full[i]);
            geometry++;
         }
      }
      printf("quality low (half res, 32 frames): wall corner %.3f (full res %.3f), under the floating box %.3f (%.3f), open ground %.3f, sky %.3f, mean abs diff to full res %.4f\n", half_corner, full_corner,
         half_under, full_under, half_open, half_sky, diff / std::max(geometry, 1));
      CHECK(finite, "half resolution AO must be finite and in [0, 1]");
      CHECK(std::fabs(half_corner - full_corner) < 0.06f && std::fabs(half_under - full_under) < 0.06f, "half resolution must match full resolution in the corners");
      CHECK(half_open > 0.97f && half_sky > 0.999f, "half resolution: open ground and sky unoccluded");
      CHECK(diff / std::max(geometry, 1) < 0.03, "half resolution close to full resolution overall");
   }

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   timer.Shutdown();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   capture.Shutdown();
   deferred->Release();
   for (IUnknown* p : std::initializer_list<IUnknown*>{ vb, ib, direct_buffer, direct_srv, direct_cb_buffer, indexed_buffer, indexed_srv, index_buffer, index_srv, indexed_cb_buffer, model_cb, cutout_cb, view_model_cb, mvp_cb, per_view_cb, depth_srv, depth_texture })
      p->Release();
}

// ---- Per frame meshes: skinned draws (bone palette + instance transform replayed by skin_vertices_cs) and a DYNAMIC
// vertex buffer, through the real D3D11 compute path, built/refit as current-frame-only BLASes

// Bone rows mapping the unit cube to "box" minus "offset" (the instance/model transform adds the offset back)
static void BoneRows(const Box& box, V3 offset, float out[12])
{
   const float rows[12] = { box.max.x - box.min.x, 0, 0, box.min.x - offset.x, 0, box.max.y - box.min.y, 0, box.min.y - offset.y, 0, 0, box.max.z - box.min.z, box.min.z - offset.z };
   std::memcpy(out, rows, sizeof(rows));
}

static uint16_t ToHalf(float f)
{
   // Exact for the small integers used here
   uint32_t bits;
   std::memcpy(&bits, &f, 4);
   if ((bits & 0x7FFFFFFF) == 0)
      return uint16_t(bits >> 16);
   const uint32_t sign = (bits >> 16) & 0x8000;
   const int32_t exponent = int32_t((bits >> 23) & 0xFF) - 127 + 15;
   const uint32_t mantissa = (bits >> 13) & 0x3FF;
   return uint16_t(sign | (uint32_t(exponent) << 10) | mantissa);
}

static void TestFrameMeshes(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps)
{
   std::string error;
   const UINT W = 320, H = 180;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());

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

   // Scene: 0 ground (static capture), A/B skinned draw 1 (direct instance buffer, 4 bones, game indices),
   // C/D skinned draw 2 (MVP only, 8 bones, half float positions, triangle soup), E dynamic vertex buffer (cbuffer rows).
   // Pose 2 moves B (a bone) and E (the dynamic vertices): it must show up in the very frame it is drawn.
   enum { kGround, kA, kB, kC, kD, kE, kSceneBoxes };
   Box scene[2][kSceneBoxes] = {};
   scene[0][kGround] = kBoxes[0];
   scene[0][kA] = { { -3, 0, -1 }, { -2, 1, 0 } };
   scene[0][kB] = { { -3, 1.5f, -1 }, { -2, 2, 0 } };
   scene[0][kC] = { { 2, 0, -1 }, { 3, 1, 0 } };
   scene[0][kD] = { { 2, 1.5f, -1 }, { 3, 2, 0 } };
   scene[0][kE] = { { -0.5f, 0, 2 }, { 0.5f, 1, 3 } };
   std::memcpy(scene[1], scene[0], sizeof(scene[0]));
   scene[1][kB] = { { -3, 1.5f, 0.5f }, { -2, 2, 1.5f } };
   scene[1][kE] = { { 0.5f, 0, 2 }, { 1.5f, 1, 3 } };

   struct Reference
   {
      std::vector<float> depth;
      std::vector<int> hit_id;
      std::vector<V3> hit_pos;
   };
   auto make_reference = [&](const Box* boxes) {
      Reference ref;
      ref.depth.assign(W * H, 0.f);
      ref.hit_id.assign(W * H, -1);
      ref.hit_pos.resize(W * H);
      for (UINT y = 0; y < H; y++)
         for (UINT x = 0; x < W; x++)
         {
            float v[4] = { (x + 0.5f) / W * 2.f - 1.f, 1.f - (y + 0.5f) / H * 2.f, 0.5f, 1.f }, w[4] = {};
            for (int c = 0; c < 4; c++)
               for (int k = 0; k < 4; k++)
                  w[c] += v[k] * inv_view_proj(k, c);
            const V3 dir = Normalize(Sub({ w[0] / w[3], w[1] / w[3], w[2] / w[3] }, eye));
            float best = 1e30f;
            for (int b = 0; b < kSceneBoxes; b++)
            {
               float t;
               if (RayBox(eye, dir, boxes[b], t) && t < best)
               {
                  best = t;
                  ref.hit_id[y * W + x] = b;
               }
            }
            if (ref.hit_id[y * W + x] < 0)
               continue;
            const V3 p = { eye.x + dir.x * best, eye.y + dir.y * best, eye.z + dir.z * best };
            ref.hit_pos[y * W + x] = p;
            float clip[4] = {};
            const float pv[4] = { p.x, p.y, p.z, 1.f };
            for (int c = 0; c < 4; c++)
               for (int k = 0; k < 4; k++)
                  clip[c] += pv[k] * view_proj(k, c);
            ref.depth[y * W + x] = clip[2] / clip[3];
         }
      return ref;
   };
   const Reference refs[2] = { make_reference(scene[0]), make_reference(scene[1]) };
   D3D11_TEXTURE2D_DESC depth_desc = { W, H, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   ID3D11Texture2D* depth_texture = nullptr;
   device->CreateTexture2D(&depth_desc, nullptr, &depth_texture);
   ID3D11ShaderResourceView* depth_srv = nullptr;
   device->CreateShaderResourceView(depth_texture, nullptr, &depth_srv);

   // Mesh of the skinned draws: two unit cubes (16 vertices, 72 indices), the second one on other bones
   std::vector<uint16_t> two_cubes_indices(72);
   for (int i = 0; i < 72; i++)
      two_cubes_indices[i] = uint16_t(kCubeIndices[i % 36] + (i >= 36 ? 8 : 0));

   // Skinned draw 1: stream 0 = uv + float3 position (stride 20), stream 1 = R8G8B8A8_UINT indices + R8G8B8A8_UNORM
   // weights (stride 8). Cube 0: bones 0 and 1 (identical) blended 128/255 + 127/255; cube 1: bone 2. cb_mtxoffset 2.
   struct Vertex1
   {
      float uv[2];
      float pos[3];
   };
   std::vector<Vertex1> vertices1(16);
   std::vector<uint8_t> skin1(16 * 8, 0);
   for (int i = 0; i < 16; i++)
   {
      vertices1[i] = { { 0, 0 }, { kCubeVertices[(i % 8) * 3], kCubeVertices[(i % 8) * 3 + 1], kCubeVertices[(i % 8) * 3 + 2] } };
      uint8_t* s = &skin1[i * 8];
      if (i < 8)
      {
         s[0] = 0; s[1] = 1; s[4] = 128; s[5] = 127;
      }
      else
      {
         s[0] = 2; s[4] = 255;
      }
   }
   ID3D11Buffer* vb1 = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, vertices1.data(), UINT(vertices1.size() * sizeof(Vertex1)));
   ID3D11Buffer* skin_vb1 = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, skin1.data(), UINT(skin1.size()));
   ID3D11Buffer* ib1 = MakeBuffer(device, D3D11_BIND_INDEX_BUFFER, two_cubes_indices.data(), 72 * 2);

   // Skinned draw 2: one stream, stride 56: half4 position @0, R16G16B16A16_UINT indices0 @8 and indices1 @16, float4
   // weights0 @24 and weights1 @40. Cube 0: 0.25 + 0.25 on bone 0 and 0.5 on bone 1 (identical), cube 1: bone 2.
   // 4 unrelated vertices in front (base vertex 4) and 200 after: more vertices than indices, so it goes as a soup.
   const uint32_t stride2 = 56;
   std::vector<uint8_t> vertices2((4 + 16 + 200) * stride2, 0x3C);
   for (int i = 0; i < 16; i++)
   {
      uint8_t* v = &vertices2[(4 + i) * stride2];
      const uint16_t half[4] = { ToHalf(kCubeVertices[(i % 8) * 3]), ToHalf(kCubeVertices[(i % 8) * 3 + 1]), ToHalf(kCubeVertices[(i % 8) * 3 + 2]), ToHalf(1.f) };
      std::memcpy(v, half, 8);
      uint16_t idx0[4] = { 0, 0, 0, 0 }, idx1[4] = { 1, 0, 0, 0 };
      float w0[4] = { 0.25f, 0.25f, 0, 0 }, w1[4] = { 0.5f, 0, 0, 0 };
      if (i >= 8)
      {
         idx0[0] = 2;
         w0[0] = 1.f;
         w0[1] = 0.f;
         w1[0] = 0.f;
      }
      std::memcpy(v + 8, idx0, 8);
      std::memcpy(v + 16, idx1, 8);
      std::memcpy(v + 24, w0, 16);
      std::memcpy(v + 40, w1, 16);
   }
   ID3D11Buffer* vb2 = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, vertices2.data(), UINT(vertices2.size()));
   std::vector<uint16_t> ib2_data(6 + 72, 1);
   std::memcpy(&ib2_data[6], two_cubes_indices.data(), 72 * 2);
   ID3D11Buffer* ib2 = MakeBuffer(device, D3D11_BIND_INDEX_BUFFER, ib2_data.data(), UINT(ib2_data.size() * 2));

   // Bone palette (Buffer<float4>, 3 rows per bone): [0..1] garbage, [2..4] draw 1 (A, A, B) relative to its instance
   // offset (+0.5, 0, 0), [5..7] the view model (a huge box), [8..10] draw 2 (C, C, D) relative to its model offset
   // (0, 0, 0.5)
   const V3 offset1 = { 0.5f, 0, 0 }, offset2 = { 0, 0, 0.5f };
   auto make_palette = [&](int pose) {
      std::vector<float> palette(11 * 12, 1000.f);
      BoneRows(scene[pose][kA], offset1, &palette[2 * 12]);
      BoneRows(scene[pose][kA], offset1, &palette[3 * 12]);
      BoneRows(scene[pose][kB], offset1, &palette[4 * 12]);
      for (int b = 5; b < 8; b++)
         BoneRows({ { -50, -50, -50 }, { 50, 50, 50 } }, offset1, &palette[b * 12]);
      BoneRows(scene[pose][kC], offset2, &palette[8 * 12]);
      BoneRows(scene[pose][kC], offset2, &palette[9 * 12]);
      BoneRows(scene[pose][kD], offset2, &palette[10 * 12]);
      return palette;
   };
   std::vector<float> palette = make_palette(0);
   ID3D11Buffer* bones_buffer = MakeBuffer(device, D3D11_BIND_SHADER_RESOURCE, palette.data(), UINT(palette.size() * 4), D3D11_USAGE_DEFAULT);
   D3D11_SHADER_RESOURCE_VIEW_DESC bones_desc = {};
   bones_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
   bones_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
   bones_desc.Buffer.NumElements = UINT(palette.size() / 4);
   ID3D11ShaderResourceView* bones_srv = nullptr;
   device->CreateShaderResourceView(bones_buffer, &bones_desc, &bones_srv);

   // Draw 1 instance data: direct instance buffer (stride 96, rows @48), PerInstanceCB: cb_mtxoffset @0,
   // cb_instancebufferoffset @16, cb_modeldepthhack @20
   std::vector<float> direct(2 * 24, 0.f);
   const float instance_rows[12] = { 1, 0, 0, offset1.x, 0, 1, 0, offset1.y, 0, 0, 1, offset1.z };
   std::memcpy(&direct[24 + 12], instance_rows, sizeof(instance_rows));
   ID3D11Buffer* direct_buffer = MakeBuffer(device, D3D11_BIND_SHADER_RESOURCE, direct.data(), UINT(direct.size() * 4), D3D11_USAGE_IMMUTABLE, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 96);
   ID3D11ShaderResourceView* direct_srv = nullptr;
   device->CreateShaderResourceView(direct_buffer, nullptr, &direct_srv);
   auto make_cb1 = [&](uint32_t mtx_offset, float depth_hack) {
      uint32_t cb[16] = {};
      cb[0] = mtx_offset;
      cb[16 / 4] = 1; // Instance 1 of the buffer
      std::memcpy(&cb[20 / 4], &depth_hack, 4);
      return MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb, sizeof(cb));
   };
   ID3D11Buffer* cb1 = make_cb1(2, 1.f);
   ID3D11Buffer* cb_view_model = make_cb1(5, 0.25f);

   // Draw 2: MVP only (cb_mtxoffset @0, cb_mvpmatrixz @16, y @32, x @48, w @64, depth hack @84), PerViewCB VP @32
   ID3D11Buffer* cb2 = nullptr;
   ID3D11Buffer* per_view_cb = nullptr;
   {
      Mat4 model;
      model(0, 0) = model(1, 1) = model(2, 2) = model(3, 3) = 1.f;
      model(3, 0) = offset2.x;
      model(3, 1) = offset2.y;
      model(3, 2) = offset2.z;
      const Mat4 mvp = Mul(model, view_proj);
      std::vector<float> cb(96 / 4, 0.f);
      reinterpret_cast<uint32_t*>(cb.data())[0] = 8;
      const uint32_t column_offsets[4] = { 48, 32, 16, 64 };
      for (int c = 0; c < 4; c++)
         for (int r = 0; r < 4; r++)
            cb[column_offsets[c] / 4 + r] = mvp(r, c);
      cb[84 / 4] = 1.f;
      cb2 = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb.data(), 96);
      std::vector<float> view(112 / 4, 0.f);
      std::memcpy(&view[32 / 4], view_proj.m, 64);
      per_view_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, view.data(), 112);
   }

   // Dynamic mesh E: DYNAMIC float3 vertex buffer rewritten every frame (box corners directly), identity cbuffer rows
   // (z @160, y @176, x @192, depth hack @464), the static cube index buffer
   D3D11_BUFFER_DESC dynamic_desc = { 8 * 12, D3D11_USAGE_DYNAMIC, D3D11_BIND_VERTEX_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
   ID3D11Buffer* dynamic_vb = nullptr;
   device->CreateBuffer(&dynamic_desc, nullptr, &dynamic_vb);
   ID3D11Buffer* cube_ib = MakeBuffer(device, D3D11_BIND_INDEX_BUFFER, std::vector<uint16_t>(kCubeIndices, kCubeIndices + 36).data(), 72);
   ID3D11Buffer* identity_cb = nullptr;
   ID3D11Buffer* ground_cb = nullptr;
   {
      std::vector<float> cb(480 / 4, 0.f);
      cb[192 / 4] = 1.f;
      cb[176 / 4 + 1] = 1.f;
      cb[160 / 4 + 2] = 1.f;
      cb[464 / 4] = 1.f;
      identity_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb.data(), 480);
      float t[12];
      BoxTransform(kBoxes[0], t);
      std::memcpy(&cb[192 / 4], &t[0], 16);
      std::memcpy(&cb[176 / 4], &t[4], 16);
      std::memcpy(&cb[160 / 4], &t[8], 16);
      ground_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb.data(), 480);
   }
   ID3D11Buffer* cube_vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, kCubeVertices, sizeof(kCubeVertices));

   // Vertex shader descriptions, as ReflectVertexShader produces them for the game's shaders
   RTAO::VertexShaderInfo skinned1_vs, skinned2_vs, cb_vs;
   skinned1_vs.source = RTAO::MatrixSource::DirectInstanceBuffer;
   skinned1_vs.skip_reason = nullptr;
   skinned1_vs.instance_cb_slot = 2;
   skinned1_vs.instance_offset_cb = 16;
   skinned1_vs.depth_hack_cb = 20;
   skinned1_vs.instance_buffer_slot = 1;
   skinned1_vs.instance_buffer_stride = 96;
   skinned1_vs.record_row0 = 48;
   skinned1_vs.skinned = true;
   skinned1_vs.skin_sets = 1;
   skinned1_vs.bones_slot = 0;
   skinned1_vs.mtx_offset_cb = 0;
   skinned2_vs.source = RTAO::MatrixSource::MvpConstantBuffer;
   skinned2_vs.skip_reason = nullptr;
   skinned2_vs.instance_cb_slot = 2;
   skinned2_vs.mvp_cb[0] = 48;
   skinned2_vs.mvp_cb[1] = 32;
   skinned2_vs.mvp_cb[2] = 16;
   skinned2_vs.mvp_cb[3] = 64;
   skinned2_vs.depth_hack_cb = 84;
   skinned2_vs.per_view_cb_slot = 1;
   skinned2_vs.view_proj_cb = 32;
   skinned2_vs.skinned = true;
   skinned2_vs.skin_sets = 2;
   skinned2_vs.bones_slot = 0;
   skinned2_vs.mtx_offset_cb = 0;
   cb_vs.source = RTAO::MatrixSource::ConstantBuffer;
   cb_vs.skip_reason = nullptr;
   cb_vs.instance_cb_slot = 2;
   cb_vs.row_cb[0] = 192;
   cb_vs.row_cb[1] = 176;
   cb_vs.row_cb[2] = 160;
   cb_vs.depth_hack_cb = 464;

   RTAO::DrawInfo draw1;
   draw1.vs = &skinned1_vs;
   draw1.vertex_buffer = vb1;
   draw1.vertex_stride = sizeof(Vertex1);
   draw1.position_format = DXGI_FORMAT_R32G32B32_FLOAT;
   draw1.position_offset = 8;
   draw1.index_buffer = ib1;
   draw1.index_format = DXGI_FORMAT_R16_UINT;
   draw1.index_count = 72;
   draw1.instance_cb = cb1;
   draw1.instance_buffer = direct_srv;
   draw1.bones = bones_srv;
   draw1.skin_indices[0] = { skin_vb1, 8, 0, 0, DXGI_FORMAT_R8G8B8A8_UINT };
   draw1.skin_weights[0] = { skin_vb1, 8, 0, 4, DXGI_FORMAT_R8G8B8A8_UNORM };
   RTAO::DrawInfo draw_view_model = draw1;
   draw_view_model.instance_cb = cb_view_model;
   RTAO::DrawInfo draw_instanced = draw1;
   draw_instanced.instance_count = 2;
   RTAO::DrawInfo draw2;
   draw2.vs = &skinned2_vs;
   draw2.vertex_buffer = vb2;
   draw2.vertex_stride = stride2;
   draw2.position_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
   draw2.position_offset = 0;
   draw2.index_buffer = ib2;
   draw2.index_format = DXGI_FORMAT_R16_UINT;
   draw2.index_count = 72;
   draw2.first_index = 6;
   draw2.base_vertex = 4;
   draw2.instance_cb = cb2;
   draw2.per_view_cb = per_view_cb;
   draw2.bones = bones_srv;
   draw2.skin_indices[0] = { vb2, stride2, 0, 8, DXGI_FORMAT_R16G16B16A16_UINT };
   draw2.skin_indices[1] = { vb2, stride2, 0, 16, DXGI_FORMAT_R16G16B16A16_UINT };
   draw2.skin_weights[0] = { vb2, stride2, 0, 24, DXGI_FORMAT_R32G32B32A32_FLOAT };
   draw2.skin_weights[1] = { vb2, stride2, 0, 40, DXGI_FORMAT_R32G32B32A32_FLOAT };
   RTAO::DrawInfo draw_dynamic;
   draw_dynamic.vs = &cb_vs;
   draw_dynamic.vertex_buffer = dynamic_vb;
   draw_dynamic.vertex_stride = 12;
   draw_dynamic.position_format = DXGI_FORMAT_R32G32B32_FLOAT;
   draw_dynamic.index_buffer = cube_ib;
   draw_dynamic.index_format = DXGI_FORMAT_R16_UINT;
   draw_dynamic.index_count = 36;
   draw_dynamic.instance_cb = identity_cb;
   RTAO::DrawInfo draw_ground = draw_dynamic;
   draw_ground.vertex_buffer = cube_vb;
   draw_ground.instance_cb = ground_cb;

   ID3DBlob* cs_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "capture_instances_cs", "cs_5_0");
   ID3D11ComputeShader* capture_cs = nullptr;
   device->CreateComputeShader(cs_code->GetBufferPointer(), cs_code->GetBufferSize(), nullptr, &capture_cs);
   cs_code->Release();
   cs_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "skin_vertices_cs", "cs_5_0");
   ID3D11ComputeShader* skin_cs = nullptr;
   device->CreateComputeShader(cs_code->GetBufferPointer(), cs_code->GetBufferSize(), nullptr, &skin_cs);
   cs_code->Release();

   RTAO::GeometryCapture capture;
   CHECK(capture.Init(device, interop, capture_cs, skin_cs, error), "GeometryCapture::Init: %s", error.c_str());
   CHECK(capture.IsSkinningAvailable(), "skinning resources: %s", capture.skinning_error.c_str());
   capture_cs->Release();
   skin_cs->Release();
   ID3D11DeviceContext* deferred = nullptr;
   device->CreateDeferredContext(0, &deferred);

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

   uint32_t tlas_count = 0;
   bool repeat_draws = false; // Draws two of the per frame meshes a second time (another pass): skipped as duplicates
   bool skip_dynamic = false, skip_skinned_a = false; // The game doesn't draw them this frame
   auto run_frame = [&](int pose, uint32_t debug_mode) {
      // The pose: bones (DEFAULT buffer) and the dynamic vertices (Map discard), as the game would update them
      palette = make_palette(pose);
      context->UpdateSubresource(bones_buffer, 0, nullptr, palette.data(), 0, 0);
      D3D11_MAPPED_SUBRESOURCE mapped = {};
      if (SUCCEEDED(context->Map(dynamic_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
      {
         const Box& e = scene[pose][kE];
         float* v = static_cast<float*>(mapped.pData);
         for (int i = 0; i < 8; i++)
         {
            v[i * 3 + 0] = kCubeVertices[i * 3 + 0] ? e.max.x : e.min.x;
            v[i * 3 + 1] = kCubeVertices[i * 3 + 1] ? e.max.y : e.min.y;
            v[i * 3 + 2] = kCubeVertices[i * 3 + 2] ? e.max.z : e.min.z;
         }
         context->Unmap(dynamic_vb, 0);
      }
      context->UpdateSubresource(depth_texture, 0, nullptr, refs[pose].depth.data(), W * 4, 0);

      CHECK(capture.CaptureDraw(context, draw_ground), "capture (ground)");
      if (!skip_skinned_a)
         CHECK(capture.CaptureDraw(context, draw1), "capture (skinned, direct instance buffer)");
      CHECK(capture.CaptureDraw(context, draw_view_model), "capture (skinned view model: dispatched, writes inactive triangles)");
      CHECK(!capture.CaptureDraw(context, draw_instanced), "an instanced skinned draw must be skipped");
      if (!skip_dynamic)
         CHECK(capture.CaptureDraw(context, draw_dynamic), "capture (dynamic vertex buffer)");
      CHECK(capture.CaptureDraw(deferred, draw2), "capture (skinned, MVP only, deferred)");
      if (repeat_draws)
      {
         CHECK(capture.CaptureDraw(context, draw1), "repeated skinned draw");
         CHECK(capture.CaptureDraw(deferred, draw_dynamic), "repeated dynamic draw (other context)");
      }
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
      ID3D11ShaderResourceView* null_srv = nullptr;
      context->PSSetShaderResources(0, 1, &null_srv);
      rtv->Release();

      settings.filter_mode = RTAO::FilterMode::Off;
      settings.debug_mode = debug_mode;
      CHECK(interop.Run(context, [&](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& err) {
         retirer.Collect(info.completed_fence_value);
         if (!capture.Record(list, info, blases, tlas, retirer, 64, err))
            return false;
         const RTAO::BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         if (!tlas.Build(ctx, blases, info.frame_slot, top, tlas_count, err))
            return false;
         return pass.Record(list, info, top, settings, retirer, err);
      }, error), "Run (frame meshes): %s", error.c_str());
      capture.OnPresent();
      capture.PollCounters(context);
      ID3D11ShaderResourceView* ao_srv = interop.GetAOSRV();
      UINT w = 0, h = 0;
      std::vector<float> ao = ReadAO(device, context, ao_srv, w, h);
      ao_srv->Release();
      return ao;
   };

   std::vector<float> raw;
   for (int i = 0; i < 6; i++) // The ground's matrix is read back 3 frames late
      raw = run_frame(0, 0);
   const RTAO::CaptureStats stats = capture.GetStats();
   printf("frame meshes: skinned %llu captured / %llu instanced skipped, dynamic %llu captured, last frame %llu draws %llu vertices, soup %llu, indexed %llu, BLAS builds %llu refits %llu failed %llu (%llu live), TLAS %u\n",
      (unsigned long long)stats.skinned_captured, (unsigned long long)stats.skinned_instanced, (unsigned long long)stats.dynamic_captured, (unsigned long long)stats.frame_draws,
      (unsigned long long)stats.frame_vertices, (unsigned long long)stats.frame_soup, (unsigned long long)stats.frame_indexed, (unsigned long long)stats.frame_blas_builds,
      (unsigned long long)stats.frame_blas_refits, (unsigned long long)stats.frame_blas_failed, (unsigned long long)stats.frame_blas_count, tlas_count);
   // Per frame: draw 1, the view model (its triangles are inactive, the CPU can't know it is one), draw 2, the dynamic mesh
   CHECK(stats.skinned_captured == 18 && stats.skinned_instanced == 6 && stats.dynamic_captured == 6, "per frame: 3 skinned draws captured, 1 instanced skipped, 1 dynamic");
   CHECK(stats.frame_draws == 4 && stats.frame_vertices == 16 + 16 + 72 + 8, "last frame: 4 per frame meshes (2 x 16 indexed skinned, 72 soup, 8 dynamic)");
   CHECK(stats.frame_soup == 6 && stats.frame_indexed == 18, "draw 2 is a soup, the others indexed");
   CHECK(stats.frame_blas_builds == 4 && stats.frame_blas_refits == 20 && stats.frame_blas_failed == 0, "built once, refit after");
   CHECK(tlas_count == 5, "TLAS: the ground + 4 per frame meshes (got %u)", tlas_count);

   const Reference& ref = refs[0];
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
   auto near_box = [&](int b) { return [&, b](UINT i) { return ref.hit_id[i] == kGround && BoxDistance(ref.hit_pos[i], scene[0][b]) < 0.15f; }; };
   auto is_open_ground = [&](UINT i) {
      if (ref.hit_id[i] != kGround)
         return false;
      for (int b = 1; b < kSceneBoxes; b++)
         if (BoxDistance(ref.hit_pos[i], scene[0][b]) < 2.f)
            return false;
      return true;
   };
   int c[4];
   const float open = average(raw, is_open_ground, c[0]), near_a = average(raw, near_box(kA), c[1]), near_c = average(raw, near_box(kC), c[2]), near_e = average(raw, near_box(kE), c[3]);
   printf("frame meshes AO (raw): open ground %.3f, next to A (skinned, direct) %.3f (%d px), next to C (skinned, MVP, soup) %.3f (%d), next to E (dynamic) %.3f (%d)\n", open, near_a, c[1], near_c, c[2], near_e, c[3]);
   CHECK(c[0] > 50 && open > 0.97f, "open ground unoccluded (the skinned view model must be left out)");
   CHECK(c[1] > 5 && near_a < 0.8f, "skinned box A occludes");
   CHECK(c[2] > 5 && near_c < 0.8f, "skinned box C occludes");
   CHECK(c[3] > 5 && near_e < 0.8f, "dynamic box E occludes");

   // Coverage: the per frame meshes line up with the depth, also in the frame the pose changes
   for (int pose = 0; pose < 2; pose++)
   {
      const std::vector<float> coverage = run_frame(pose, 3);
      int hits = 0, matched = 0, frame_hits = 0, frame_matched = 0;
      for (UINT i = 0; i < W * H; i++)
         if (refs[pose].hit_id[i] >= 0)
         {
            const bool ok = coverage[i] > 0.99f;
            hits++;
            matched += ok;
            if (refs[pose].hit_id[i] != kGround)
            {
               frame_hits++;
               frame_matched += ok;
            }
         }
      printf("frame meshes coverage, pose %d: %d of %d geometry pixels match the depth (%.1f%%), per frame meshes %d of %d\n", pose, matched, hits, 100.0 * matched / std::max(hits, 1), frame_matched, frame_hits);
      CHECK(frame_hits > 500 && matched > hits * 0.98 && frame_matched > frame_hits * 0.98, "per frame meshes must line up with the depth (pose %d)", pose);
   }

   // Repeated draws in a frame: no second skinning dispatch, no second per frame BLAS
   {
      const RTAO::CaptureStats before = capture.GetStats();
      repeat_draws = true;
      run_frame(0, 0);
      repeat_draws = false;
      const RTAO::CaptureStats after = capture.GetStats();
      printf("frame meshes dedup: %llu repeated draws skipped, last frame %llu per frame meshes, TLAS %u\n", (unsigned long long)(after.dedup_frame_meshes - before.dedup_frame_meshes),
         (unsigned long long)after.frame_draws, tlas_count);
      // Same context: skipped at the draw. Other context: captured (the first one recorded may run after the AO pass and
      // be dropped), the pass keeps the first executed one
      CHECK(after.dedup_frame_meshes - before.dedup_frame_meshes == 1, "the repeated per frame draw on the same context must be skipped");
      CHECK(after.dynamic_captured - before.dynamic_captured == 2, "the repeated per frame draw on another context must be captured");
      CHECK(after.skinned_captured - before.skinned_captured == 3 && after.frame_draws == 4 && tlas_count == 5, "no duplicate per frame BLAS");
      // The trace of that frame (the pass ran before Present): 'D' for the same context repeat, 'd' for the other one
      run_frame(0, 0);
      int same_context = 0, pass_dropped = 0, in_pass = 0;
      for (const RTAO::FrameMeshEvent& e : capture.GetLastFrameMeshTrace())
      {
         same_context += e.event == 'D';
         pass_dropped += e.event == 'd';
         in_pass += e.event == 'p';
      }
      printf("frame mesh trace (no repeats): %d in the pass, %d same context repeats, %d dropped by the pass\n", in_pass, same_context, pass_dropped);
      CHECK(in_pass == 4 && same_context == 0 && pass_dropped == 0, "frame mesh trace");
   }

   // Frames the game doesn't draw a movable object (dynamic vertex buffer, e.g. an openable window out of view): it
   // stays in its last pose; a skinned character not drawn is gone at once (it moves)
   {
      // Skinned held for a few frames: A not drawn for 3 frames keeps its last pose
      skip_skinned_a = true;
      for (int i = 0; i < 3; i++)
         run_frame(0, 0);
      skip_skinned_a = false;
      const RTAO::CaptureStats held = capture.GetStats();
      printf("skinned A not drawn for 3 frames (hold %u): retained %llu, TLAS %u\n", capture.skinned_hold_frames.load(), (unsigned long long)held.frame_retained, tlas_count);
      CHECK(held.frame_retained == 1 && tlas_count == 5, "a skinned mesh missing a few frames must keep its last pose");
      run_frame(0, 0);
      CHECK(capture.GetStats().frame_retained == 0 && tlas_count == 5, "a skinned mesh drawn again is current again");
      capture.skinned_hold_frames = 0; // Applies at an entry's next draw
      run_frame(0, 0);
      skip_dynamic = true;
      skip_skinned_a = true;
      std::vector<float> ao;
      for (int i = 0; i < 3; i++)
         ao = run_frame(0, 0);
      skip_dynamic = false;
      skip_skinned_a = false;
      capture.skinned_hold_frames = 30;
      int count = 0;
      const float near_e_kept = average(ao, near_box(kE), count);
      const RTAO::CaptureStats kept = capture.GetStats();
      printf("frame meshes not drawn for 3 frames: dynamic E kept (AO next to it %.3f), retained %llu, TLAS %u\n", near_e_kept, (unsigned long long)kept.frame_retained, tlas_count);
      CHECK(count > 5 && near_e_kept < 0.8f, "a movable object not drawn must keep occluding in its last pose (%.3f)", near_e_kept);
      CHECK(kept.frame_retained == 1 && tlas_count == 4, "TLAS: ground + view model + C + retained E, skinned A gone (retained %llu, TLAS %u)", (unsigned long long)kept.frame_retained, tlas_count);
      // GPU counters (read back a few frames late): per frame meshes dispatched, none with a rejected matrix
      printf("frame mesh GPU counters: main view %llu (rejected %llu), shadow maps %llu (rejected %llu)\n", (unsigned long long)kept.gpu_main_frame, (unsigned long long)kept.gpu_main_frame_rejected,
         (unsigned long long)kept.gpu_shadow_frame, (unsigned long long)kept.gpu_shadow_frame_rejected);
      CHECK(kept.gpu_main_frame > 0 && kept.gpu_main_frame_rejected == 0 && kept.gpu_shadow_frame == 0, "frame mesh GPU counters");
      // Probe ring: vertex 0 of each per frame mesh draw, in world space (the dynamic box E's vertex 0 is its min corner)
      const auto probes = capture.GetProbes();
      int e_found = 0;
      for (const auto& probe : probes)
      {
         const Box& e = scene[0][kE];
         if (!(probe.flags & 2) && std::fabs(probe.pos[0] - e.min.x) < 1e-3f && std::fabs(probe.pos[1] - e.min.y) < 1e-3f && std::fabs(probe.pos[2] - e.min.z) < 1e-3f)
            e_found++;
      }
      printf("frame mesh probes: %zu, dynamic box E's vertex 0 found in %d\n", probes.size(), e_found);
      CHECK(!probes.empty() && e_found > 0, "probes must hold the dynamic mesh's world position");
   }

   // Skinned shadow map draws (the game draws many of its sun casters with skinned shaders): captured when enabled,
   // skipped when off
   {
      capture.OnPresent(); // A new frame: no duplicate of this frame's main view draw
      RTAO::DrawInfo shadow_draw1 = draw1;
      shadow_draw1.shadow_view = true;
      shadow_draw1.shadow_slice = 0;
      const RTAO::CaptureStats before = capture.GetStats();
      CHECK(capture.CaptureDraw(context, shadow_draw1), "a skinned shadow map draw must be captured");
      capture.shadow_skinned_enabled = false;
      capture.OnPresent();
      CHECK(!capture.CaptureDraw(context, shadow_draw1), "a skinned shadow map draw must be skipped when off");
      capture.shadow_skinned_enabled = true;
      const RTAO::CaptureStats after = capture.GetStats();
      printf("skinned shadow map draws: captured %llu, skipped when off %llu\n", (unsigned long long)(after.skinned_captured - before.skinned_captured),
         (unsigned long long)(after.shadow_skipped_skinned - before.shadow_skipped_skinned));
      CHECK(after.skinned_captured - before.skinned_captured == 1 && after.shadow_skipped_skinned - before.shadow_skipped_skinned == 1, "skinned shadow map draw counts");
      run_frame(0, 0);
   }

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   capture.Shutdown();
   deferred->Release();
   for (IUnknown* p : std::initializer_list<IUnknown*>{ vb1, skin_vb1, ib1, vb2, ib2, bones_buffer, bones_srv, direct_buffer, direct_srv, cb1, cb_view_model, cb2, per_view_cb, dynamic_vb, cube_ib, identity_cb, ground_cb, cube_vb, depth_srv, depth_texture })
      p->Release();
}

// ---- Texture exact cutouts: an alpha tested wall whose BC3 texture is a 2x2 alpha checker, captured through the real
// D3D11 path (material mask rendered by alpha_strip_cs into the shared strip, copied into the atlas; uv scale/bias and
// alpha reference read from the draw's material cbuffer by capture_instances_cs). The ground AO in front of it must follow
// the mask: occluded in front of the solid lower quadrant, open in front of the hole.

// A BC3 block (4x4 texels) of uniform alpha, black color
static void Bc3Block(uint8_t alpha, uint8_t out[16])
{
   std::memset(out, 0, 16);
   out[0] = alpha;
   out[1] = alpha; // a0 == a1: every index decodes to a0
}

// A BC3 64x64 texture with its mips whose alpha is a "cells" x "cells" checker (solid where (u cell + v cell) is odd,
// i.e. for 2 cells the lower left and upper right quadrants), black color
static void MakeCheckerBc3(ID3D11Device* device, UINT cells, ID3D11Texture2D*& texture, ID3D11ShaderResourceView*& srv)
{
   const UINT T = 64, mips = 7;
   std::vector<std::vector<uint8_t>> mip_data(mips);
   std::vector<D3D11_SUBRESOURCE_DATA> init(mips);
   for (UINT m = 0; m < mips; m++)
   {
      const UINT size = (std::max)(T >> m, 1u), blocks = (std::max)((size + 3) / 4, 1u);
      const bool resolved = size >= 4 * cells; // Smaller mips average the checker
      mip_data[m].resize(blocks * blocks * 16);
      for (UINT by = 0; by < blocks; by++)
         for (UINT bx = 0; bx < blocks; bx++)
         {
            const UINT cu = UINT((bx * 4 + 2.f) / float(blocks * 4) * cells), cv = UINT((by * 4 + 2.f) / float(blocks * 4) * cells);
            Bc3Block(resolved ? (((cu + cv) & 1) ? 255 : 0) : 128, &mip_data[m][(by * blocks + bx) * 16]);
         }
      init[m] = { mip_data[m].data(), blocks * 16, 0 };
   }
   D3D11_TEXTURE2D_DESC texture_desc = { T, T, mips, 1, DXGI_FORMAT_BC3_UNORM, { 1, 0 }, D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   HRESULT hr = device->CreateTexture2D(&texture_desc, init.data(), &texture);
   CHECK(SUCCEEDED(hr), "BC3 texture 0x%08lX", (unsigned long)hr);
   srv = nullptr;
   if (texture)
      device->CreateShaderResourceView(texture, nullptr, &srv);
}

static void TestCutoutTextures(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps)
{
   std::string error;
   const UINT W = 320, H = 180;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());

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

   // Depth: the ground only (the wall is only in the ray tracing scene)
   std::vector<float> depth(W * H, 0.f);
   std::vector<V3> hit_pos(W * H);
   std::vector<bool> ground(W * H, false);
   for (UINT y = 0; y < H; y++)
      for (UINT x = 0; x < W; x++)
      {
         float v[4] = { (x + 0.5f) / W * 2.f - 1.f, 1.f - (y + 0.5f) / H * 2.f, 0.5f, 1.f }, w[4] = {};
         for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
               w[c] += v[k] * inv_view_proj(k, c);
         const V3 dir = Normalize(Sub({ w[0] / w[3], w[1] / w[3], w[2] / w[3] }, eye));
         float t;
         if (!RayBox(eye, dir, kBoxes[0], t))
            continue;
         const V3 p = { eye.x + dir.x * t, eye.y + dir.y * t, eye.z + dir.z * t };
         hit_pos[y * W + x] = p;
         ground[y * W + x] = true;
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

   // The wall: x -3..3, y 0..2, z 0 (world space, identity model matrix). Positions in slot 0 (float3), TEXCOORD0 in slot 1
   // (R16G16_FLOAT): u = (x + 3) / 6, v = 1 - y / 2. 16 bit indices after 3 unrelated ones.
   const float positions[4 * 3] = { -3, 0, 0, 3, 0, 0, 3, 2, 0, -3, 2, 0 };
   const uint16_t uvs[4 * 2] = { ToHalf(0.f), ToHalf(1.f), ToHalf(1.f), ToHalf(1.f), ToHalf(1.f), ToHalf(0.f), ToHalf(0.f), ToHalf(0.f) };
   const uint16_t indices[3 + 6] = { 7, 7, 7, 0, 1, 2, 0, 2, 3 };
   ID3D11Buffer* position_vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, positions, sizeof(positions));
   ID3D11Buffer* uv_vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, uvs, sizeof(uvs));
   ID3D11Buffer* ib = MakeBuffer(device, D3D11_BIND_INDEX_BUFFER, indices, sizeof(indices));

   // The texture: BC3 64x64 with its mips, alpha 2x2 checker: solid where (u < 0.5) == (v >= 0.5), i.e. the lower left and
   // upper right quadrants of the wall
   ID3D11Texture2D* alpha_texture = nullptr;
   ID3D11ShaderResourceView* alpha_srv = nullptr;
   MakeCheckerBc3(device, 2, alpha_texture, alpha_srv);

   // Vertex shader: cbuffer rows (as in the game: z @160, y @176, x @192, depth hack @464). Pixel shader material cbuffer
   // like 0x048591C8: cb_layer0_scalebias @16, cb_defaultalphatest @32.
   std::vector<float> vs_cb(480 / 4, 0.f);
   const float rows[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
   std::memcpy(&vs_cb[192 / 4], &rows[0], 16);
   std::memcpy(&vs_cb[176 / 4], &rows[4], 16);
   std::memcpy(&vs_cb[160 / 4], &rows[8], 16);
   vs_cb[464 / 4] = 1.f;
   ID3D11Buffer* instance_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, vs_cb.data(), 480);
   auto make_material_cb = [&](float bias_u) {
      std::vector<float> cb(48 / 4, 0.f);
      cb[16 / 4 + 0] = 1.f;
      cb[16 / 4 + 1] = 1.f;
      cb[16 / 4 + 2] = bias_u;
      cb[32 / 4] = 0.5f;
      return MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, cb.data(), 48);
   };
   ID3D11Buffer* material_cb = make_material_cb(0.f);
   ID3D11Buffer* shifted_cb = make_material_cb(0.5f); // Shifts u by half the texture: the checker swaps sides

   RTAO::VertexShaderInfo cb_vs;
   cb_vs.source = RTAO::MatrixSource::ConstantBuffer;
   cb_vs.skip_reason = nullptr;
   cb_vs.instance_cb_slot = 2;
   cb_vs.row_cb[0] = 192;
   cb_vs.row_cb[1] = 176;
   cb_vs.row_cb[2] = 160;
   cb_vs.depth_hack_cb = 464;
   RTAO::PixelShaderInfo ps_info;
   ps_info.discard = ps_info.alpha_tested = ps_info.texture_exact = true;
   ps_info.alpha_slot = 0;
   ps_info.alpha_channel = 3;
   ps_info.material_cb_slot = 2;
   ps_info.alpha_ref_cb = 32;
   ps_info.uv_scalebias_cb = 16;

   RTAO::DrawInfo draw;
   draw.vs = &cb_vs;
   draw.vertex_buffer = position_vb;
   draw.vertex_stride = 12;
   draw.position_format = DXGI_FORMAT_R32G32B32_FLOAT;
   draw.index_buffer = ib;
   draw.index_format = DXGI_FORMAT_R16_UINT;
   draw.index_count = 6;
   draw.first_index = 3;
   draw.instance_cb = instance_cb;
   draw.cutout = true;
   draw.ps = &ps_info;
   draw.alpha_srv = alpha_srv;
   draw.material_cb = material_cb;
   draw.uv.buffer = uv_vb;
   draw.uv.stride = 4;
   draw.uv.format = DXGI_FORMAT_R16G16_FLOAT;

   ID3DBlob* cs_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "capture_instances_cs", "cs_5_0");
   ID3D11ComputeShader* capture_cs = nullptr;
   device->CreateComputeShader(cs_code->GetBufferPointer(), cs_code->GetBufferSize(), nullptr, &capture_cs);
   cs_code->Release();
   ID3DBlob* strip_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "alpha_strip_cs", "cs_5_0");
   ID3D11ComputeShader* strip_cs = nullptr;
   device->CreateComputeShader(strip_code->GetBufferPointer(), strip_code->GetBufferSize(), nullptr, &strip_cs);
   strip_code->Release();

   RTAO::GeometryCapture capture;
   CHECK(capture.Init(device, interop, capture_cs, nullptr, error), "GeometryCapture::Init: %s", error.c_str());
   CHECK(capture.InitCutouts(device, interop, strip_cs, error), "GeometryCapture::InitCutouts: %s", error.c_str());
   capture_cs->Release();
   strip_cs->Release();

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
   settings.filter_mode = RTAO::FilterMode::Off;
   settings.ray_count = 8;
   // RT sun shadows through the same cutouts (recorded after the AO when sun_on, as the addon does)
   RTSun::SunPass sun;
   RTSun::SunSettings sun_settings;
   sun_settings.casters_only = false; // The scene's instances are added as main view ones (no shadow map)
   bool sun_on = false;

   auto run_frame = [&]() {
      CHECK(capture.CaptureDraw(context, draw), "capture (textured cutout)");
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
         if (!pass.Record(list, info, top, settings, retirer, err, nullptr, &capture.GetCutoutBindings()))
            return false;
         timer.End(list, info.frame_slot);
         if (sun_on && !sun.Record(list, info, top, sun_settings, retirer, err, &capture.GetCutoutBindings()))
            return false;
         return true;
      }, error), "Run (cutout textures): %s", error.c_str());
      capture.OnPresent();
      ID3D11ShaderResourceView* ao_srv = interop.GetAOSRV();
      UINT w = 0, h = 0;
      std::vector<float> ao = ReadAO(device, context, ao_srv, w, h);
      ao_srv->Release();
      return ao;
   };

   // Ground pixels just in front of the wall, left (solid lower quadrant) and right (hole in the lower quadrant)
   auto in_front = [&](UINT i, bool left) {
      if (!ground[i])
         return false;
      const V3 p = hit_pos[i];
      return p.z > -0.35f && p.z < -0.05f && (left ? (p.x > -2.5f && p.x < -0.75f) : (p.x > 0.75f && p.x < 2.5f));
   };
   auto measure = [&](float& left, float& right, float& ms) {
      double sum[2] = {}, time = 0;
      int count[2] = {}, timed = 0;
      for (int frame = 0; frame < 16; frame++)
      {
         std::vector<float> ao = run_frame();
         for (UINT i = 0; i < W * H; i++)
            for (int side = 0; side < 2; side++)
               if (in_front(i, side == 0))
               {
                  sum[side] += ao[i];
                  count[side]++;
               }
         if (frame >= 4)
         {
            time += timer.LastMs();
            timed++;
         }
      }
      CHECK(count[0] > 16 * 20 && count[1] > 16 * 20, "pixels in front of the wall (%d / %d)", count[0] / 16, count[1] / 16);
      left = float(sum[0] / (std::max)(count[0], 1));
      right = float(sum[1] / (std::max)(count[1], 1));
      ms = float(time / (std::max)(timed, 1));
   };

   for (int i = 0; i < 6; i++) // The mask is copied at the first pass, the instance arrives 3 frames late
      run_frame();
   const RTAO::CutoutStats stats = capture.GetCutoutStats();
   printf("cutout textures: materials %llu (masked %llu), atlas slices %llu, records %llu, strips rendered %llu copied %llu, draws with a mask %llu\n", (unsigned long long)stats.materials,
      (unsigned long long)stats.materials_ready, (unsigned long long)stats.atlas_slices_used, (unsigned long long)stats.records, (unsigned long long)stats.strips_rendered, (unsigned long long)stats.strips_copied,
      (unsigned long long)stats.draws_exact);
   CHECK(stats.materials == 1 && stats.materials_ready == 1 && stats.atlas_slices_used == 1 && stats.strips_copied == 1, "one material with a mask");
   CHECK(stats.records == 1 && stats.draws_exact == 6 && stats.draws_no_material == 0, "one record, every draw with a material");

   float exact_left, exact_right, exact_ms, shifted_left, shifted_right, shifted_ms, stochastic_left, stochastic_right, stochastic_ms;
   settings.cutout_opacity = 0.5f;
   settings.cutout_exact = true;
   measure(exact_left, exact_right, exact_ms);
   const RTAO::AOPass::CutoutCounters counters = pass.GetCutoutCounters();
   // The scale/bias comes from the draw's cbuffer: u + 0.5 swaps the checker's sides (a new record, same material)
   draw.material_cb = shifted_cb;
   for (int i = 0; i < 4; i++)
      run_frame();
   measure(shifted_left, shifted_right, shifted_ms);
   draw.material_cb = material_cb;
   for (int i = 0; i < 4; i++)
      run_frame();
   settings.cutout_exact = false;
   measure(stochastic_left, stochastic_right, stochastic_ms);
   settings.cutout_exact = true;
   // Quality Low (half resolution trace, upsampled by the filter) and fast cutouts (per ray coin): masks keep deciding
   float low_left, low_right, low_ms, fast_left, fast_right, fast_ms;
   settings.half_res = true;
   measure(low_left, low_right, low_ms);
   settings.half_res = false;
   settings.cutout_mode = 2;
   measure(fast_left, fast_right, fast_ms);
   const RTAO::AOPass::CutoutCounters fast_counters = pass.GetCutoutCounters();
   settings.cutout_mode = 0;
   const RTAO::CutoutStats after = capture.GetCutoutStats();
   printf("ground AO in front of the textured cutout wall (raw, 16 frames), left (solid below) / right (hole below): texture exact %.3f / %.3f, "
          "u shifted by 0.5 %.3f / %.3f, stochastic (opacity 0.5) %.3f / %.3f\n", exact_left, exact_right, shifted_left, shifted_right, stochastic_left, stochastic_right);
   printf("cutout candidates per frame: masked %llu (%llu solid), stochastic %llu; records %llu; RT pass GPU time: texture exact %.3f ms, stochastic %.3f ms\n", (unsigned long long)counters.exact_tests,
      (unsigned long long)counters.exact_hits, (unsigned long long)counters.fallback_tests, (unsigned long long)after.records, exact_ms, stochastic_ms);
   CHECK(exact_left < exact_right - 0.1f, "texture exact: the solid quadrant must occlude more than the hole");
   CHECK(shifted_right < shifted_left - 0.1f, "the uv scale/bias must move the mask");
   CHECK(std::fabs(stochastic_left - stochastic_right) < 0.05f, "stochastic: both sides alike");
   printf("texture exact in the other modes, left / right: half resolution trace (Low) %.3f / %.3f, fast cutouts %.3f / %.3f (masked candidates %llu, stochastic %llu)\n", low_left, low_right, fast_left, fast_right,
      (unsigned long long)fast_counters.exact_tests, (unsigned long long)fast_counters.fallback_tests);
   CHECK(low_left < low_right - 0.1f, "texture exact at half resolution");
   CHECK(fast_left < fast_right - 0.1f && fast_counters.exact_tests > 0, "texture exact with fast cutouts (masks take precedence)");
   CHECK(counters.exact_tests > 0 && counters.exact_hits > 0 && counters.exact_hits < counters.exact_tests && counters.fallback_tests == 0, "masked candidates counted");
   CHECK(after.records == 2, "the shifted scale/bias is a second record");

   // RT sun shadows: the light behind the wall, 45 degrees up, so the ground just in front of it sees the wall's lower
   // quadrants: shadowed by the solid one (left), lit through the hole (right). Masks as the AO's, stochastic without.
   CHECK(sun.Init(interop.GetDevice(), interop.GetTimestampFrequency(), error), "SunPass::Init: %s", error.c_str());
   CHECK(sun.EnsureTargets(device, W, H, error), "SunPass::EnsureTargets: %s", error.c_str());
   std::memcpy(sun_settings.view_proj, view_proj.m, sizeof(sun_settings.view_proj));
   std::memcpy(sun_settings.inv_view_proj, inv_view_proj.m, sizeof(sun_settings.inv_view_proj));
   std::memcpy(sun_settings.cam_pos, settings.cam_pos, sizeof(sun_settings.cam_pos));
   std::memcpy(sun_settings.cam_fwd, settings.cam_fwd, sizeof(sun_settings.cam_fwd));
   const V3 sun_l = Normalize({ 0.f, 1.f, 1.f });
   sun_settings.light_dir[0] = sun_l.x; sun_settings.light_dir[1] = sun_l.y; sun_settings.light_dir[2] = sun_l.z;
   sun_settings.angular_diameter_deg = 0.f;
   sun_settings.ray_count = 4;
   sun_settings.t_max = 100.f;
   sun_settings.output = RTSun::Output::Raw;
   sun_settings.cutout_opacity = 0.5f;
   sun_on = true;
   auto measure_sun = [&](float& left, float& right) {
      double sum[2] = {};
      int count[2] = {};
      for (int frame = 0; frame < 8; frame++)
      {
         run_frame();
         ID3D11ShaderResourceView* sun_srv = sun.GetSRV();
         UINT w = 0, h = 0;
         const std::vector<float> vis = ReadAO(device, context, sun_srv, w, h);
         sun_srv->Release();
         for (UINT i = 0; i < W * H; i++)
            for (int side = 0; side < 2; side++)
               if (in_front(i, side == 0))
               {
                  sum[side] += vis[i];
                  count[side]++;
               }
      }
      left = float(sum[0] / (std::max)(count[0], 1));
      right = float(sum[1] / (std::max)(count[1], 1));
   };
   float sun_exact_left, sun_exact_right, sun_stochastic_left, sun_stochastic_right, sun_fast_left, sun_fast_right;
   sun_settings.cutout_exact = true;
   measure_sun(sun_exact_left, sun_exact_right);
   for (int i = 0; i < 3; i++)
      run_frame(); // Counters are read back a few frames late
   const RTAO::AOPass::CutoutCounters sun_counters = sun.GetCutoutCounters();
   sun_settings.cutout_mode = 2;
   measure_sun(sun_fast_left, sun_fast_right);
   sun_settings.cutout_mode = 0;
   sun_settings.cutout_exact = false;
   measure_sun(sun_stochastic_left, sun_stochastic_right);
   sun_on = false;
   printf("RT sun through the textured cutout wall (raw, 8 frames), left (solid below) / right (hole below): texture exact %.3f / %.3f, fast cutouts %.3f / %.3f, "
          "stochastic (opacity 0.5) %.3f / %.3f; masked candidates %llu (%llu solid), stochastic %llu\n", sun_exact_left, sun_exact_right, sun_fast_left, sun_fast_right,
      sun_stochastic_left, sun_stochastic_right, (unsigned long long)sun_counters.exact_tests, (unsigned long long)sun_counters.exact_hits, (unsigned long long)sun_counters.fallback_tests);
   CHECK(sun_exact_left < 0.1f && sun_exact_right > 0.9f, "RT sun, texture exact: shadow of the solid quadrant, light through the hole");
   CHECK(sun_fast_left < 0.1f && sun_fast_right > 0.9f, "RT sun, texture exact with fast cutouts");
   CHECK(std::fabs(sun_stochastic_left - sun_stochastic_right) < 0.1f && sun_stochastic_left > 0.3f && sun_stochastic_left < 0.7f, "RT sun, stochastic: both sides half lit");
   CHECK(sun_counters.exact_tests > 0 && sun_counters.exact_hits > 0 && sun_counters.fallback_tests == 0, "RT sun: masked candidates counted");

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   sun.Shutdown(nullptr, 0);
   timer.Shutdown();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   capture.Shutdown();
   for (IUnknown* p : std::initializer_list<IUnknown*>{ position_vb, uv_vb, ib, alpha_texture, alpha_srv, instance_cb, material_cb, shifted_cb, depth_srv, depth_texture })
      p->Release();
}

// Opt-in (RTAO_PERF=1) cost of the texture exact path in a cutout heavy view: 1920x1080, 4 rays, a "hedge" of 16 cutout
// layers (an 8x8 alpha checker, one instanced draw) standing on the ground the camera looks at
static void TestCutoutPerf(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps)
{
   std::string error;
   const UINT W = 1920, H = 1080, layers = 16;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());
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
   for (UINT y = 0; y < H; y++)
      for (UINT x = 0; x < W; x++)
      {
         float v[4] = { (x + 0.5f) / W * 2.f - 1.f, 1.f - (y + 0.5f) / H * 2.f, 0.5f, 1.f }, w[4] = {};
         for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
               w[c] += v[k] * inv_view_proj(k, c);
         const V3 dir = Normalize(Sub({ w[0] / w[3], w[1] / w[3], w[2] / w[3] }, eye));
         float t;
         if (!RayBox(eye, dir, kBoxes[0], t))
            continue;
         const float pv[4] = { eye.x + dir.x * t, eye.y + dir.y * t, eye.z + dir.z * t, 1.f };
         float clip[4] = {};
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

   // One wall (x -6..6, y 0..2), as 16 instances (direct instance buffer, rows at 48) every 0.25 m in z from -2
   const float positions[4 * 3] = { -6, 0, 0, 6, 0, 0, 6, 2, 0, -6, 2, 0 };
   const float uvs[4 * 2] = { 0, 1, 4, 1, 4, 0, 0, 0 }; // The checker 4 times across
   const uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
   ID3D11Buffer* position_vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, positions, sizeof(positions));
   ID3D11Buffer* uv_vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, uvs, sizeof(uvs));
   ID3D11Buffer* ib = MakeBuffer(device, D3D11_BIND_INDEX_BUFFER, indices, sizeof(indices));
   std::vector<float> instances(layers * 24, 0.f);
   for (UINT i = 0; i < layers; i++)
   {
      const float rows[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2.f + 0.25f * i };
      std::memcpy(&instances[i * 24 + 12], rows, sizeof(rows));
   }
   ID3D11Buffer* instance_buffer = MakeBuffer(device, D3D11_BIND_SHADER_RESOURCE, instances.data(), UINT(instances.size() * 4), D3D11_USAGE_IMMUTABLE, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 96);
   ID3D11ShaderResourceView* instance_srv = nullptr;
   device->CreateShaderResourceView(instance_buffer, nullptr, &instance_srv);
   const float material[8] = { 0, 0, 0, 0, 0.5f, 0, 0, 0 }; // cb_defaultalphatest @16, no scale/bias
   ID3D11Buffer* material_cb = MakeBuffer(device, D3D11_BIND_CONSTANT_BUFFER, material, sizeof(material));
   ID3D11Texture2D* alpha_texture = nullptr;
   ID3D11ShaderResourceView* alpha_srv = nullptr;
   MakeCheckerBc3(device, 8, alpha_texture, alpha_srv);

   RTAO::VertexShaderInfo direct_vs;
   direct_vs.source = RTAO::MatrixSource::DirectInstanceBuffer;
   direct_vs.skip_reason = nullptr;
   direct_vs.instance_buffer_slot = 0;
   direct_vs.instance_buffer_stride = 96;
   direct_vs.record_row0 = 48;
   RTAO::PixelShaderInfo ps_info;
   ps_info.discard = ps_info.alpha_tested = ps_info.texture_exact = true;
   ps_info.alpha_slot = 0;
   ps_info.material_cb_slot = 2;
   ps_info.alpha_ref_cb = 16;
   RTAO::DrawInfo draw;
   draw.vs = &direct_vs;
   draw.vertex_buffer = position_vb;
   draw.vertex_stride = 12;
   draw.position_format = DXGI_FORMAT_R32G32B32_FLOAT;
   draw.index_buffer = ib;
   draw.index_format = DXGI_FORMAT_R32_UINT;
   draw.index_count = 6;
   draw.instance_count = layers;
   draw.instance_buffer = instance_srv;
   draw.cutout = true;
   draw.ps = &ps_info;
   draw.alpha_srv = alpha_srv;
   draw.material_cb = material_cb;
   draw.uv.buffer = uv_vb;
   draw.uv.stride = 8;
   draw.uv.format = DXGI_FORMAT_R32G32_FLOAT;

   ID3DBlob* cs_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "capture_instances_cs", "cs_5_0");
   ID3D11ComputeShader* capture_cs = nullptr;
   device->CreateComputeShader(cs_code->GetBufferPointer(), cs_code->GetBufferSize(), nullptr, &capture_cs);
   cs_code->Release();
   ID3DBlob* strip_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "alpha_strip_cs", "cs_5_0");
   ID3D11ComputeShader* strip_cs = nullptr;
   device->CreateComputeShader(strip_code->GetBufferPointer(), strip_code->GetBufferSize(), nullptr, &strip_cs);
   strip_code->Release();
   RTAO::GeometryCapture capture;
   CHECK(capture.Init(device, interop, capture_cs, nullptr, error), "GeometryCapture::Init: %s", error.c_str());
   CHECK(capture.InitCutouts(device, interop, strip_cs, error), "GeometryCapture::InitCutouts: %s", error.c_str());
   capture_cs->Release();
   strip_cs->Release();
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
   settings.ray_count = 4;

   auto run_frame = [&]() {
      CHECK(capture.CaptureDraw(context, draw), "capture (perf)");
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
      CHECK(interop.Run(context, [&](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& err) {
         retirer.Collect(info.completed_fence_value);
         if (!capture.Record(list, info, blases, tlas, retirer, 64, err))
            return false;
         const RTAO::BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         uint32_t count = 0;
         if (!tlas.Build(ctx, blases, info.frame_slot, top, count, err))
            return false;
         timer.Collect(info.frame_slot);
         timer.Begin(list, info.frame_slot); // The pass only (trace + filter)
         if (!pass.Record(list, info, top, settings, retirer, err, nullptr, &capture.GetCutoutBindings()))
            return false;
         timer.End(list, info.frame_slot);
         return true;
      }, error), "Run (perf): %s", error.c_str());
      capture.OnPresent();
   };
   auto measure = [&](bool exact, float opacity, const char* name) {
      settings.cutout_exact = exact;
      settings.cutout_opacity = opacity;
      for (int i = 0; i < 8; i++)
         run_frame();
      double ms = 0;
      for (int i = 0; i < 32; i++)
      {
         run_frame();
         ms += timer.LastMs();
      }
      interop.WaitForIdle(5000);
      const RTAO::AOPass::CutoutCounters counters = pass.GetCutoutCounters();
      printf("perf %s: AO pass %.3f ms (1920x1080, 4 rays, %u cutout layers), candidates per frame: masked %llu (%llu solid), stochastic %llu\n", name, ms / 32, layers,
         (unsigned long long)counters.exact_tests, (unsigned long long)counters.exact_hits, (unsigned long long)counters.fallback_tests);
   };
   for (int i = 0; i < 6; i++)
      run_frame();
   measure(true, 0.5f, "texture exact");
   measure(false, 0.5f, "stochastic 0.5");
   measure(false, 1.f, "opaque (opacity 1)");
   measure(true, 0.5f, "texture exact");

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   timer.Shutdown();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   capture.Shutdown();
   for (IUnknown* p : std::initializer_list<IUnknown*>{ position_vb, uv_vb, ib, instance_buffer, instance_srv, material_cb, alpha_texture, alpha_srv, depth_srv, depth_texture })
      p->Release();
}
// ---- Benchmark ("rtao_interop_test.exe bench", without RTAO_D3D12_DEBUG): a heavy synthetic street at 3440x1440
// (Karnaca docks like: ~2900 TLAS instances, ~650 of them foliage cutouts made of thousands of overlapping leaf cards),
// depth rasterized by D3D11, then BLAS/TLAS + the AO pass. Prints the GPU time per span (TLAS build, trace, filter)
// and, per configuration, the error of the denoised AO against a 128 ray reference (static and moving camera).

struct BenchMesh
{
   std::vector<float> positions;
   std::vector<uint32_t> indices;
   std::vector<float> uvs; // Foliage: per card 0..1 (texture exact cutouts)
};

static uint32_t BenchRand(uint32_t& state)
{
   state = state * 1664525u + 1013904223u;
   return state >> 8;
}
static float BenchRandF(uint32_t& state) { return float(BenchRand(state) & 0xFFFF) / 65535.f; }
static V3 Add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 Scale(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }

// Leaf cards: "count" random quads of half size [h0, h1], centers in an ellipsoid of radii "r" around "center"
static BenchMesh MakeFoliage(uint32_t seed, uint32_t count, V3 center, V3 r, float h0, float h1)
{
   BenchMesh mesh;
   uint32_t s = seed;
   for (uint32_t i = 0; i < count; i++)
   {
      V3 p;
      do
         p = { BenchRandF(s) * 2 - 1, BenchRandF(s) * 2 - 1, BenchRandF(s) * 2 - 1 };
      while (Dot(p, p) > 1.f);
      p = Add(center, { p.x * r.x, p.y * r.y, p.z * r.z });
      V3 d1 = Normalize({ BenchRandF(s) * 2 - 1, BenchRandF(s) * 2 - 1, BenchRandF(s) * 2 - 1 });
      V3 d2 = Normalize(Cross(d1, Normalize({ BenchRandF(s) * 2 - 1, BenchRandF(s) * 2 - 1, BenchRandF(s) * 2 - 1 })));
      const float h = h0 + (h1 - h0) * BenchRandF(s);
      d1 = Scale(d1, h);
      d2 = Scale(d2, h);
      const uint32_t base = uint32_t(mesh.positions.size() / 3);
      const V3 corners[4] = { Sub(Sub(p, d1), d2), Sub(Add(p, d1), d2), Add(Add(p, d1), d2), Add(Sub(p, d1), d2) };
      for (const V3& c : corners)
         mesh.positions.insert(mesh.positions.end(), { c.x, c.y, c.z });
      mesh.uvs.insert(mesh.uvs.end(), { 0, 0, 1, 0, 1, 1, 0, 1 });
      mesh.indices.insert(mesh.indices.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
   }
   return mesh;
}

static void WritePgm(const char* path, const std::vector<float>& image, UINT w, UINT h, UINT x0, UINT y0, UINT cw, UINT ch)
{
   FILE* file = fopen(path, "wb");
   if (!file)
      return;
   fprintf(file, "P5\n%u %u\n255\n", cw, ch);
   std::vector<uint8_t> row(cw);
   for (UINT y = y0; y < y0 + ch && y < h; y++)
   {
      for (UINT x = 0; x < cw; x++)
         row[x] = uint8_t(std::clamp(image[y * w + std::min(x0 + x, w - 1)], 0.f, 1.f) * 255.f + 0.5f);
      fwrite(row.data(), 1, cw, file);
   }
   fclose(file);
}

struct BenchConfig
{
   const char* name;
   RTAO::PassSettings settings; // Camera fields are filled per frame
};

static void RunBenchmark(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps, const std::vector<BenchConfig>& configs, const char* image_dir)
{
   std::string error;
   const UINT W = 3440, H = 1440;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());

   // Meshes: 0 cube (buildings, details, crates, trunks, ground), 1 tree canopy, 2 bush
   std::vector<BenchMesh> meshes(3);
   meshes[0].positions.assign(kCubeVertices, kCubeVertices + 24);
   meshes[0].indices.assign(kCubeIndices, kCubeIndices + 36);
   meshes[1] = MakeFoliage(1, 1200, { 0, 5, 0 }, { 2.6f, 2.f, 2.6f }, 0.3f, 0.45f);
   meshes[2] = MakeFoliage(2, 250, { 0, 0.7f, 0 }, { 0.9f, 0.7f, 0.9f }, 0.15f, 0.25f);

   struct Instance
   {
      uint32_t mesh;
      float xform[3][4];
      bool cutout;
   };
   std::vector<Instance> instances;
   auto add_box = [&](V3 mn, V3 mx) {
      Instance in = { 0, { { mx.x - mn.x, 0, 0, mn.x }, { 0, mx.y - mn.y, 0, mn.y }, { 0, 0, mx.z - mn.z, mn.z } }, false };
      instances.push_back(in);
   };
   auto add_mesh = [&](uint32_t mesh, V3 at, float yaw, float scale, bool cutout) {
      const float c = std::cos(yaw) * scale, s = std::sin(yaw) * scale;
      Instance in = { mesh, { { c, 0, s, at.x }, { 0, scale, 0, at.y }, { -s, 0, c, at.z } }, cutout };
      instances.push_back(in);
   };
   // Streets every P along x and z (6 m wide), 10 x 10 blocks of buildings with balconies, crates along the walls,
   // trees on both sidewalks of the streets running along x, bushes at the walls
   uint32_t seed = 7;
   const float P = 16.f;
   add_box({ -40, -1, -40 }, { 10 * P + 40, 0, 10 * P + 40 });
   for (int bx = 0; bx < 10; bx++)
      for (int bz = 0; bz < 10; bz++)
      {
         const float x0 = bx * P + 3, z0 = bz * P + 3, height = 6 + 10 * BenchRandF(seed);
         add_box({ x0, 0, z0 }, { x0 + 10, height, z0 + 10 });
         for (int d = 0; d < 6; d++)
         {
            const float y = 2.5f + d * 1.5f, x = x0 + 1 + 7 * BenchRandF(seed);
            if (y < height)
               add_box({ x, y, z0 - 0.8f }, { x + 1.6f, y + 0.15f, z0 });
         }
         for (int c = 0; c < 12; c++)
         {
            const float x = x0 + 10 * BenchRandF(seed), size = 0.5f + 0.7f * BenchRandF(seed);
            const float z = BenchRandF(seed) < 0.5f ? z0 - 0.3f - size : z0 + 10.3f;
            add_box({ x, 0, z }, { x + size, size, z + size });
         }
         for (int b = 0; b < 3; b++)
            add_mesh(2, { x0 + 10 * BenchRandF(seed), 0, BenchRandF(seed) < 0.5f ? z0 - 0.6f : z0 + 10.6f }, 6.28f * BenchRandF(seed), 0.8f + 0.5f * BenchRandF(seed), true);
      }
   for (int street = 0; street <= 10; street++)
      for (int side = 0; side < 2; side++)
         for (float x = 1.5f; x < 10 * P; x += 10.f)
         {
            const V3 at = { x + 2 * BenchRandF(seed), 0, street * P + (side ? 2.f : -2.f) };
            add_box({ at.x - 0.15f, 0, at.z - 0.15f }, { at.x + 0.15f, 3.5f, at.z + 0.15f });
            add_mesh(1, at, 6.28f * BenchRandF(seed), 0.8f + 0.4f * BenchRandF(seed), true);
         }
   uint32_t cutouts = 0;
   size_t triangles = 0;
   for (const Instance& in : instances)
   {
      cutouts += in.cutout;
      triangles += meshes[in.mesh].indices.size() / 3;
   }
   printf("bench scene: %zu instances (%u cutouts), %zu triangles, %ux%u\n", instances.size(), cutouts, triangles, W, H);

   // D3D11 depth: every instance's triangles pre-transformed into one buffer, reversed Z
   std::vector<float> world;
   world.reserve(triangles * 9);
   for (const Instance& in : instances)
      for (uint32_t index : meshes[in.mesh].indices)
      {
         const float* p = &meshes[in.mesh].positions[index * 3];
         for (int r = 0; r < 3; r++)
            world.push_back(in.xform[r][0] * p[0] + in.xform[r][1] * p[1] + in.xform[r][2] * p[2] + in.xform[r][3]);
      }
   ID3D11Buffer* world_vb = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, world.data(), UINT(world.size() * 4));
   const char* vs_source = "cbuffer C : register(b0) { row_major float4x4 VP; }; float4 main(float3 p : POSITION) : SV_Position { return mul(float4(p, 1), VP); }";
   ID3DBlob* vs_blob = nullptr;
   ID3DBlob* vs_errors = nullptr;
   D3DCompile(vs_source, strlen(vs_source), "bench_vs", nullptr, nullptr, "main", "vs_5_0", 0, 0, &vs_blob, &vs_errors);
   if (!vs_blob)
   {
      printf("bench VS: %s\n", vs_errors ? (const char*)vs_errors->GetBufferPointer() : "?");
      g_failures++;
      return;
   }
   ID3D11VertexShader* bench_vs = nullptr;
   device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &bench_vs);
   const D3D11_INPUT_ELEMENT_DESC element = { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };
   ID3D11InputLayout* layout = nullptr;
   device->CreateInputLayout(&element, 1, vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), &layout);
   vs_blob->Release();
   D3D11_TEXTURE2D_DESC depth_desc = { W, H, 1, 1, DXGI_FORMAT_R32_TYPELESS, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   ID3D11Texture2D* depth_texture = nullptr;
   device->CreateTexture2D(&depth_desc, nullptr, &depth_texture);
   D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
   dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
   dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
   ID3D11DepthStencilView* dsv = nullptr;
   device->CreateDepthStencilView(depth_texture, &dsv_desc, &dsv);
   D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
   srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
   srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
   srv_desc.Texture2D.MipLevels = 1;
   ID3D11ShaderResourceView* depth_srv = nullptr;
   device->CreateShaderResourceView(depth_texture, &srv_desc, &depth_srv);
   D3D11_DEPTH_STENCIL_DESC ds_desc = {};
   ds_desc.DepthEnable = TRUE;
   ds_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
   ds_desc.DepthFunc = D3D11_COMPARISON_GREATER;
   ID3D11DepthStencilState* ds_state = nullptr;
   device->CreateDepthStencilState(&ds_desc, &ds_state);
   D3D11_RASTERIZER_DESC rs_desc = {};
   rs_desc.FillMode = D3D11_FILL_SOLID;
   rs_desc.CullMode = D3D11_CULL_NONE;
   rs_desc.DepthClipEnable = TRUE;
   ID3D11RasterizerState* rs_state = nullptr;
   device->CreateRasterizerState(&rs_desc, &rs_state);
   D3D11_BUFFER_DESC cb_desc = { 64, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
   ID3D11Buffer* vp_cb = nullptr;
   device->CreateBuffer(&cb_desc, nullptr, &vp_cb);
   D3D11_TEXTURE2D_DESC staging_desc = depth_desc;
   staging_desc.Usage = D3D11_USAGE_STAGING;
   staging_desc.BindFlags = 0;
   staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
   ID3D11Texture2D* depth_staging = nullptr;
   device->CreateTexture2D(&staging_desc, nullptr, &depth_staging);

   struct Camera
   {
      Mat4 view_proj, inv_view_proj;
      V3 eye, fwd;
   };
   // Street level, looking down a tree lined street; "t" moves the camera forward and turns it a little
   auto make_camera = [&](float t) {
      Camera cam;
      cam.eye = { 20 + t * 0.04f, 1.7f, 5 * P + 0.5f };
      const float yaw = 0.15f + t * 0.004f;
      const V3 target = Add(cam.eye, { std::cos(yaw), -0.12f, std::sin(yaw) });
      const V3 up = { 0, 1, 0 };
      const V3 f = Normalize(Sub(target, cam.eye)), r = Normalize(Cross(up, f)), u = Cross(f, r);
      cam.fwd = f;
      Mat4 view;
      view(0, 0) = r.x; view(0, 1) = u.x; view(0, 2) = f.x;
      view(1, 0) = r.y; view(1, 1) = u.y; view(1, 2) = f.y;
      view(2, 0) = r.z; view(2, 1) = u.z; view(2, 2) = f.z;
      view(3, 0) = -Dot(r, cam.eye); view(3, 1) = -Dot(u, cam.eye); view(3, 2) = -Dot(f, cam.eye); view(3, 3) = 1;
      const float n = 0.1f, fr = 2000.f, sy = 1.f / std::tan(0.5f * 1.0f), sx = sy / (float(W) / float(H));
      Mat4 proj;
      proj(0, 0) = sx;
      proj(1, 1) = sy;
      proj(2, 2) = -n / (fr - n);
      proj(2, 3) = 1;
      proj(3, 2) = n * fr / (fr - n);
      cam.view_proj = Mul(view, proj);
      cam.inv_view_proj = Inverse(cam.view_proj);
      return cam;
   };
   std::vector<uint8_t> sky_mask; // Of the last rasterized camera
   auto rasterize = [&](const Camera& cam, bool read_mask) {
      context->UpdateSubresource(vp_cb, 0, nullptr, cam.view_proj.m, 0, 0);
      context->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 0.f, 0);
      D3D11_VIEWPORT viewport = { 0, 0, (float)W, (float)H, 0, 1 };
      context->RSSetViewports(1, &viewport);
      context->RSSetState(rs_state);
      context->OMSetDepthStencilState(ds_state, 0);
      context->OMSetRenderTargets(0, nullptr, dsv);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->IASetInputLayout(layout);
      const UINT stride = 12, offset = 0;
      context->IASetVertexBuffers(0, 1, &world_vb, &stride, &offset);
      context->VSSetShader(bench_vs, nullptr, 0);
      context->VSSetConstantBuffers(0, 1, &vp_cb);
      context->PSSetShader(nullptr, nullptr, 0);
      context->Draw(UINT(world.size() / 3), 0);
      context->OMSetRenderTargets(0, nullptr, nullptr);
      context->RSSetState(nullptr);
      context->OMSetDepthStencilState(nullptr, 0);
      if (read_mask)
      {
         context->CopyResource(depth_staging, depth_texture);
         D3D11_MAPPED_SUBRESOURCE mapped;
         context->Map(depth_staging, 0, D3D11_MAP_READ, 0, &mapped);
         sky_mask.assign(W * H, 0);
         for (UINT y = 0; y < H; y++)
            for (UINT x = 0; x < W; x++)
               sky_mask[y * W + x] = ((const float*)((const uint8_t*)mapped.pData + y * mapped.RowPitch))[x] <= 0.f;
         context->Unmap(depth_staging, 0);
      }
      // Into the interop's depth (what DrawRTAO does)
      ID3D11RenderTargetView* rtv = interop.GetDepthRTV();
      context->OMSetRenderTargets(1, &rtv, nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
      context->IASetInputLayout(nullptr);
      context->VSSetShader(copy_vs, nullptr, 0);
      context->PSSetShader(depth_ps, nullptr, 0);
      context->PSSetShaderResources(0, 1, &depth_srv);
      context->Draw(4, 0);
      ID3D11ShaderResourceView* null_srv = nullptr;
      context->PSSetShaderResources(0, 1, &null_srv);
      ID3D11RenderTargetView* null_rtv = nullptr;
      context->OMSetRenderTargets(1, &null_rtv, nullptr);
      rtv->Release();
   };

   RTAO::Retirer retirer;
   RTAO::BlasCache blases;
   RTAO::TlasBuilder tlas;
   RTAO::AOPass pass;
   RTAO::GpuTimer timer;
   CHECK(pass.Init(interop.GetDevice(), error), "AOPass::Init: %s", error.c_str());
   CHECK(timer.Init(interop.GetDevice(), interop.GetTimestampFrequency(), error), "GpuTimer::Init: %s", error.c_str());
   uint32_t tlas_count = 0;

   // Texture exact cutouts: the foliage meshes' indices and uvs in a shared "arena" (copied by D3D11 like the capture
   // does), one material whose mask is a round leaf (alpha 1 inside a radius 0.4 disc: about half of each card), rendered
   // through alpha_strip_cs
   RTAO::CutoutMaterials bench_cutouts;
   ID3D12Resource* arena12 = nullptr;
   ID3D11Buffer* arena11 = nullptr;
   ID3D11Texture2D* leaf_texture = nullptr;
   ID3D11ShaderResourceView* leaf_srv = nullptr;
   uint64_t leaf_material = 0;
   RTAO::CutoutGeometry foliage_geometry[3] = {};
   {
      std::vector<uint8_t> arena;
      for (uint32_t m = 1; m < meshes.size(); m++)
      {
         foliage_geometry[m].index_offset = uint32_t(arena.size());
         const uint8_t* indices = reinterpret_cast<const uint8_t*>(meshes[m].indices.data());
         arena.insert(arena.end(), indices, indices + meshes[m].indices.size() * 4);
         foliage_geometry[m].uv_offset = uint32_t(arena.size());
         const uint8_t* uvs = reinterpret_cast<const uint8_t*>(meshes[m].uvs.data());
         arena.insert(arena.end(), uvs, uvs + meshes[m].uvs.size() * 4);
         foliage_geometry[m].uv_stride = 8;
         foliage_geometry[m].uv_code = RTAO::VertexFormatCode(DXGI_FORMAT_R32G32_FLOAT);
         foliage_geometry[m].index32 = 1;
      }
      CHECK(interop.CreateSharedBuffer(device, arena.size(), arena12, arena11, error), "bench arena: %s", error.c_str());
      ID3D11Buffer* source = MakeBuffer(device, D3D11_BIND_VERTEX_BUFFER, arena.data(), UINT(arena.size()));
      context->CopySubresourceRegion(arena11, 0, 0, 0, 0, source, 0, nullptr);
      source->Release();
      std::vector<uint32_t> texels(64 * 64);
      for (UINT y = 0; y < 64; y++)
         for (UINT x = 0; x < 64; x++)
         {
            const float u = (x + 0.5f) / 64.f - 0.5f, v = (y + 0.5f) / 64.f - 0.5f;
            texels[y * 64 + x] = (u * u + v * v < 0.16f ? 0xFF000000u : 0u) | 0x00204020u;
         }
      D3D11_TEXTURE2D_DESC leaf_desc = { 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 }, D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
      D3D11_SUBRESOURCE_DATA leaf_data = { texels.data(), 64 * 4, 0 };
      device->CreateTexture2D(&leaf_desc, &leaf_data, &leaf_texture);
      device->CreateShaderResourceView(leaf_texture, nullptr, &leaf_srv);
      ID3DBlob* strip_code = Compile(L"..\\..\\..\\..\\Shaders\\Dishonored 2\\Luma_RTAO.hlsl", "alpha_strip_cs", "cs_5_0");
      ID3D11ComputeShader* strip_cs = nullptr;
      device->CreateComputeShader(strip_code->GetBufferPointer(), strip_code->GetBufferSize(), nullptr, &strip_cs);
      strip_code->Release();
      CHECK(bench_cutouts.Init(device, interop, strip_cs, error), "bench cutouts: %s", error.c_str());
      strip_cs->Release();
      uint32_t tile = RTAO::CutoutMaterials::kNoTile;
      leaf_material = bench_cutouts.AcquireMaterial(leaf_srv, 3, tile);
      CHECK(leaf_material && tile != RTAO::CutoutMaterials::kNoTile && bench_cutouts.RecordStrip(context, leaf_srv, 3, tile), "bench leaf mask");
      bench_cutouts.OnStripExecuted(leaf_material, tile); // Immediate context: executed by the next Run
   }
   // One frame: TLAS (all instances re-added, like the capture's readback), AO pass. Spans: BLAS, TLAS, trace, filter.
   auto run = [&](const Camera& cam, RTAO::PassSettings settings) {
      std::memcpy(settings.view_proj, cam.view_proj.m, sizeof(settings.view_proj));
      std::memcpy(settings.inv_view_proj, cam.inv_view_proj.m, sizeof(settings.inv_view_proj));
      settings.cam_pos[0] = cam.eye.x; settings.cam_pos[1] = cam.eye.y; settings.cam_pos[2] = cam.eye.z;
      settings.cam_fwd[0] = cam.fwd.x; settings.cam_fwd[1] = cam.fwd.y; settings.cam_fwd[2] = cam.fwd.z;
      settings.reversed_z = true;
      CHECK(interop.Run(context, [&](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& err) {
         retirer.Collect(info.completed_fence_value);
         timer.Collect(info.frame_slot);
         timer.Begin(list, info.frame_slot);
         const RTAO::BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         for (uint32_t m = 0; m < meshes.size(); m++)
         {
            RTAO::CpuMesh mesh = { meshes[m].positions.data(), uint32_t(meshes[m].positions.size() / 3), 12, meshes[m].indices.data(), uint32_t(meshes[m].indices.size()) };
            if (!blases.BuildFromCpu(ctx, m, mesh, err))
               return false;
         }
         blases.FlushBarrier(list);
         uint32_t records[3] = { RTAO::kNoCutoutRecord, RTAO::kNoCutoutRecord, RTAO::kNoCutoutRecord };
         static const float identity_scalebias[4] = { 1, 1, 0, 0 };
         for (uint32_t m = 1; m < meshes.size(); m++)
            records[m] = bench_cutouts.ResolveRecord(leaf_material, foliage_geometry[m], identity_scalebias, 0.5f);
         RTAO::CutoutBindings bindings;
         if (!bench_cutouts.Record(list, info, retirer, arena12, bindings, err))
            return false;
         timer.Mark(list, info.frame_slot);
         tlas.BeginFrame();
         for (size_t i = 0; i < instances.size(); i++)
            tlas.AddInstance(i, instances[i].mesh, instances[i].xform, instances[i].cutout ? RTAO::kInstanceMaskCutout : RTAO::kInstanceMaskOpaque, records[instances[i].mesh]);
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         if (!tlas.Build(ctx, blases, info.frame_slot, top, tlas_count, err))
            return false;
         timer.Mark(list, info.frame_slot);
         if (!pass.Record(list, info, top, settings, retirer, err, &timer, &bindings))
            return false;
         timer.End(list, info.frame_slot);
         return true;
      }, error), "Run (bench): %s", error.c_str());
   };
   auto read_ao = [&]() {
      ID3D11ShaderResourceView* ao_srv = interop.GetAOSRV();
      UINT w = 0, h = 0;
      std::vector<float> ao = ReadAO(device, context, ao_srv, w, h);
      ao_srv->Release();
      return ao;
   };
   auto mae = [&](const std::vector<float>& a, const std::vector<float>& b) {
      double sum = 0;
      size_t count = 0;
      for (size_t i = 0; i < a.size(); i++)
         if (!sky_mask[i])
         {
            sum += std::fabs(a[i] - b[i]);
            count++;
         }
      return count ? sum / count : 0.0;
   };

   // References: 128 rays, no denoiser, at the static pose and at the end of the camera move
   const int kMoveFrames = 24;
   const Camera cam_static = make_camera(0.f), cam_moved = make_camera(float(kMoveFrames));
   RTAO::PassSettings reference_settings;
   reference_settings.ray_count = 128;
   reference_settings.filter_mode = RTAO::FilterMode::Off;
   reference_settings.strength = 1.f;
   // [exact][pose]: a configuration is compared with the reference of its own cutout model (stochastic opacity 0.5, or
   // the leaf masks)
   std::vector<float> references[2][2];
   for (int exact = 0; exact < 2; exact++)
      for (int pose = 0; pose < 2; pose++)
      {
         reference_settings.cutout_exact = exact != 0;
         rasterize(pose ? cam_moved : cam_static, true);
         run(pose ? cam_moved : cam_static, reference_settings);
         references[exact][pose] = read_ao();
      }
   const std::vector<float>* reference = references[0];
   size_t sky = 0;
   for (uint8_t s : sky_mask)
      sky += s;
   printf("bench: TLAS %u instances, %.1f%% sky\n", tlas_count, 100.0 * sky / sky_mask.size());
   if (image_dir)
   {
      char path[512];
      snprintf(path, sizeof(path), "%s\\reference.pgm", image_dir);
      WritePgm(path, reference[1], W, H, 0, 0, W, H);
      snprintf(path, sizeof(path), "%s\\reference_static.pgm", image_dir);
      WritePgm(path, reference[0], W, H, 0, 0, W, H);
   }

   const char* filter = getenv("RTAO_BENCH_FILTER"); // Only the configurations whose name contains it
   for (const BenchConfig& config : configs)
   {
      if (filter && !strstr(config.name, filter))
         continue;
      RTAO::PassSettings settings = config.settings;
      settings.strength = 1.f;
      reference = references[settings.cutout_exact ? 1 : 0];
      settings.reset_history = true;
      // Static camera: converge, time, measure flicker and error
      rasterize(cam_static, true);
      // Medians over the timed frames (other GPU work on the machine makes single frames noisy)
      std::vector<double> span_samples[RTAO::GpuTimer::kMaxMarks + 2];
      const int kWarmup = 40, kTimed = 60;
      std::vector<float> previous, current;
      double flicker = 0;
      for (int frame = 0; frame < kWarmup + kTimed; frame++)
      {
         run(cam_static, settings);
         settings.reset_history = false;
         if (frame >= kWarmup)
         {
            for (uint32_t s = 0; s < RTAO::GpuTimer::kMaxMarks + 1; s++)
               span_samples[s].push_back(timer.LastSpanMs(s));
            span_samples[RTAO::GpuTimer::kMaxMarks + 1].push_back(timer.LastMs());
         }
         if (frame >= kWarmup + kTimed - 4)
         {
            previous.swap(current);
            current = read_ao();
            if (!previous.empty())
               flicker += mae(current, previous) / 3.0;
         }
      }
      const double static_error = mae(current, reference[0]);
      if (image_dir)
      {
         char path[512];
         snprintf(path, sizeof(path), "%s\\%s_static.pgm", image_dir, config.name);
         WritePgm(path, current, W, H, 0, 0, W, H);
      }
      // Moving camera: from a converged history, kMoveFrames frames of motion, error at the last pose
      for (int frame = 1; frame <= kMoveFrames; frame++)
      {
         const Camera cam = make_camera(float(frame));
         rasterize(cam, frame == kMoveFrames);
         run(cam, settings);
      }
      const std::vector<float> moved = read_ao();
      const double moving_error = mae(moved, reference[1]);
      auto median = [&](uint32_t s) {
         std::vector<double>& v = span_samples[s];
         std::sort(v.begin(), v.end());
         return v.empty() ? 0.0 : v[v.size() / 2];
      };
      printf("%-34s GPU %.3f ms (BLAS %.3f, TLAS %.3f, trace %.3f, filter %.3f) | error static %.4f moving %.4f, flicker %.4f\n", config.name, median(RTAO::GpuTimer::kMaxMarks + 1), median(0), median(1), median(2), median(3),
         static_error, moving_error, flicker);
      if (image_dir)
      {
         char path[512];
         snprintf(path, sizeof(path), "%s\\%s.pgm", image_dir, config.name);
         WritePgm(path, moved, W, H, 0, 0, W, H);
      }
   }

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   timer.Shutdown();
   pass.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   bench_cutouts.Shutdown();
   for (IUnknown* p : std::initializer_list<IUnknown*>{ world_vb, bench_vs, layout, depth_texture, dsv, depth_srv, ds_state, rs_state, vp_cb, depth_staging, arena11, arena12, leaf_texture, leaf_srv })
      if (p)
         p->Release();
}

static std::vector<BenchConfig> BenchConfigs()
{
   std::vector<BenchConfig> configs;
   RTAO::PassSettings base;
   base.cutout_exact = false; // The stochastic cutouts (texture exact variants at the end)
   configs.push_back({ "high_4rays_opacity0.5", base });
   RTAO::PassSettings s = base;
   s.cutout_opacity = 1.f;
   configs.push_back({ "high_4rays_opacity1", s });
   s.cutout_opacity = 0.f;
   configs.push_back({ "high_4rays_opacity0", s });
   s = base;
   s.ray_count = 2;
   configs.push_back({ "medium_2rays", s });
   s.ray_count = 1;
   configs.push_back({ "1ray", s });
   s = base;
   s.cutout_mode = 1;
   configs.push_back({ "high_4rays_perray", s });
   s.cutout_mode = 2;
   configs.push_back({ "high_4rays_perray_sq", s });
   s = base;
   s.half_res = true;
   configs.push_back({ "low_half_4rays", s });
   s.ray_count = 2;
   configs.push_back({ "low_half_2rays", s });
   s.ray_count = 4;
   s.cutout_mode = 2;
   configs.push_back({ "low_half_4rays_fastcutouts", s });
   s = base;
   s.ray_count = 2;
   s.cutout_mode = 2;
   configs.push_back({ "medium_2rays_perray_sq", s });
   // Texture exact cutouts (round leaf masks)
   s = base;
   s.cutout_exact = true;
   configs.push_back({ "high_4rays_exact", s });
   s.ray_count = 2;
   configs.push_back({ "medium_2rays_exact", s });
   s.ray_count = 4;
   s.half_res = true;
   configs.push_back({ "low_half_4rays_exact", s });
   s.cutout_mode = 2;
   configs.push_back({ "low_half_4rays_fastcutouts_exact", s });
   return configs;
}

#include "rtsun_test.inl"

int main(int argc, char** argv)
{
   setvbuf(stdout, nullptr, _IONBF, 0);
   const bool bench = argc > 1 && strcmp(argv[1], "bench") == 0;
   const char* image_dir = bench && argc > 2 ? argv[2] : nullptr;
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
   // Shadow map casters: MVP derived matrices must be rotation x scale + translation (RTAO::IsRigidTransform)
   {
      const float identity[12] = { 1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 100 };
      const float c = 0.8f, s = 0.6f; // Rotation about z, then non uniform scale (2, 0.5, 3)
      const float rotated_scaled[12] = { c * 2, -s * 0.5f, 0, 10, s * 2, c * 0.5f, 0, 20, 0, 0, 3, 30 };
      const float skewed[12] = { 1, 0.4f, 0, 0, 0, 1, 0, 0, 0, 0.3f, 1, 0 }; // A view's skew left in (wrong VP)
      const float flat[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }; // A projection's collapsed axis
      CHECK(RTAO::IsRigidTransform(identity), "rigid check: identity rejected");
      CHECK(RTAO::IsRigidTransform(rotated_scaled), "rigid check: rotation x scale rejected");
      CHECK(!RTAO::IsRigidTransform(skewed), "rigid check: skewed matrix accepted");
      CHECK(!RTAO::IsRigidTransform(flat), "rigid check: flat matrix accepted");
      printf("rigid transform check: ok\n");
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

   if (bench)
   {
      RunBenchmark(device, context, interop, vs, ps, BenchConfigs(), image_dir);
      interop.Shutdown();
      printf(g_failures ? "FAILED (%d)\n" : "PASSED\n", g_failures);
      return g_failures ? 1 : 0;
   }

   TestTlasRetention();
   TestSharedBuffer(device, context, interop);
   TestRayTracedScene(device, context, interop, vs, ps);
   TestReflection();
   TestGeometryCapture(device, context, interop, vs, ps);
   TestFrameMeshes(device, context, interop, vs, ps);
   TestSunShadows(device, context, interop, vs, ps);
   TestLightCapture(device, context);
   TestScreenShadowsShader(device, context);
   TestCutoutTextures(device, context, interop, vs, ps);
   if (GetEnvironmentVariableA("RTAO_PERF", nullptr, 0) != 0)
      TestCutoutPerf(device, context, interop, vs, ps);

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
