// RT sun shadows tests, included by rtao_interop_test.cpp (uses its CHECK, Mat4, V3, Box, kBoxes, RayBox, ReadAO).
//  - TestSunShadows: the box scene lit by a directional light, traced by RTSun::SunPass from the RTAO TLAS. A hard
//    light must match the analytic CPU shadow per pixel; a wider light must widen the penumbra, and the denoised soft
//    result must match the CPU's disc integral.
//  - TestLightCapture: the light direction from a copy of the screen shadows constant buffer (bound with an offset).
//  - TestScreenShadowsShader: Shaders/Dishonored 2/ScreenShadows_*.hlsl against the game's own two pixel shaders (from
//    Luma's dump, if present) on random inputs, and the RT modes.

#include "rtsun_pass.h"

#include <d3d11shader.h>
#include <random>

// Includes relative to the including file, like Luma's FxcD3DInclude (D3D_COMPILE_STANDARD_FILE_INCLUDE resolves
// nested includes relative to the first file only)
class RelativeInclude : public ID3DInclude
{
public:
   explicit RelativeInclude(std::wstring root_dir) : root(std::move(root_dir)) {}
   HRESULT __stdcall Open(D3D_INCLUDE_TYPE, LPCSTR name, LPCVOID parent, LPCVOID* data, UINT* bytes) override
   {
      std::wstring dir = root;
      for (const auto& opened : files)
         if (opened.first == parent)
            dir = opened.second;
      std::wstring path = dir + L"\\";
      for (const char* c = name; *c; c++)
         path += wchar_t(*c == '/' ? '\\' : *c);
      FILE* file = _wfopen(path.c_str(), L"rb");
      if (!file)
         return E_FAIL;
      fseek(file, 0, SEEK_END);
      const long size = ftell(file);
      fseek(file, 0, SEEK_SET);
      char* buffer = new char[size_t(size) + 1];
      fread(buffer, 1, size_t(size), file);
      buffer[size] = 0;
      fclose(file);
      files.emplace_back(buffer, path.substr(0, path.find_last_of(L'\\')));
      *data = buffer;
      *bytes = UINT(size);
      return S_OK;
   }
   HRESULT __stdcall Close(LPCVOID data) override
   {
      for (size_t i = 0; i < files.size(); i++)
         if (files[i].first == data)
         {
            files.erase(files.begin() + i);
            break;
         }
      delete[] static_cast<const char*>(data);
      return S_OK;
   }

private:
   std::wstring root;
   std::vector<std::pair<const void*, std::wstring>> files;
};

// Add() and Scale() come from rtao_interop_test.cpp

// Analytic visibility of the light at p (ground or box top, normal +y): the fraction of "samples" directions inside
// the light's disc (half angle "radius", radians) that no box blocks
static float SunVisibilityCPU(V3 p, V3 L, float radius, int samples)
{
   const V3 t = Normalize(std::fabs(L.z) < 0.999f ? Cross(L, { 0, 0, 1 }) : V3{ 1, 0, 0 });
   const V3 b = Cross(L, t);
   const V3 origin = Add(p, { 0, 1e-3f, 0 });
   int lit = 0;
   const int n = radius > 0.f ? samples : 1;
   const int rings = int(std::sqrt(float(n)));
   for (int i = 0; i < n; i++)
   {
      V3 d = L;
      if (radius > 0.f)
      {
         // Stratified over the disc (rings x sectors)
         const float u = (float(i / rings) + 0.5f) / float((n + rings - 1) / rings);
         const float v = (float(i % rings) + 0.5f) / float(rings);
         const float r = std::sqrt(u) * std::tan(radius), phi = 6.2831853f * v;
         d = Normalize(Add(L, Add(Scale(t, r * std::cos(phi)), Scale(b, r * std::sin(phi)))));
      }
      bool blocked = false;
      for (int k = 1; k < kBoxCount && !blocked; k++)
      {
         float th;
         blocked = RayBox(origin, d, kBoxes[k], th) && th > 0.f;
      }
      lit += blocked ? 0 : 1;
   }
   return float(lit) / float(n);
}

static void TestSunShadows(ID3D11Device* device, ID3D11DeviceContext* context, RTAO::Interop& interop, ID3D11VertexShader* copy_vs, ID3D11PixelShader* depth_ps)
{
   std::string error;
   const UINT W = 320, H = 180;
   CHECK(interop.EnsureTargets(device, W, H, W, H, error), "EnsureTargets: %s", error.c_str());

   // Same camera and CPU "rasterization" as TestRayTracedScene
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

   RTAO::Retirer retirer;
   RTAO::BlasCache blases;
   RTAO::TlasBuilder tlas;
   RTSun::SunPass sun;
   CHECK(sun.Init(interop.GetDevice(), interop.GetTimestampFrequency(), error), "SunPass::Init: %s", error.c_str());
   CHECK(sun.EnsureTargets(device, W, H, error), "SunPass::EnsureTargets: %s", error.c_str());

   // Light from above, behind and to the right of box 1: its shadow falls toward the camera, on visible ground
   const V3 L = Normalize({ 0.3f, 1.f, 0.6f });
   RTSun::SunSettings settings;
   std::memcpy(settings.view_proj, view_proj.m, sizeof(settings.view_proj));
   std::memcpy(settings.inv_view_proj, inv_view_proj.m, sizeof(settings.inv_view_proj));
   settings.cam_pos[0] = eye.x; settings.cam_pos[1] = eye.y; settings.cam_pos[2] = eye.z;
   settings.cam_fwd[0] = f.x; settings.cam_fwd[1] = f.y; settings.cam_fwd[2] = f.z;
   settings.light_dir[0] = L.x; settings.light_dir[1] = L.y; settings.light_dir[2] = L.z;
   settings.t_max = 100.f;

   uint32_t instances = 0;
   auto run_frame = [&]() {
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
         const RTAO::BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         RTAO::CpuMesh cube = { kCubeVertices, 8, 12, kCubeIndices, 36 };
         if (!blases.BuildFromCpu(ctx, 1, cube, err))
            return false;
         blases.FlushBarrier(list);
         tlas.BeginFrame();
         for (int b = 0; b < kBoxCount; b++)
         {
            const Box& box = kBoxes[b];
            const float xform[3][4] = { { box.max.x - box.min.x, 0, 0, box.min.x }, { 0, box.max.y - box.min.y, 0, box.min.y }, { 0, 0, box.max.z - box.min.z, box.min.z } };
            tlas.AddInstance(uint64_t(b), 1, xform);
         }
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         if (!tlas.Build(ctx, blases, info.frame_slot, top, instances, err))
            return false;
         return sun.Record(list, info, top, settings, retirer, err);
      }, error), "Run (RT sun): %s", error.c_str());
      ID3D11ShaderResourceView* srv = sun.GetSRV();
      UINT w = 0, h = 0;
      std::vector<float> vis = ReadAO(device, context, srv, w, h);
      srv->Release();
      return vis;
   };

   // Receivers with a known normal (+y): the ground, and box tops
   auto is_receiver = [&](UINT i) { return hit_id[i] == 0 || (hit_id[i] > 0 && hit_pos[i].y >= kBoxes[hit_id[i]].max.y - 1e-3f); };

   // 1) Hard light: the raw trace must match the analytic shadow, away from the boundary (1 px of bias and edges)
   settings.angular_diameter_deg = 0.f;
   settings.ray_count = 1;
   settings.output = RTSun::Output::Raw;
   std::vector<float> hard = run_frame();
   std::vector<float> expected_hard(W * H, -1.f);
   for (UINT i = 0; i < W * H; i++)
      if (is_receiver(i))
         expected_hard[i] = SunVisibilityCPU(hit_pos[i], L, 0.f, 1);
   int compared = 0, mismatched = 0, shadowed = 0;
   for (UINT y = 1; y + 1 < H; y++)
      for (UINT x = 1; x + 1 < W; x++)
      {
         const UINT i = y * W + x;
         if (expected_hard[i] < 0.f)
            continue;
         bool boundary = false;
         for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
               const float e = expected_hard[(y + dy) * W + (x + dx)];
               boundary = boundary || e != expected_hard[i] || hit_id[(y + dy) * W + (x + dx)] != hit_id[i];
            }
         if (boundary)
            continue;
         compared++;
         shadowed += expected_hard[i] < 0.5f ? 1 : 0;
         mismatched += (hard[i] > 0.5f) != (expected_hard[i] > 0.5f) ? 1 : 0;
      }
   printf("RT sun hard (%u TLAS instances): %d receiver pixels compared, %d in shadow, %d mismatched (%.3f %%)\n", instances, compared, shadowed, mismatched, compared ? 100.0 * mismatched / compared : 0.0);
   CHECK(compared > 10000 && shadowed > 500, "the scene must have lit and shadowed receivers (%d, %d)", compared, shadowed);
   CHECK(mismatched <= compared / 500, "hard RT sun shadow must match the analytic shadow (%d of %d differ)", mismatched, compared);

   // 1b) Debug scene view from the camera: geometry pixels show the "main view only" color (green, RGB565 in the 16 bits),
   // sky pixels the "nothing" grey
   {
      settings.output = RTSun::Output::SceneView;
      const std::vector<float> scene = run_frame();
      int geometry = 0, green = 0, sky = 0, grey = 0;
      for (UINT i = 0; i < W * H; i++)
      {
         const uint32_t v = uint32_t(std::lround(std::clamp(scene[i], 0.f, 1.f) * 65535.f));
         const float r = float((v >> 11) & 31) / 31.f, g = float((v >> 5) & 63) / 63.f, b = float(v & 31) / 31.f;
         if (hit_id[i] >= 0)
         {
            geometry++;
            green += (g > 0.3f && g > r * 1.5f && g > b * 1.5f) ? 1 : 0;
         }
         else
         {
            sky++;
            grey += (std::fabs(r - 0.12f) < 0.05f && std::fabs(g - 0.12f) < 0.05f && std::fabs(b - 0.12f) < 0.05f) ? 1 : 0;
         }
      }
      printf("RT scene view: %d of %d geometry pixels in the main view color, %d of %d sky pixels grey\n", green, geometry, grey, sky);
      CHECK(geometry > 0 && green >= geometry * 98 / 100, "scene view: geometry must show its capture source color (%d of %d)", green, geometry);
      CHECK(grey >= sky * 98 / 100, "scene view: rays that hit nothing must be grey (%d of %d)", grey, sky);
      settings.output = RTSun::Output::SceneSun;
      run_frame(); // Must run (no validation errors); content checked in game
   }

   // 2) Soft light: the traced penumbra (raw rays, averaged here over frames) has the analytic width and grows with
   // the angle; the denoised result (its 3x3 spatial pass softens edges by about a pixel) follows the disc integral
   auto soft_case = [&](float diameter_deg, float& mae, int& gpu_penumbra, int& cpu_penumbra) {
      settings.angular_diameter_deg = diameter_deg;
      settings.ray_count = 4;
      settings.output = RTSun::Output::Raw;
      std::vector<double> mean(W * H, 0.0);
      const int raw_frames = 32;
      for (int i = 0; i < raw_frames; i++)
      {
         const std::vector<float> raw = run_frame();
         for (UINT p = 0; p < W * H; p++)
            mean[p] += raw[p] / raw_frames;
      }
      settings.output = RTSun::Output::Denoised;
      settings.reset_history = true;
      std::vector<float> vis = run_frame();
      settings.reset_history = false;
      for (int i = 0; i < 40; i++)
         vis = run_frame();
      const float radius = diameter_deg * 0.5f * 3.14159265f / 180.f;
      double err = 0;
      int count = 0;
      gpu_penumbra = cpu_penumbra = 0;
      for (UINT i = 0; i < W * H; i++)
      {
         if (hit_id[i] != 0)
            continue; // Ground only
         const float e = SunVisibilityCPU(hit_pos[i], L, radius, 256);
         err += std::fabs(vis[i] - e);
         count++;
         cpu_penumbra += (e > 0.1f && e < 0.9f) ? 1 : 0;
         gpu_penumbra += (mean[i] > 0.1 && mean[i] < 0.9) ? 1 : 0;
      }
      mae = count ? float(err / count) : 1.f;
   };
   float mae_small = 0, mae_large = 0;
   int gpu_small = 0, cpu_small = 0, gpu_large = 0, cpu_large = 0;
   soft_case(0.53f, mae_small, gpu_small, cpu_small);
   const double ms_small = sun.LastGpuMs();
   soft_case(6.f, mae_large, gpu_large, cpu_large);
   printf("RT sun soft: 0.53 deg: penumbra %d px (CPU %d), mean abs error %.4f; 6 deg: penumbra %d px (CPU %d), mean abs error %.4f; GPU %.3f / %.3f ms\n",
      gpu_small, cpu_small, mae_small, gpu_large, cpu_large, mae_large, ms_small, sun.LastGpuMs());
   CHECK(cpu_large > cpu_small * 3 && gpu_large > gpu_small * 3, "the penumbra must widen with the light's angle");
   CHECK(gpu_large > cpu_large / 2 && gpu_large < cpu_large * 2, "the soft penumbra must have about the analytic size");
   CHECK(mae_small < 0.03f && mae_large < 0.05f, "the denoised soft shadow must follow the analytic disc integral");

   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   // Resize path: re-created targets, old ones retired by the next Record
   CHECK(sun.EnsureTargets(device, W / 2, H / 2, error), "SunPass resize: %s", error.c_str());
   CHECK(interop.WaitForIdle(5000), "WaitForIdle");
   retirer.ReleaseAll();
   sun.Shutdown(nullptr, 0);
   tlas.Clear(nullptr, 0);
   blases.Clear(nullptr, 0);
   depth_srv->Release();
   depth_texture->Release();
}

// A cascade 0 light space whose depth axis is "axis" (orthographic: rows are the world axes' images)
static void FillScreenShadowsCB(float cb[RTSun::kScreenShadowsCBBytes / 4], V3 axis, V3 sun)
{
   std::memset(cb, 0, RTSun::kScreenShadowsCBBytes);
   const V3 t = Normalize(std::fabs(axis.y) < 0.999f ? Cross(axis, { 0, 1, 0 }) : V3{ 1, 0, 0 });
   const V3 b = Cross(axis, t);
   const float scale_xy = 0.005f, scale_z = 0.001f;
   float* m = cb + RTSun::kShadowProjectionsOffset / 4;
   const V3 cols[3] = { t, b, axis };
   const float scales[3] = { scale_xy, scale_xy, scale_z };
   for (int row = 0; row < 3; row++)
      for (int c = 0; c < 3; c++)
         m[row * 4 + c] = (&cols[c].x)[row] * scales[c];
   m[3 * 4 + 0] = 0.25f; // Light space xy (x the map size, + 0.5) in the map's texels for the test's world
   m[3 * 4 + 1] = 0.25f;
   m[3 * 4 + 2] = 0.5f;
   m[3 * 4 + 3] = 1.f;
   float* s = cb + RTSun::kAtmosphereSunOffset / 4;
   s[0] = sun.x * 3.f; // Not normalized on purpose
   s[1] = sun.y * 3.f;
   s[2] = sun.z * 3.f;
}

static void TestLightCapture(ID3D11Device* device, ID3D11DeviceContext* context)
{
   const V3 L = Normalize({ -0.2f, 0.8f, 0.5f });
   float cb[RTSun::kScreenShadowsCBBytes / 4];
   float dir[3] = {}, sun_dir[3] = {};
   // The depth axis may point either way (depth convention): the result is the side above the horizon (y up here)
   FillScreenShadowsCB(cb, L, L);
   bool ok = RTSun::ComputeLightDirection(cb, sizeof(cb), 1, dir, sun_dir);
   CHECK(ok && std::fabs(Dot({ dir[0], dir[1], dir[2] }, L) - 1.f) < 1e-4f, "light direction from the cascade axis (%.3f %.3f %.3f)", dir[0], dir[1], dir[2]);
   CHECK(std::fabs(Dot({ sun_dir[0], sun_dir[1], sun_dir[2] }, L) - 1.f) < 1e-4f, "atmosphere sun normalized");
   FillScreenShadowsCB(cb, Scale(L, -1.f), L);
   ok = RTSun::ComputeLightDirection(cb, sizeof(cb), 1, dir, sun_dir);
   CHECK(ok && std::fabs(Dot({ dir[0], dir[1], dir[2] }, L) - 1.f) < 1e-4f, "light direction from a flipped cascade axis (%.3f %.3f %.3f)", dir[0], dir[1], dir[2]);
   std::memset(cb, 0, sizeof(cb));
   CHECK(!RTSun::ComputeLightDirection(cb, sizeof(cb), 1, dir, sun_dir), "a zero cbuffer has no direction");

   // Through the D3D11 path: a DYNAMIC constant buffer bound with an offset (PSSetConstantBuffers1), copied and read back
   ID3D11DeviceContext1* context1 = nullptr;
   if (FAILED(context->QueryInterface(IID_PPV_ARGS(&context1))))
   {
      CHECK(false, "ID3D11DeviceContext1");
      return;
   }
   D3D11_BUFFER_DESC desc = { 1024, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
   ID3D11Buffer* buffer = nullptr;
   device->CreateBuffer(&desc, nullptr, &buffer);
   const V3 L2 = Normalize({ 0.4f, 0.6f, -0.3f });
   FillScreenShadowsCB(cb, Scale(L2, -1.f), L2);
   D3D11_MAPPED_SUBRESOURCE mapped;
   context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
   std::memset(mapped.pData, 0x7F, 1024);
   std::memcpy((uint8_t*)mapped.pData + 256, cb, sizeof(cb));
   context->Unmap(buffer, 0);
   const UINT first = 16, num = 32; // 256 bytes in, 512 bytes long (multiples of 16 constants)
   context1->PSSetConstantBuffers1(2, 1, &buffer, &first, &num);

   RTSun::LightCapture capture;
   std::string error;
   CHECK(capture.Init(device, error), "LightCapture::Init: %s", error.c_str());
   capture.up_axis = 1;
   capture.OnScreenShadowsDraw(context, 10);
   capture.OnScreenShadowsDraw(context, 10); // Once per frame
   capture.Poll(context, 11); // Too early (latency)
   CHECK(!capture.GetDirection(dir), "no direction before the latency");
   context->Flush();
   uint32_t frame = 0;
   ok = false;
   for (int attempt = 0; attempt < 200 && !ok; attempt++)
   {
      capture.Poll(context, 12); // Never waits: the copy may still be in flight
      ok = capture.GetDirection(dir, &frame, sun_dir);
      if (!ok)
         Sleep(1);
   }
   printf("light capture: %s, direction %.3f %.3f %.3f (frame %u), atmosphere sun %.3f %.3f %.3f\n", ok ? "ok" : "none", dir[0], dir[1], dir[2], frame, sun_dir[0], sun_dir[1], sun_dir[2]);
   CHECK(ok && frame == 10 && std::fabs(Dot({ dir[0], dir[1], dir[2] }, L2) - 1.f) < 1e-4f, "captured light direction");
   ID3D11Buffer* null_buffer = nullptr;
   context->PSSetConstantBuffers(2, 1, &null_buffer);
   capture.Shutdown();
   buffer->Release();
   context1->Release();
}

// The replacement screen shadows shaders against the game's (Luma's dump), on random inputs
static void TestScreenShadowsShader(ID3D11Device* device, ID3D11DeviceContext* context)
{
   const std::string dump_dir = GameShaderDumpDir(); // Luma's shader dump of the game (read at test time, not committed)
   const char* dump = dump_dir.c_str();
   struct Variant
   {
      const char* original;
      const wchar_t* replacement;
   };
   const Variant variants[] = {
      { "0xF324B47B.ps_5_0.cso", L"..\\..\\..\\..\\Shaders\\Dishonored 2\\ScreenShadows_0xF324B47B.ps_5_0.hlsl" },
      { "0x6C8C2397.ps_5_0.cso", L"..\\..\\..\\..\\Shaders\\Dishonored 2\\ScreenShadows_Dither_0x6C8C2397.ps_5_0.hlsl" },
   };
   const UINT S = 64; // Target size
   const UINT M = 128; // Shadow map size
   std::mt19937 rng(1234);
   std::uniform_real_distribution<float> uni(0.f, 1.f);

   // VS: INTERP0 = uv, SV_POSITION (a fullscreen triangle)
   const char* vs_source = "void main(uint id : SV_VertexID, out float4 uv : INTERP0, out float4 pos : SV_POSITION)"
                           "{ float2 t = float2((id << 1) & 2, id & 2); uv = float4(t, 0, 0); pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); }";
   ID3DBlob* vs_code = nullptr;
   ID3DBlob* errors = nullptr;
   if (FAILED(D3DCompile(vs_source, strlen(vs_source), "vs", nullptr, nullptr, "main", "vs_5_0", 0, 0, &vs_code, &errors)))
   {
      CHECK(false, "screen shadows test VS: %s", errors ? (const char*)errors->GetBufferPointer() : "");
      return;
   }
   ID3D11VertexShader* vs = nullptr;
   device->CreateVertexShader(vs_code->GetBufferPointer(), vs_code->GetBufferSize(), nullptr, &vs);
   vs_code->Release();

   auto make_texture = [&](UINT w, UINT h, UINT slices, const std::vector<float>& data, ID3D11ShaderResourceView** srv) {
      D3D11_TEXTURE2D_DESC desc = { w, h, 1, slices, DXGI_FORMAT_R32_FLOAT, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
      std::vector<D3D11_SUBRESOURCE_DATA> init(slices);
      for (UINT s = 0; s < slices; s++)
         init[s] = { data.data() + size_t(s) * w * h, w * 4, 0 };
      ID3D11Texture2D* texture = nullptr;
      device->CreateTexture2D(&desc, init.data(), &texture);
      device->CreateShaderResourceView(texture, nullptr, srv);
      texture->Release();
   };
   std::vector<float> shadow_data(size_t(M) * M * 4), lit_data(size_t(M) * M * 4, 1.f), cloud_data(64 * 64), depth_data(S * S), rt_data(S * S);
   for (float& v : shadow_data)
      v = 0.3f + 0.4f * uni(rng);
   for (float& v : cloud_data)
      v = 2.f * uni(rng);
   for (float& v : depth_data)
      v = 0.002f + 0.05f * uni(rng);
   for (float& v : rt_data)
      v = uni(rng);
   ID3D11ShaderResourceView *shadow_srv = nullptr, *lit_srv = nullptr, *cloud_srv = nullptr, *depth_srv = nullptr, *rt_srv = nullptr;
   make_texture(M, M, 4, shadow_data, &shadow_srv);
   make_texture(M, M, 4, lit_data, &lit_srv);
   make_texture(64, 64, 1, cloud_data, &cloud_srv);
   make_texture(S, S, 1, depth_data, &depth_srv);
   make_texture(S, S, 1, rt_data, &rt_srv);

   D3D11_SAMPLER_DESC sampler = {};
   sampler.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
   sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
   sampler.ComparisonFunc = D3D11_COMPARISON_LESS_EQUAL;
   sampler.MaxLOD = D3D11_FLOAT32_MAX;
   ID3D11SamplerState* samplers[3] = {};
   device->CreateSamplerState(&sampler, &samplers[0]);
   sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
   sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
   sampler.ComparisonFunc = D3D11_COMPARISON_NEVER;
   device->CreateSamplerState(&sampler, &samplers[1]);
   sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
   sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
   device->CreateSamplerState(&sampler, &samplers[2]);

   // PerViewCB (b1): a camera; matrices as the registers the shaders read (row vectors)
   std::vector<float> per_view(50 * 4, 0.f);
   {
      const V3 eye = { 3, 2, 40 }, target = { 0, 0, 0 }, up = { 0, 0, 1 };
      const V3 f = Normalize(Sub(target, eye)), r = Normalize(Cross(up, f)), u = Cross(f, r);
      Mat4 view;
      view(0, 0) = r.x; view(0, 1) = u.x; view(0, 2) = -f.x;
      view(1, 0) = r.y; view(1, 1) = u.y; view(1, 2) = -f.y;
      view(2, 0) = r.z; view(2, 1) = u.z; view(2, 2) = -f.z;
      view(3, 0) = -Dot(r, eye); view(3, 1) = -Dot(u, eye); view(3, 2) = Dot(f, eye); view(3, 3) = 1;
      Mat4 proj;
      const float n = 0.1f;
      proj(0, 0) = 1.2f;
      proj(1, 1) = 1.6f;
      proj(2, 3) = -1.f;
      proj(3, 2) = n; // Reversed infinite: depth = n / view distance
      const Mat4 inv_view_proj = Inverse(Mul(view, proj));
      std::memcpy(&per_view[6 * 4], view.m, sizeof(view.m));
      std::memcpy(&per_view[30 * 4], inv_view_proj.m, sizeof(inv_view_proj.m));
      per_view[45 * 4 + 0] = eye.x; per_view[45 * 4 + 1] = eye.y; per_view[45 * 4 + 2] = eye.z;
      per_view[47 * 4 + 2] = 12.3f; // cb_worldtime.x
      per_view[48 * 4 + 2] = 1.f; per_view[48 * 4 + 3] = 1.f; // cb_resolutionscale
      per_view[49 * 4 + 0] = float(M); per_view[49 * 4 + 1] = float(M); // cb_parallelshadowmapslicedimensions
   }
   // PerInstanceCB (b2)
   float per_instance[25 * 4] = {};
   {
      FillScreenShadowsCB(per_instance, Normalize({ 0.3f, -0.2f, 0.9f }), Normalize({ 0.2f, 0.1f, 0.95f }));
      per_instance[0] = 10.f; per_instance[1] = 20.f; per_instance[2] = 40.f; per_instance[3] = 80.f; // slice fars
      // Cascades 1..3: the same axis, other scales and offsets
      for (int c = 1; c < 4; c++)
         for (int k = 0; k < 16; k++)
            per_instance[4 + c * 16 + k] = per_instance[4 + k] * (k < 12 ? 1.f / float(1 << c) : 1.f) + (k == 12 ? 0.01f * c : 0.f);
      per_instance[21 * 4 + 3] = 2.f; // cloud altitude
      per_instance[22 * 4 + 0] = 0.0005f; // shadow bias
      per_instance[22 * 4 + 1] = 0.3f; // step size
      per_instance[22 * 4 + 2] = 1.f; // model depth hack
      per_instance[22 * 4 + 3] = 0.7f; // cloud tiling y
      per_instance[23 * 4 + 0] = 0.9f; // cloud tiling x
      per_instance[23 * 4 + 1] = 0.01f; // speed y
      per_instance[23 * 4 + 2] = 0.02f; // speed x
      per_instance[23 * 4 + 3] = 0.05f; // rotation
      per_instance[24 * 4 + 0] = 0.8f; // density
   }
   auto make_cb = [&](const void* data, UINT size) {
      D3D11_BUFFER_DESC desc = { (size + 15) & ~15u, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
      std::vector<uint8_t> padded(desc.ByteWidth, 0);
      std::memcpy(padded.data(), data, size);
      D3D11_SUBRESOURCE_DATA init = { padded.data(), 0, 0 };
      ID3D11Buffer* buffer = nullptr;
      device->CreateBuffer(&desc, &init, &buffer);
      return buffer;
   };
   ID3D11Buffer* per_view_cb = make_cb(per_view.data(), UINT(per_view.size() * 4));
   ID3D11Buffer* per_instance_cb = make_cb(per_instance, sizeof(per_instance));

   D3D11_TEXTURE2D_DESC target_desc = { S, S, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
   ID3D11Texture2D* target = nullptr;
   device->CreateTexture2D(&target_desc, nullptr, &target);
   ID3D11RenderTargetView* target_rtv = nullptr;
   ID3D11ShaderResourceView* target_srv = nullptr;
   device->CreateRenderTargetView(target, nullptr, &target_rtv);
   device->CreateShaderResourceView(target, nullptr, &target_srv);
   target_desc.Format = DXGI_FORMAT_R32_FLOAT;
   target_desc.Usage = D3D11_USAGE_STAGING;
   target_desc.BindFlags = 0;
   target_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
   ID3D11Texture2D* staging = nullptr;
   device->CreateTexture2D(&target_desc, nullptr, &staging);

   auto draw = [&](ID3D11PixelShader* ps, ID3D11ShaderResourceView* shadow, ID3D11ShaderResourceView* rt, ID3D11Buffer* luma_cb) {
      const float clear[4] = { -1, -1, -1, -1 };
      context->ClearRenderTargetView(target_rtv, clear);
      D3D11_VIEWPORT viewport = { 0, 0, (float)S, (float)S, 0, 1 };
      context->RSSetViewports(1, &viewport);
      context->OMSetRenderTargets(1, &target_rtv, nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->IASetInputLayout(nullptr);
      context->VSSetShader(vs, nullptr, 0);
      context->PSSetShader(ps, nullptr, 0);
      ID3D11ShaderResourceView* srvs[4] = { shadow, cloud_srv, depth_srv, rt };
      context->PSSetShaderResources(0, 4, srvs);
      context->PSSetSamplers(0, 3, samplers);
      ID3D11Buffer* cbs[2] = { per_view_cb, per_instance_cb };
      context->PSSetConstantBuffers(1, 2, cbs);
      context->PSSetConstantBuffers(13, 1, &luma_cb);
      context->Draw(3, 0);
      ID3D11RenderTargetView* null_rtv = nullptr;
      context->OMSetRenderTargets(1, &null_rtv, nullptr);
      ID3D11ShaderResourceView* null_srvs[4] = {};
      context->PSSetShaderResources(0, 4, null_srvs);
      context->CopyResource(staging, target);
      D3D11_MAPPED_SUBRESOURCE mapped;
      context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
      std::vector<float> out(S * S);
      for (UINT y = 0; y < S; y++)
         std::memcpy(&out[y * S], (const uint8_t*)mapped.pData + y * mapped.RowPitch, S * 4);
      context->Unmap(staging, 0);
      return out;
   };

   int tested = 0;
   for (const Variant& variant : variants)
   {
      std::vector<uint8_t> original_code;
      {
         FILE* file = fopen((std::string(dump) + variant.original).c_str(), "rb");
         if (!file)
            continue;
         fseek(file, 0, SEEK_END);
         original_code.resize(size_t(ftell(file)));
         fseek(file, 0, SEEK_SET);
         fread(original_code.data(), 1, original_code.size(), file);
         fclose(file);
      }
      ID3DBlob* code = nullptr;
      errors = nullptr;
      RelativeInclude include(L"..\\..\\..\\..\\Shaders\\Dishonored 2");
      if (FAILED(D3DCompileFromFile(variant.replacement, nullptr, &include, "main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors)))
      {
         CHECK(false, "compile %ls: %s", variant.replacement, errors ? (const char*)errors->GetBufferPointer() : "");
         continue;
      }
      // Where LumaSettings.GameSettings.RTSunMode lives (b13), by reflection
      UINT mode_offset = UINT_MAX, luma_size = 0;
      {
         ID3D11ShaderReflection* reflection = nullptr;
         if (SUCCEEDED(D3DReflect(code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&reflection))))
         {
            ID3D11ShaderReflectionConstantBuffer* luma = reflection->GetConstantBufferByName("LumaSettings");
            D3D11_SHADER_BUFFER_DESC luma_desc = {};
            if (SUCCEEDED(luma->GetDesc(&luma_desc)))
            {
               luma_size = luma_desc.Size;
               ID3D11ShaderReflectionVariable* var = luma->GetVariableByIndex(0);
               D3D11_SHADER_VARIABLE_DESC var_desc = {};
               var->GetDesc(&var_desc);
               ID3D11ShaderReflectionType* type = var->GetType();
               ID3D11ShaderReflectionType* game = type->GetMemberTypeByName("GameSettings");
               D3D11_SHADER_TYPE_DESC game_desc = {}, mode_desc = {};
               if (game && SUCCEEDED(game->GetDesc(&game_desc)))
               {
                  ID3D11ShaderReflectionType* mode = game->GetMemberTypeByName("RTSunMode");
                  if (mode && SUCCEEDED(mode->GetDesc(&mode_desc)))
                     mode_offset = var_desc.StartOffset + game_desc.Offset + mode_desc.Offset;
               }
            }
            reflection->Release();
         }
      }
      CHECK(mode_offset != UINT_MAX && luma_size > mode_offset, "RTSunMode not found in LumaSettings");
      if (mode_offset == UINT_MAX)
      {
         code->Release();
         continue;
      }
      ID3D11PixelShader* original = nullptr;
      ID3D11PixelShader* replacement = nullptr;
      device->CreatePixelShader(original_code.data(), original_code.size(), nullptr, &original);
      device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &replacement);
      code->Release();
      ID3D11Buffer* luma_cbs[3] = {};
      for (uint32_t mode = 0; mode < 3; mode++)
      {
         std::vector<uint8_t> luma(luma_size, 0);
         std::memcpy(luma.data() + mode_offset, &mode, 4);
         luma_cbs[mode] = make_cb(luma.data(), luma_size);
      }

      auto compare = [&](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
         float worst = 0.f;
         int off = 0;
         for (UINT i = 0; i < S * S; i++)
         {
            const float d = std::fabs(a[i] - b[i]);
            worst = (std::max)(worst, d);
            off += d > 2e-3f ? 1 : 0;
         }
         printf("  %s: max diff %.2e, %d of %u pixels off by > 2e-3\n", what, worst, off, S * S);
         CHECK(off <= int(S * S) / 200, "%s %s must match", variant.original, what);
      };
      printf("screen shadows %s:\n", variant.original);
      const std::vector<float> game = draw(original, shadow_srv, nullptr, luma_cbs[0]);
      float lo = 1.f, hi = 0.f;
      for (float v : game)
      {
         lo = (std::min)(lo, v);
         hi = (std::max)(hi, v);
      }
      CHECK(hi - lo > 0.2f, "the test inputs must give varied shadows (%.3f..%.3f)", lo, hi);
      compare(game, draw(replacement, shadow_srv, nullptr, luma_cbs[0]), "port, mode 0");
      compare(game, draw(replacement, shadow_srv, rt_srv, luma_cbs[0]), "port, mode 0 with RT bound");
      compare(game, draw(replacement, shadow_srv, nullptr, luma_cbs[1]), "port, mode 1 without RT bound");
      // RT x game: the game's value times the traced visibility of the pixel
      std::vector<float> expected(S * S);
      for (UINT i = 0; i < S * S; i++)
         expected[i] = game[i] * rt_data[i];
      compare(expected, draw(replacement, shadow_srv, rt_srv, luma_cbs[2]), "RT x game");
      // RT: the cloud term (the game's value with a fully lit shadow map) times the traced visibility
      const std::vector<float> clouds = draw(original, lit_srv, nullptr, luma_cbs[0]);
      for (UINT i = 0; i < S * S; i++)
         expected[i] = clouds[i] * rt_data[i];
      compare(expected, draw(replacement, shadow_srv, rt_srv, luma_cbs[1]), "RT");
      for (ID3D11Buffer* b : luma_cbs)
         b->Release();
      original->Release();
      replacement->Release();
      tested++;
   }
   printf("screen shadows shaders: %d of 2 compared with the game's (dump %s)\n", tested, tested ? "found" : "missing");

   staging->Release();
   target_srv->Release();
   target_rtv->Release();
   target->Release();
   per_view_cb->Release();
   per_instance_cb->Release();
   for (ID3D11SamplerState* s : samplers)
      s->Release();
   shadow_srv->Release();
   lit_srv->Release();
   cloud_srv->Release();
   depth_srv->Release();
   rt_srv->Release();
   vs->Release();
}
