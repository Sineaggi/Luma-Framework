#pragma once

// RT sun shadows: a ray traced visibility mask for the game's directional light (sun or moon), traced in the RTAO
// D3D12 submission with the RTAO TLAS and depth, denoised, and handed to D3D11, where the game's screen shadows pass
// (Shaders/Dishonored 2/ScreenShadows_*.hlsl) uses it instead of (or on top of) its cascaded shadow maps. See
// "RT sun shadows" in RTAO_NOTES.md.
//
//  - SunPass (D3D12): Shaders/rtsun_trace.hlsl (RayQuery, rays jittered inside the light's disc for a soft penumbra)
//    + Shaders/rtsun_filter.hlsl (the RTAO denoiser with its own history). It owns its output: a private UAV texture
//    copied, inside the same command list, into a texture shared with D3D11 (Interop::Run is unchanged).
//  - LightCapture (D3D11): the light direction, from the screen shadows draw's own constant buffer (the cascades'
//    depth axis), copied into a staging ring and read back a few frames later without stalling.

#include "rtao_accel.h"
#include "rtao_interop.h"
#include "rtao_pass.h"

#include <d3d11_1.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace RTSun
{
   // Must match the cbuffer in Shaders/rtsun_common.hlsli
   struct SunConstants
   {
      float inv_view_proj[16];
      float prev_view_proj[16];
      float cam_pos[3];
      float t_max;
      float cam_fwd[3];
      uint32_t ray_count;
      float prev_cam_pos[3];
      uint32_t frame;
      float prev_cam_fwd[3];
      uint32_t hist_valid;
      float light_dir[3];
      float cone_tan;
      uint32_t size[2];
      uint32_t depth_size[2];
      uint32_t depth_packed;
      uint32_t reversed_z;
      uint32_t filter_mode;
      float alpha_min;
      float clamp_k;
      float bias;
      uint32_t debug_mode;
      float cutout_opacity;
      uint32_t cutout_mode;
      uint32_t cutout_exact;
      uint32_t pad[2];
   };
   static_assert(sizeof(SunConstants) == 272);

   enum class Output : uint32_t
   {
      Denoised = 0,
      Raw = 1, // No denoiser
      NdotL = 2, // Geometric N.L of the depth normal, no rays (debug)
      History = 3, // Denoiser history: white kept, black rejected (debug)
      Count
   };

   // Matrices use row vectors (clip = float4(world, 1) * view_proj), the game's PerViewCB memory layout
   struct SunSettings
   {
      float view_proj[16] = {};
      float inv_view_proj[16] = {};
      float cam_pos[3] = {};
      float cam_fwd[3] = { 0, 0, 1 };
      bool reversed_z = true;
      float light_dir[3] = { 0, 0, 1 }; // World space, toward the light
      float angular_diameter_deg = 0.75f; // The light's apparent size: 0.53 is the real sun
      uint32_t ray_count = 1;
      float t_max = 2000.f; // World units
      float bias = 0.02f; // World units, plus a view distance proportional term in the shader (as the AO)
      float alpha_min = 1.f / 32.f;
      float clamp_k = 1.f / 16.f;
      // Cutouts, as the AO's (RTAO::PassSettings): stochastic opacity, fast cutouts (per ray), texture exact masks
      float cutout_opacity = 0.5f;
      uint32_t cutout_mode = 0;
      bool cutout_exact = true;
      Output output = Output::Denoised;
      bool reset_history = false;
   };

   class SunPass
   {
   public:
      ~SunPass() { Shutdown(nullptr, 0); }

      bool Init(ID3D12Device5* device, uint64_t timestamp_frequency, std::string& error);
      bool IsInitialized() const { return trace_pso != nullptr; }

      // D3D11 side, any thread (not under the interop lock): (re)creates the textures for a width x height output
      // (the AO size), including the one shared with D3D11. Old D3D12 objects are retired by the next Record.
      bool EnsureTargets(ID3D11Device* device11, uint32_t width, uint32_t height, std::string& error);
      // The shared visibility texture (R16_UNORM, or R8_UNORM), AddRef'd. Null until EnsureTargets succeeded.
      ID3D11ShaderResourceView* GetSRV();

      // Inside an Interop recorder, after the TLAS of the frame was built: trace + filter into the private texture,
      // then copy it into the shared one. Needs EnsureTargets at info.ao_width x info.ao_height. Always full resolution
      // (RTAO Quality's half resolution trace doesn't apply). "cutouts": the RTAO texture exact cutout resources
      // (GeometryCapture::GetCutoutBindings), null: stochastic cutouts only.
      bool Record(ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, D3D12_GPU_VIRTUAL_ADDRESS tlas, const SunSettings& settings, RTAO::Retirer& retirer, std::string& error, const RTAO::CutoutBindings* cutouts = nullptr);
      // Cutout candidates the sun trace resolved, one frame read back a few frames late (as RTAO::AOPass's)
      RTAO::AOPass::CutoutCounters GetCutoutCounters() const { return cutout_counters; }

      double LastGpuMs() const { return timer.LastMs(); }
      uint32_t TraceCount() const { return trace_count.load(); }

      // With a null retirer the GPU must be idle
      void Shutdown(RTAO::Retirer* retirer, uint64_t fence_value);

   private:
      bool EnsureHeap(const RTAO::RecordInfo& info, RTAO::Retirer& retirer, std::string& error);
      void ReleaseTargetsLocked(); // Moves the D3D12 objects to the graveyard

      std::mutex mutex; // Targets (EnsureTargets vs Record)
      ID3D12Device5* device = nullptr;
      ID3D12RootSignature* root_signature = nullptr;
      ID3D12PipelineState* trace_pso = nullptr;
      ID3D12PipelineState* filter_pso = nullptr;
      RTAO::GpuTimer timer;

      static constexpr uint32_t kSlots = 3;
      static constexpr uint32_t kConstantsStride = 512; // CBV alignment 256, constants 272
      ID3D12Resource* constants_upload = nullptr;
      uint8_t* constants_mapped = nullptr;
      // Texture exact cutouts (Shaders/rtao_cutout.hlsli): stand-ins when nothing is bound, and the candidate counters
      ID3D12Resource* dummy_buffer = nullptr;
      ID3D12Resource* counters = nullptr; // 256 bytes, COMMON outside the trace
      ID3D12Resource* counters_readback = nullptr; // kSlots x 16 bytes
      bool counters_pending[kSlots] = {};
      uint32_t counters_last[3] = {};
      bool counters_have_last = false;
      RTAO::AOPass::CutoutCounters cutout_counters;
      ID3D12Resource* bound_atlas = nullptr; // The atlas the heap's descriptor 10 points at (identity only)

      // Targets (under "mutex")
      uint32_t width = 0, height = 0;
      uint32_t targets_version = 0; // Bumped by EnsureTargets when it re-creates them
      DXGI_FORMAT vis_format = DXGI_FORMAT_UNKNOWN;
      ID3D12Resource* raw = nullptr; // R32_FLOAT
      ID3D12Resource* vis = nullptr; // vis_format, UAV
      ID3D12Resource* history[2] = {}; // R32G32_UINT
      ID3D12Resource* shared12 = nullptr; // vis_format, shared with D3D11 (COMMON outside the recorder)
      ID3D11Texture2D* shared11 = nullptr;
      ID3D11ShaderResourceView* srv11 = nullptr;
      std::vector<IUnknown*> graveyard; // D3D12 objects to retire at the next Record

      // Descriptors (recorder thread). Per history parity p, 5 from p * 5: [depth SRV, raw UAV, vis UAV, hist prev, hist cur],
      // then [10] the cutout atlas SRV
      ID3D12DescriptorHeap* heap = nullptr;
      uint32_t descriptor_size = 0;
      uint32_t heap_generation = UINT32_MAX; // Interop targets generation the heap was made for
      uint32_t heap_targets_version = UINT32_MAX;

      uint32_t parity = 0;
      bool has_history = false;
      uint32_t frame = 0;
      float prev_view_proj[16] = {};
      float prev_cam_pos[3] = {};
      float prev_cam_fwd[3] = {};
      std::atomic<uint32_t> trace_count = 0;
   };

   // The screen shadows pass's PerInstanceCB (b2 of PS 0xF324B47B / 0x6C8C2397), see RTAO_NOTES.md
   constexpr uint32_t kScreenShadowsCBBytes = 400;
   constexpr uint32_t kShadowProjectionsOffset = 16; // float4x4 cb_shadowprojections[5], cascade 0 first
   constexpr uint32_t kAtmosphereSunOffset = 336; // float3 cb_atm_sky_sundirection

   // From a copy of that cbuffer: the direction toward the shadow casting light, as the depth axis of cascade 0's
   // light space (the gradient of light space z), oriented to the side above the horizon (up_axis: 2 for the game's
   // z up). "atmosphere_sun" gets cb_atm_sky_sundirection (normalized, may be the sun when the moon casts). False if
   // the data is unusable.
   bool ComputeLightDirection(const void* cb, uint32_t size, int up_axis, float out_dir[3], float out_atmosphere_sun[3]);

   class LightCapture
   {
   public:
      ~LightCapture() { Shutdown(); }
      bool Init(ID3D11Device* device, std::string& error);
      bool IsInitialized() const { return slots[0].staging != nullptr; }
      void Shutdown();

      // At the game's screen shadows draw, on the context recording it (any thread): copies the bound PS b2 (with its
      // PSSetConstantBuffers1 offset) into the staging ring, once per "frame".
      void OnScreenShadowsDraw(ID3D11DeviceContext* context, uint32_t frame);
      // Immediate context, once per frame: reads back the copies at least kLatency frames old (never waits).
      void Poll(ID3D11DeviceContext* immediate_context, uint32_t frame);

      // The latest light direction (toward the light) and the frame its copy was made on. False if none yet.
      bool GetDirection(float out_dir[3], uint32_t* out_frame = nullptr, float* out_atmosphere_sun = nullptr);
      int up_axis = 2;

      static constexpr uint32_t kSlots = 4;
      static constexpr uint32_t kLatency = 2;

   private:
      struct Slot
      {
         ID3D11Buffer* staging = nullptr;
         uint32_t frame = 0;
         uint32_t bytes = 0;
         bool pending = false;
      };
      std::mutex mutex;
      Slot slots[kSlots];
      uint32_t head = 0;
      uint32_t last_copy_frame = UINT32_MAX;
      bool valid = false;
      float direction[3] = {};
      float atmosphere_sun[3] = {};
      uint32_t direction_frame = 0;
   };
}
