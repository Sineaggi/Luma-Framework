#pragma once

// RTAO pass (D3D12): inline ray query AO from the reconstructed depth (Shaders/rtao_trace.hlsl) followed by the
// cross-bilateral + temporal denoiser (Shaders/rtao_filter.hlsl), written into the interop's AO texture.
// Used from an Interop::Run recorder, after the TLAS for the frame has been built.

#include "rtao_accel.h"
#include "rtao_interop.h"

#include <cstdint>
#include <string>

namespace RTAO
{
   // Must match the cbuffer in Shaders/rtao_common.hlsli
   struct PassConstants
   {
      float inv_view_proj[16];
      float prev_view_proj[16];
      float cam_pos[3];
      float radius;
      float cam_fwd[3];
      uint32_t ray_count;
      float prev_cam_pos[3];
      uint32_t frame;
      float prev_cam_fwd[3];
      uint32_t hist_valid;
      uint32_t size[2];
      uint32_t depth_size[2];
      uint32_t depth_packed;
      uint32_t reversed_z;
      uint32_t filter_mode;
      float alpha_min;
      float clamp_k;
      float bias;
      uint32_t debug_mode;
      float pad0;
   };
   static_assert(sizeof(PassConstants) == 240);

   enum class FilterMode : uint32_t
   {
      Off = 0,
      Spatial = 1,
      Temporal = 2,
   };

   // Matrices use row vectors (clip = float4(world, 1) * view_proj), i.e. the game's PerViewCB memory layout.
   struct PassSettings
   {
      float view_proj[16] = {};
      float inv_view_proj[16] = {};
      float cam_pos[3] = {};
      float cam_fwd[3] = { 0, 0, 1 };
      bool reversed_z = true;
      float radius = 1.5f; // World units
      uint32_t ray_count = 4;
      float bias = 0.02f; // World units, plus a view distance proportional term in the shader
      FilterMode filter_mode = FilterMode::Temporal;
      float alpha_min = 1.f / 32.f;
      float clamp_k = 1.f / 16.f;
      uint32_t debug_mode = 0; // 1 reconstructed normal Y (raw), 2 history rejection (filtered), 3 BLAS coverage (raw)
      bool reset_history = false;
   };

   // GPU time of a span of a command list, read back when the frame slot comes around again (no stall)
   class GpuTimer
   {
   public:
      ~GpuTimer() { Shutdown(); }
      bool Init(ID3D12Device* device, uint64_t timestamp_frequency, std::string& error);
      // Call Collect, then Begin ... End, once per frame slot use
      void Collect(uint32_t frame_slot);
      void Begin(ID3D12GraphicsCommandList* list, uint32_t frame_slot);
      void End(ID3D12GraphicsCommandList* list, uint32_t frame_slot);
      double LastMs() const { return last_ms; }
      void Shutdown();

   private:
      static constexpr uint32_t kSlots = 3;
      ID3D12QueryHeap* heap = nullptr;
      ID3D12Resource* readback = nullptr;
      uint64_t frequency = 0;
      bool pending[kSlots] = {};
      double last_ms = 0.0;
   };

   class AOPass
   {
   public:
      ~AOPass() { Shutdown(nullptr, 0); }

      bool Init(ID3D12Device5* device, std::string& error);
      bool IsInitialized() const { return trace_pso != nullptr; }
      // Records trace + filter into info.ao (left in UNORDERED_ACCESS state)
      bool Record(ID3D12GraphicsCommandList4* list, const RecordInfo& info, D3D12_GPU_VIRTUAL_ADDRESS tlas, const PassSettings& settings, Retirer& retirer, std::string& error);
      // With a null retirer the GPU must be idle
      void Shutdown(Retirer* retirer, uint64_t fence_value);

   private:
      bool EnsureTargets(const RecordInfo& info, Retirer& retirer, std::string& error);

      ID3D12Device5* device = nullptr;
      ID3D12RootSignature* root_signature = nullptr;
      ID3D12PipelineState* trace_pso = nullptr;
      ID3D12PipelineState* filter_pso = nullptr;

      static constexpr uint32_t kSlots = 3;
      static constexpr uint32_t kConstantsStride = 256;
      ID3D12Resource* constants_upload = nullptr;
      uint8_t* constants_mapped = nullptr;

      // Re-created (and the old one retired) whenever the targets change, so no descriptor the GPU may still
      // read is ever overwritten. Layout per history parity p, 5 descriptors from p * 5:
      // [depth SRV, raw UAV, out UAV, history prev UAV, history cur UAV]
      ID3D12DescriptorHeap* heap = nullptr;
      uint32_t descriptor_size = 0;
      ID3D12Resource* raw = nullptr; // R32_FLOAT
      ID3D12Resource* history[2] = {}; // R32G32_UINT
      uint32_t width = 0, height = 0;
      uint32_t generation = UINT32_MAX;

      uint32_t parity = 0;
      bool has_history = false;
      uint32_t frame = 0;
      float prev_view_proj[16] = {};
      float prev_cam_pos[3] = {};
      float prev_cam_fwd[3] = {};
   };
}
