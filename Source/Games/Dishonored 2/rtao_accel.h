#pragma once

// RTAO acceleration structures: engine agnostic, D3D12 only. Ported in spirit from the user's Fallout 4 RT core
// (f4-mods/rtcore/blas_cache.h, tlas_build.h) with the Creation Engine parts left out:
//  - BlasCache: one bottom level AS per unique mesh, keyed by a caller supplied 64 bit id. Built once (opaque,
//    PREFER_FAST_TRACE), kept until evicted.
//  - TlasBuilder: instances retained in a sliding window keyed by a caller supplied instance id, rebuilt into a
//    top level AS every frame (PREFER_FAST_BUILD). AO needs occluders that are off screen (behind and beside the
//    shading point), and the game only draws what is in the frustum, so an instance is kept for kRetainFrames
//    after it was last seen (F4 RtAo: without this, facing a wall dropped the room behind and the AO flashed).
//  - Retirer: releases D3D12 objects once the GPU has finished the submission that last used them.
//
// Threading: TlasBuilder::AddInstance is thread safe (the game's draws may be recorded on several threads).
// Everything else runs on the thread that records the D3D12 command list (inside Interop::Run's recorder).

#include <d3d12.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RTAO
{
   // TLAS instance masks (the ray's mask selects what it can hit). Every instance is forced opaque, except cutouts
   // (alpha tested geometry: leaves, grates, fences), which are forced non opaque and get kInstanceIdCutoutBit in their
   // InstanceID, so the trace can resolve them in its candidate loop (see Shaders/rtao_trace.hlsl).
   constexpr uint8_t kInstanceMaskOpaque = 0x01;
   constexpr uint8_t kInstanceMaskCutout = 0x02;
   constexpr uint32_t kInstanceIdCutoutBit = 1u << 23;

   class Retirer
   {
   public:
      ~Retirer() { ReleaseAll(); }
      // "object" is released once the own-fence reaches "fence_value". Takes ownership of one reference.
      void Retire(IUnknown* object, uint64_t fence_value);
      void Collect(uint64_t completed_fence_value);
      // Only when the GPU is idle
      void ReleaseAll();
      size_t Pending() const { return pending.size(); }

   private:
      std::vector<std::pair<uint64_t, IUnknown*>> pending;
   };

   // CPU side triangle mesh (positions are float3, "vertex_stride" bytes apart, 32 bit indices)
   struct CpuMesh
   {
      const float* positions = nullptr;
      uint32_t vertex_count = 0;
      uint32_t vertex_stride = 12;
      const uint32_t* indices = nullptr;
      uint32_t index_count = 0;
   };

   // GPU resident triangles (e.g. game vertex/index data copied into a shared buffer). The buffers must stay
   // alive and in a NON_PIXEL_SHADER_RESOURCE compatible state until the build has executed.
   struct GpuMesh
   {
      D3D12_GPU_VIRTUAL_ADDRESS vertices = 0;
      uint32_t vertex_count = 0;
      uint32_t vertex_stride = 12;
      DXGI_FORMAT vertex_format = DXGI_FORMAT_R32G32B32_FLOAT; // Also R16G16B16A16_FLOAT / _SNORM etc (DXR 1.1 list)
      D3D12_GPU_VIRTUAL_ADDRESS indices = 0; // 0: non indexed
      uint32_t index_count = 0;
      DXGI_FORMAT index_format = DXGI_FORMAT_R32_UINT;
   };

   struct BuildContext
   {
      ID3D12Device5* device;
      ID3D12GraphicsCommandList4* list;
      Retirer* retirer;
      uint64_t submit_fence_value; // From RecordInfo
   };

   class BlasCache
   {
   public:
      ~BlasCache() { Clear(nullptr, 0); }

      bool Has(uint64_t key) const { return entries.find(key) != entries.end(); }
      D3D12_GPU_VIRTUAL_ADDRESS Get(uint64_t key) const;
      size_t Count() const { return entries.size(); }
      uint64_t ResultBytes() const { return result_bytes; }

      // Records the build into ctx.list. Returns true if the key is (now) present. Call FlushBarrier() after a batch
      // of builds and before building the TLAS.
      bool BuildFromCpu(const BuildContext& ctx, uint64_t key, const CpuMesh& mesh, std::string& error);
      bool BuildFromGpu(const BuildContext& ctx, uint64_t key, const GpuMesh& mesh, std::string& error);
      void FlushBarrier(ID3D12GraphicsCommandList4* list);

      void Evict(uint64_t key, Retirer* retirer, uint64_t fence_value);
      // With a null retirer the GPU must be idle
      void Clear(Retirer* retirer, uint64_t fence_value);

   private:
      bool Build(const BuildContext& ctx, uint64_t key, const D3D12_RAYTRACING_GEOMETRY_DESC& geometry, std::string& error);

      struct Entry
      {
         ID3D12Resource* result = nullptr;
         uint64_t bytes = 0;
      };
      std::unordered_map<uint64_t, Entry> entries;
      uint64_t result_bytes = 0;
      bool barrier_owed = false;
   };

   class TlasBuilder
   {
   public:
      static constexpr uint32_t kRetainFrames = 180;
      // An instance only earns kRetainFrames once it was seen in kPromoteFrames different frames. Until then it is kept
      // kYoungRetainFrames. A moving object gets a new identity (its transform) every frame, so its copies never get
      // promoted and age out at once: no trail of copies, no pile-up. A static object in view is promoted quickly.
      static constexpr uint32_t kPromoteFrames = 4;
      static constexpr uint32_t kYoungRetainFrames = 2;

      ~TlasBuilder() { Clear(nullptr, 0); }

      // Advances the frame and prunes instances not seen for more than "retain_frames"
      void BeginFrame();
      // Thread safe. "object_to_world" is a row major 3x4 (the D3D12 instance transform layout).
      // "mask": kInstanceMaskOpaque, or kInstanceMaskCutout for alpha tested geometry (non opaque in the TLAS).
      void AddInstance(uint64_t instance_key, uint64_t blas_key, const float object_to_world[3][4], uint8_t mask = kInstanceMaskOpaque);
      void RemoveInstance(uint64_t instance_key);
      size_t RetainedCount();

      // Builds the TLAS from every retained instance whose BLAS is in "blases" (the others are skipped). "out" stays
      // valid until the next Build. An empty scene still builds a valid (empty) TLAS.
      bool Build(const BuildContext& ctx, const BlasCache& blases, uint32_t frame_slot, D3D12_GPU_VIRTUAL_ADDRESS& out, uint32_t& instance_count, std::string& error);

      void Clear(Retirer* retirer, uint64_t fence_value);
      uint32_t LastCutoutCount() const { return last_cutout_count; } // Cutout instances in the last Build
      uint32_t retain_frames = kRetainFrames;
      uint32_t promote_frames = kPromoteFrames;
      uint32_t young_retain_frames = kYoungRetainFrames;

   private:
      struct Instance
      {
         float transform[3][4];
         uint64_t blas_key;
         uint32_t last_frame;
         uint32_t seen_frames;
         uint8_t mask;
      };
      std::mutex mutex;
      std::unordered_map<uint64_t, Instance> instances;
      uint32_t frame = 0;
      uint32_t last_cutout_count = 0;

      static constexpr uint32_t kSlots = 3; // Matches Interop's frames in flight
      ID3D12Resource* instance_upload[kSlots] = {};
      uint64_t instance_upload_capacity[kSlots] = {};
      ID3D12Resource* scratch = nullptr;
      uint64_t scratch_capacity = 0;
      ID3D12Resource* result = nullptr;
      uint64_t result_capacity = 0;
      std::vector<D3D12_RAYTRACING_INSTANCE_DESC> descs;
   };
}
