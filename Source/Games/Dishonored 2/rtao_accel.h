#pragma once

// RTAO acceleration structures: engine agnostic, D3D12 only. Ported from f4-mods by Sineaggi
// (Fallout 4 RT core: rtcore/blas_cache.h, tlas_build.h) with the Creation Engine parts left out:
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
#include <unordered_set>
#include <utility>
#include <vector>

namespace RTAO
{
   // TLAS instance masks (the ray's mask selects what it can hit). Every instance is forced opaque, except cutouts
   // (alpha tested geometry: leaves, grates, fences), which are forced non opaque and get kInstanceIdCutoutBit in their
   // InstanceID, so the trace can resolve them in its candidate loop (see Shaders/rtao_trace.hlsl).
   constexpr uint8_t kInstanceMaskOpaque = 0x01;
   constexpr uint8_t kInstanceMaskCutout = 0x02;
   // Retained instances the game only ever drew for the camera, never into a shadow map (window glass, light shaft meshes):
   // they occlude the AO but the RT sun can leave them out, like the game's own shadows (RTSun "shadow casters only")
   constexpr uint8_t kInstanceMaskCameraOnly = 0x04;
   constexpr uint8_t kInstanceMaskCameraOnlyCutout = 0x08;
   constexpr uint32_t kInstanceIdCutoutBit = 1u << 23;
   // The low 23 bits of a cutout's InstanceID: its texture exact cutout record (rtao_alpha.h), or this (none: the trace
   // falls back to the stochastic opacity)
   constexpr uint32_t kNoCutoutRecord = kInstanceIdCutoutBit - 1;

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

   // Deforming meshes (skinned characters): one BLAS per caller identity, built (ALLOW_UPDATE, PREFER_FAST_BUILD) the
   // first time and refit in place every later frame while the topology stays the same. A full rebuild every
   // kMaxRefits frames keeps the BVH from degrading when the pose drifts far from the one it was built for (or when an
   // identity ends up on another character drawing the same mesh). Entries not updated for kEvictFrames are released.
   // Scratch memory is one buffer sub-allocated per frame (sized to the previous frames' need, a too-small frame uses
   // a temporary one), so builds in a frame don't serialize on a shared scratch.
   class DynamicBlasSet
   {
   public:
      static constexpr uint32_t kMaxRefits = 30;
      static constexpr uint32_t kEvictFrames = 2;
      // Entries updated with "retain" (movable objects with a DYNAMIC vertex buffer: openable windows, doors) are kept
      // this long after their last draw, with their last pose (ForEachRetained), instead of vanishing on the frames the
      // game doesn't draw them (out of the view and of the shadow cascades). At most kMaxRetained, oldest evicted first.
      static constexpr uint32_t kRetainFrames = 600;
      static constexpr size_t kMaxRetained = 512;

      ~DynamicBlasSet() { Clear(nullptr, 0); }

      void BeginFrame();
      // Records a build or refit into ctx.list (the vertices/indices must be readable by the build). Returns the BLAS
      // address, 0 on failure. Call EndFrame() after the frame's updates, before the TLAS build.
      D3D12_GPU_VIRTUAL_ADDRESS Update(const BuildContext& ctx, uint64_t key, const GpuMesh& mesh, bool& refit, std::string& error, bool retain = false, uint8_t source = 0);
      // Retained entries not updated this frame (call after EndFrame): fn(blas address, source)
      template <class F>
      void ForEachRetained(F&& fn) const
      {
         for (const auto& pair : entries)
            if (pair.second.retain && pair.second.last_frame != frame && pair.second.result)
               fn(pair.second.result->GetGPUVirtualAddress(), pair.second.source);
      }
      size_t RetainedCount() const { return retained_count; }
      // Held entries (not drawn this frame) lose "retain": EndFrame evicts them like any other entry
      void DropRetained()
      {
         for (auto& pair : entries)
            if (pair.second.last_frame != frame)
               pair.second.retain = false;
      }
      // UAV barrier for the frame's builds, evicts unused entries
      void EndFrame(const BuildContext& ctx);
      void Clear(Retirer* retirer, uint64_t fence_value);

      size_t Count() const { return entries.size(); }
      uint64_t ResultBytes() const { return result_bytes; }

   private:
      bool AllocateScratch(const BuildContext& ctx, uint64_t size, D3D12_GPU_VIRTUAL_ADDRESS& out, std::string& error);

      struct Entry
      {
         ID3D12Resource* result = nullptr;
         uint64_t bytes = 0;
         uint64_t update_scratch = 0;
         uint32_t vertex_count = 0, index_count = 0;
         DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
         D3D12_GPU_VIRTUAL_ADDRESS indices = 0;
         uint32_t last_frame = 0;
         uint32_t refits = 0;
         bool retain = false;
         uint8_t source = 0;
      };
      size_t retained_count = 0; // Retained entries not drawn in the last frame
      std::unordered_map<uint64_t, Entry> entries;
      uint64_t result_bytes = 0;
      uint32_t frame = 0;
      bool barrier_owed = false;
      ID3D12Resource* scratch = nullptr;
      uint64_t scratch_capacity = 0, scratch_used = 0, scratch_peak = 0;
   };

   // Where a retained instance was seen (TlasBuilder::AddInstance "source", OR'ed over its life)
   constexpr uint8_t kInstanceSourceMain = 0x01; // The main view
   constexpr uint8_t kInstanceSourceShadow = 0x02; // A shadow map draw
   // More bits OR'ed into "source", for the debug scene view (RT Sun Output "Scene view"; stored in the TLAS instance's
   // InstanceContributionToHitGroupIndex, which inline ray queries don't otherwise use). Must match rtsun_trace.hlsl.
   constexpr uint8_t kInstanceFlagFrameMesh = 0x04; // Per frame mesh (AddFrameInstance)
   constexpr uint8_t kInstanceFlagMvp = 0x08; // Matrix recovered from the MVP
   constexpr uint8_t kInstanceFlagSkinned = 0x10; // Skinned (per frame mesh)

   // TLAS retention diagnostics, from the last BeginFrame (and the AddInstance calls before it)
   struct RetentionStats
   {
      uint32_t young = 0, promoted = 0; // Retained instances
      uint32_t main_only = 0, shadow_only = 0, both = 0; // By where they were seen
      uint32_t evicted_young = 0, evicted_promoted = 0; // Pruned by that BeginFrame
      uint32_t added_new = 0; // AddInstance calls since the previous BeginFrame that created an instance
      uint32_t shadow_young = 0; // Retained shadow map instances not confirmed static yet (old rule)
      uint32_t shadow_static = 0; // Shadow map instances kept as static world
      uint32_t moved = 0; // Shadow map copies dropped as the previous position of a moving object (this BeginFrame)
      uint32_t capped = 0; // Static shadow map instances dropped by the cap (this BeginFrame)
   };

   class TlasBuilder
   {
   public:
      // ~10 s at 60 fps. A static caster the game stops drawing (out of the view and of the shadow cascades) stays this
      // long; the shadow maps redraw what is near the camera every frame, so this only bounds what was left behind.
      static constexpr uint32_t kRetainFrames = 600;
      // An instance only earns kRetainFrames once it was seen in kPromoteFrames different frames. Until then it is kept
      // kYoungRetainFrames. A moving object gets a new identity (its transform) every frame, so its copies never get
      // promoted and age out at once: no trail of copies, no pile-up. A static object in view is promoted quickly.
      static constexpr uint32_t kPromoteFrames = 4;
      static constexpr uint32_t kYoungRetainFrames = 2;
      // Instances seen in a shadow map (see BeginFrame): confirmed static after kShadowPromoteFrames sightings (a moving
      // object is seen at most twice per key), then kept kShadowRetainFrames after their last draw (~2 min at 60 fps):
      // casters out of the camera's view; until then kept kShadowYoungRetainFrames (cascades may be time sliced).
      static constexpr uint32_t kShadowPromoteFrames = 3;
      static constexpr uint32_t kShadowRetainFrames = 7200;
      static constexpr uint32_t kShadowYoungRetainFrames = 4;
      // A retained instance whose BLAS has been missing this many builds (mesh freed, e.g. a level change) is dropped
      static constexpr uint32_t kMissingBlasFrames = 120;
      // Shadow map retention (BeginFrame, shadow_mode 0): a redraw batch is this many new shadow map instances in one
      // readback; a moved copy is within kMovingDistance of the previous one, which stopped being drawn within
      // kMovingWindowFrames; at most kMaxShadowStatic static shadow map instances (the least recently drawn go first)
      static constexpr uint32_t kShadowBatch = 30;
      static constexpr uint32_t kMovingWindowFrames = 4;
      static constexpr float kMovingDistance = 2.f;
      static constexpr size_t kMaxShadowStatic = 60000;

      ~TlasBuilder() { Clear(nullptr, 0); }

      // Advances the frame and prunes instances not seen for more than "retain_frames"
      void BeginFrame();
      // Thread safe. "object_to_world" is a row major 3x4 (the D3D12 instance transform layout).
      // "mask": kInstanceMaskOpaque, or kInstanceMaskCutout for alpha tested geometry (non opaque in the TLAS).
      // "cutout_record": cutouts only, see kNoCutoutRecord.
      void AddInstance(uint64_t instance_key, uint64_t blas_key, const float object_to_world[3][4], uint8_t mask = kInstanceMaskOpaque, uint32_t cutout_record = kNoCutoutRecord, uint8_t source = kInstanceSourceMain, uint8_t shadow_slice = 0xFF);
      // Diagnostic capture: every retained instance (BLAS key, sources, lowest shadow map slice, position, frames)
      struct InstanceInfo
      {
         uint64_t blas_key;
         uint8_t sources, min_slice;
         float pos[3];
         uint32_t first_frame, last_frame, seen_frames;
      };
      std::vector<InstanceInfo> Snapshot();
      // The last Build's instances in TLAS index order (CommittedInstanceIndex): retained ones, then the frame instances
      // (sources = their source | kInstanceFlagFrameMesh, no BLAS key / position)
      std::vector<InstanceInfo> LastBuildOrder() { const std::lock_guard lock(mutex); return last_build_order; }
      RetentionStats GetRetentionStats();
      void RemoveInstance(uint64_t instance_key);
      size_t RetainedCount();
      // An instance for the current frame only (dropped at the next BeginFrame), with its BLAS given directly (e.g. from
      // a DynamicBlasSet). Not thread safe: call from the recording thread, between BeginFrame and Build.
      void AddFrameInstance(D3D12_GPU_VIRTUAL_ADDRESS blas, const float object_to_world[3][4], uint8_t mask = kInstanceMaskOpaque, uint8_t source = kInstanceSourceMain);
      size_t FrameInstanceCount() const { return frame_instances.size(); }

      // Builds the TLAS from every retained instance whose BLAS is in "blases" (the others are skipped). "out" stays
      // valid until the next Build. An empty scene still builds a valid (empty) TLAS.
      bool Build(const BuildContext& ctx, const BlasCache& blases, uint32_t frame_slot, D3D12_GPU_VIRTUAL_ADDRESS& out, uint32_t& instance_count, std::string& error);

      void Clear(Retirer* retirer, uint64_t fence_value);
      uint32_t LastCutoutCount() const { return last_cutout_count; } // Cutout instances in the last Build
      uint32_t LastMissingBlas() const { return last_missing_blas; } // Retained instances left out of the last Build: their BLAS isn't built (yet)
      uint32_t retain_frames = kRetainFrames;
      uint32_t promote_frames = kPromoteFrames;
      uint32_t young_retain_frames = kYoungRetainFrames;
      uint32_t shadow_young_retain_frames = kShadowYoungRetainFrames; // Old rule only
      uint32_t shadow_mode = 0; // 0: static world until unloaded, moving copies dropped; 1: the old rule (test toggle)
      std::atomic<bool> drop_shadow_request = false; // Next BeginFrame drops every instance a shadow map draw touched (the main view re-adds what it sees)

   private:
      struct Instance
      {
         float transform[3][4];
         uint64_t blas_key;
         uint32_t last_frame;
         uint32_t seen_frames;
         uint32_t cutout_record;
         uint8_t mask;
         uint8_t sources;
         uint32_t missing_frames; // Consecutive builds its BLAS was missing
         uint32_t first_frame;
         uint32_t shadow_first_frame; // The frame its first shadow map sighting was added
         uint8_t min_slice; // Lowest shadow map array slice it was drawn in (0xFF: none)
         bool moved; // A shadow map copy left behind by a moving object
         bool batch; // Arrived in a shadow map redraw batch (never taken for a move)
      };
      uint32_t added_new_shadow = 0;
      std::unordered_set<uint64_t> shadow_meshes; // BLAS keys ever drawn into a shadow map (sun casters, any instance)
      std::vector<InstanceInfo> last_build_order;
      std::vector<uint64_t> batch_candidates; // Shadow map instances created since the last BeginFrame
      RetentionStats retention_stats;
      uint32_t added_new = 0;
      std::mutex mutex;
      std::unordered_map<uint64_t, Instance> instances;
      std::vector<D3D12_RAYTRACING_INSTANCE_DESC> frame_instances;
      uint32_t frame = 0;
      uint32_t last_cutout_count = 0;
      uint32_t last_missing_blas = 0;
      uint32_t last_dropped_missing = 0;

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
