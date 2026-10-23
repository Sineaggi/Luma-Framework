#pragma once

// RTAO texture exact cutouts: alpha tested geometry (leaves, grates, fences, hair) occludes where its texture's alpha
// passes the game's test, per texel, instead of with a fixed probability. Modelled on the user's Fallout 4 code
// (f4-mods/rtcore/alpha_atlas.h: one R8 mask per material in a private Texture2DArray, crossing as a shared strip).
//
// Design (and why):
//  - What the game tests (RTAO_NOTES.md "Texture exact cutouts", ReflectPixelShader): discard if
//    tex[alpha_slot].a at (TEXCOORD0 * cb_layer0_scalebias.xy + .zw) < reference, the reference being
//    cb_defaultalphatest / cb_hairalphatest or an immediate (0.5 in the vegetation shaders).
//  - Masks: per material (the alpha texture resource + channel), the first time a cutout draw binds it, a D3D11 compute
//    pass (Luma_RTAO.hlsl alpha_strip_cs) samples that texture's channel, mip aware, into a 256x256 R8 tile, which is
//    copied into one tile of a strip shared with D3D12. Game textures (BC formats) can't be shared, and a shared texture
//    only opens in D3D11 as a render target (no UAV), hence the private tile + copy. The strip is a ring of
//    kStripTiles tiles of which at most kStripBudget are handed out between two D3D12 passes, so a tile is copied out
//    before it is rewritten even when its command list executes a few frames late (same rule as the skinned ring).
//    The D3D12 side (inside the recorder, after the strip's D3D11 work executed) copies the tile into a slice of a
//    private Texture2DArray atlas. First come, first served: kAtlasSlices slices, a slice is freed when its texture is
//    destroyed, the overflow is counted (those materials stay on the stochastic fallback).
//    The mask holds the raw channel, not the test result: the reference and the uv scale/bias live in the draw's
//    cbuffer, which only the GPU can read, so they come per instance through the capture ring (capture_instances_cs).
//  - Records: per (material, geometry, scale/bias, reference) a 48 byte record in a StructuredBuffer (CutoutGpuRecord):
//    where the mesh's indices and TEXCOORD0 are in the geometry arena (already shared with D3D12, the BLAS reads it
//    too), the uv scale/bias, the reference, the atlas slice. A cutout TLAS instance carries its record id in the low
//    23 bits of its InstanceID. The slice is resolved when the table is uploaded, so a material getting its mask (or
//    losing it) only re-uploads the table. Ids unused for kRecordStaleFrames (longer than the TLAS keeps an instance) are
//    recycled when the table is full.
//  - Trace (Shaders/rtao_cutout.hlsli): on a cutout candidate, fetch the 3 indices and uvs, interpolate with the
//    candidate's barycentrics, apply the scale/bias, sample the slice (bilinear, wrap), commit if >= reference. A
//    material without a mask (pending, overflow, unreadable uv) keeps the stochastic opacity.
//
// Threading: every public method locks the object's mutex (draw threads, the destroy hook and the recorder all call in).
// GeometryCapture may call in while holding its own lock (lock order: GeometryCapture, then this), never the reverse.

#include "rtao_accel.h"
#include "rtao_interop.h"

#include <d3d11_1.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace RTAO
{
   // Where a cutout mesh's triangles and uvs are in the geometry arena (byte offsets from the arena start)
   struct CutoutGeometry
   {
      uint32_t index_offset = 0; // The draw's first index (the BLAS's primitive 0)
      uint32_t uv_offset = 0; // TEXCOORD0 of the BLAS's vertex 0
      uint32_t uv_stride = 0;
      uint32_t uv_code = 0; // VertexFormatCode of TEXCOORD0
      uint32_t index32 = 0; // 1: 32 bit indices
   };

   // Must match CutoutRecord in Shaders/rtao_cutout.hlsli
   struct CutoutGpuRecord
   {
      uint32_t index_offset;
      uint32_t uv_offset;
      uint32_t uv_stride_index32; // stride | index32 << 31
      uint32_t uv_code_slice; // VertexFormatCode (low 16 bits) | atlas slice << 16 (0xFFFF: no mask)
      float scalebias[4];
      float alpha_ref;
      uint32_t pad[3];
   };
   static_assert(sizeof(CutoutGpuRecord) == 48);

   // What the trace needs (AOPass::Record)
   struct CutoutBindings
   {
      bool exact = false; // Texture exact cutouts on (records/atlas may still be empty: everything falls back)
      D3D12_GPU_VIRTUAL_ADDRESS records = 0; // StructuredBuffer<CutoutRecord>, GENERIC_READ (upload heap)
      ID3D12Resource* arena = nullptr; // The geometry arena, COMMON at handoff: the pass transitions it around the trace
      ID3D12Resource* atlas = nullptr; // Texture2DArray R8_UNORM, NON_PIXEL_SHADER_RESOURCE
      uint32_t atlas_slices = 0;
   };

   struct CutoutStats
   {
      uint64_t materials = 0, materials_ready = 0, materials_pending = 0; // Live
      uint64_t strips_rendered = 0, strips_copied = 0, strips_deferred = 0, strips_abandoned = 0; // Deferred: over the per pass budget, retried later
      uint64_t atlas_slices_used = 0, atlas_overflow = 0; // Overflow: materials refused a slice
      uint64_t unsupported = 0; // Materials whose alpha texture view isn't a plain Texture2D (stay on the fallback)
      uint64_t records = 0, record_overflow = 0, record_uploads = 0;
      uint64_t draws_exact = 0; // Cutout draws captured with a material (texture exact capable)
      uint64_t draws_no_material = 0; // Cutout draws on the fallback: shader form not replayable, no texture/uv bound, uv not readable
   };

   class CutoutMaterials
   {
   public:
      static constexpr uint32_t kTileSize = 256;
      static constexpr uint32_t kStripTiles = 64; // 256 x 16384 R8: 4 MB
      static constexpr uint32_t kStripBudget = 16; // New masks between two D3D12 passes
      static constexpr uint32_t kAtlasSlices = 1024; // 64 MB
      static constexpr uint32_t kMaxRecords = 65536; // 3 MB per frame slot
      static constexpr uint32_t kRecordStaleFrames = TlasBuilder::kRetainFrames + 60;
      static constexpr uint32_t kPendingTimeoutFrames = 30; // A strip that never executed (list dropped): render it again
      static constexpr uint32_t kRefreshFrames = 240; // Rendered a second time once, after this (streamed in mips)
      static constexpr uint32_t kNoTile = 0xFFFFFFFF;

      ~CutoutMaterials() { Shutdown(); }

      // strip_cs: Luma_RTAO.hlsl "alpha_strip_cs" compiled for the game device
      bool Init(ID3D11Device* device11, Interop& interop, ID3D11ComputeShader* strip_cs, std::string& error);
      bool IsInitialized() const { return atlas12 != nullptr; }
      // The caller makes sure the D3D12 queue is idle
      void Shutdown();

      // ---- D3D11 side (draw threads)
      // The material of a cutout draw (key 0: none, e.g. not initialized). "tile" receives a strip tile to render its mask
      // into now with RecordStrip (kNoTile: nothing to render).
      uint64_t AcquireMaterial(ID3D11ShaderResourceView* srv, uint32_t channel, uint32_t& tile);
      // Records the mask render into "tile" on "context" (no lock: only this context's state is touched)
      bool RecordStrip(ID3D11DeviceContext* context, ID3D11ShaderResourceView* srv, uint32_t channel, uint32_t tile);
      // RecordStrip failed, or its record will never execute: the material is rendered again at a later draw
      void AbandonStrip(uint64_t material, uint32_t tile);
      // The strip's D3D11 work has executed (its command list ran): copy it at the next D3D12 pass
      void OnStripExecuted(uint64_t material, uint32_t tile);
      void OnDestroyTexture(ID3D11Resource* texture);
      void OnPresent();
      void CountDraw(bool with_material);

      // ---- D3D12 side (recorder)
      // The record id for a cutout instance (kNoCutoutRecord if the table is full)
      uint32_t ResolveRecord(uint64_t material, const CutoutGeometry& geometry, const float scalebias[4], float alpha_ref);
      // Copies the executed strips into atlas slices and uploads the record table if it changed. "arena": the geometry arena.
      bool Record(ID3D12GraphicsCommandList4* list, const RecordInfo& info, Retirer& retirer, ID3D12Resource* arena, CutoutBindings& out, std::string& error);

      CutoutStats GetStats();

   private:
      enum class State : uint8_t { Pending, Ready, NoSlice };
      struct Material
      {
         ID3D11Resource* texture = nullptr; // Identity only
         State state = State::Pending;
         uint32_t tile = kNoTile; // Pending: the strip tile holding its mask
         uint32_t slice = 0xFFFF;
         uint32_t frame = 0; // Pending: since, Ready: rendered at
         bool refreshed = false;
      };
      struct RecordEntry
      {
         uint64_t key;
         uint64_t material;
         CutoutGpuRecord gpu; // uv_code_slice's slice resolved at upload
         uint32_t last_used;
      };
      struct ExecutedStrip
      {
         uint64_t material;
         uint32_t tile;
      };

      std::mutex mutex;
      ID3D11Device* device11 = nullptr;
      ID3D11ComputeShader* strip_cs = nullptr;
      ID3D11Buffer* params_cb = nullptr;
      ID3D11SamplerState* sampler = nullptr;
      ID3D11Texture2D* tile11 = nullptr; // Private, written by alpha_strip_cs
      ID3D11UnorderedAccessView* tile_uav = nullptr;
      ID3D12Resource* strip12 = nullptr; // Shared ring of tiles (D3D12 created render target)
      ID3D11Texture2D* strip11 = nullptr;
      ID3D12Resource* atlas12 = nullptr;
      bool atlas_written = false; // Left COPY_DEST until the first copy, NON_PIXEL_SHADER_RESOURCE after
      uint32_t strip_head = 0, strip_window = 0;
      uint32_t frame = 0; // OnPresent count
      uint32_t record_frame = 0; // Record count

      std::unordered_map<uint64_t, Material> materials;
      std::vector<ExecutedStrip> executed;
      std::vector<uint32_t> free_slices;
      uint32_t next_slice = 0;

      std::unordered_map<uint64_t, uint32_t> record_ids;
      std::vector<RecordEntry> records;
      std::vector<uint32_t> free_records;
      uint32_t last_sweep = 0;
      uint64_t version = 1;
      static constexpr uint32_t kSlots = 3;
      ID3D12Resource* upload[kSlots] = {};
      uint64_t upload_capacity[kSlots] = {};
      uint64_t upload_version[kSlots] = {};
      std::vector<CutoutGpuRecord> staging;

      CutoutStats stats;
   };
}
