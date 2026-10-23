#pragma once

// RTAO geometry capture: game draws (D3D11) -> BLAS inputs and per-instance world matrices (D3D12).
// See RTAO_NOTES.md for where Dishonored 2 keeps its geometry and object to world matrices.
//
// Design (and why):
//  - Meshes: the first time a (vertex buffer, index buffer) pair is drawn, both D3D11 buffers are copied whole, with
//    CopySubresourceRegion on the drawing context, into two large D3D11 buffers shared with D3D12 (the legacy
//    MISC_SHARED route, the only one that works for buffers, see Interop::CreateSharedBuffer). BLASes are then built
//    on the D3D12 side straight from those arenas. Whole buffers are copied because the CPU doesn't know which
//    vertex range an indexed draw touches; each buffer is copied once, however many meshes it holds.
//  - World matrices: computed on the GPU, by a tiny D3D11 compute dispatch (Luma_RTAO.hlsl capture_instances_cs)
//    recorded right next to each captured draw, with the draw's own bindings (its PerInstanceCB, instance buffer and
//    instance index SRVs). It resolves the matrix exactly like the game's vertex shader and writes it into a ring
//    buffer shared with D3D12. The CPU never stalls on the game's buffers (no D3D11 Map of GPU data).
//  - The D3D12 side copies the ring entries of the frame into a readback buffer, and the CPU reads them 3 frames
//    later (when that frame slot is reused, so no wait). That gives each instance a CPU side identity (mesh key +
//    quantized transform), which the TLAS retention (TlasBuilder, kRetainFrames) needs, as instance buffers are
//    rebuilt every frame and carry no stable id. Cost: instances appear 3 frames late, and a moving object leaves
//    copies along its path until they age out.
//  - Deferred contexts: records are kept per context, handed to the command list at FinishCommandList and only
//    count as executed when that list runs on the immediate context (ExecuteCommandList), like the RTAO split.
//  - Skinned meshes (characters): the game skins in the vertex shader (ro_matrices bone palette, see RTAO_NOTES.md),
//    so the vertices only exist for the draw. A second D3D11 compute dispatch at the draw (Luma_RTAO.hlsl
//    skin_vertices_cs) replays the skinning AND the instance transform from the arena copy of the vertex buffers and
//    the draw's own bone palette / PerInstanceCB / instance buffer, and writes WORLD space float3 positions into a
//    shared ring. World space means the TLAS instance is the identity, so nothing has to come back to the CPU: the
//    pose is the current frame's (no 3 frame matrix latency, no trail). The D3D12 side builds or refits one BLAS per
//    skinned draw from that ring in the same frame (DynamicBlasSet) and adds it as a current-frame-only instance.
//    Vertices: if the draw's vertex range (base vertex to the end of its buffers) is not larger than its index count,
//    that range is skinned and the BLAS uses the game's indices; otherwise each index gets its own skinned vertex
//    (a non indexed triangle soup), so the work is bounded by the draw's index count either way.
//    First person view models and matrices the capture can't resolve write NaN positions (DXR then treats those
//    triangles as inactive). Only single instance skinned draws are captured.
//  - Meshes with a DYNAMIC vertex buffer (movable objects: doors, windows, cabinets) take the same per frame path
//    without the skinning: the buffer is only valid for this frame, so instead of the one-time arena copy, the bytes
//    from the draw's vertex 0 to the buffer end are copied at every draw into a D3D11 only ring (kDynamicRingBytes),
//    which skin_vertices_cs reads with no bones. Their index buffers (DEFAULT) still go through the arena.
//  - Alpha tested draws (cutouts) with a replayable alpha test also get a material (texture exact cutouts, see
//    rtao_alpha.h): the mask is rendered at the first draw of a material, the mesh's TEXCOORD0 stream goes into the arena,
//    and capture_instances_cs writes the draw's uv scale/bias and alpha reference next to each instance's matrix.

#include "rtao_accel.h"
#include "rtao_alpha.h"
#include "rtao_interop.h"

#include <d3d11_1.h>
#include <atomic>
#include <memory>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace RTAO
{
   constexpr uint32_t kNoOffset = 0xFFFFFFFF;

   enum class MatrixSource : uint32_t
   {
      None = 0,
      DirectInstanceBuffer = 1, // StructuredBuffer ro_directinstancebuffer, stride 96, rows at 48/64/80
      IndexedInstanceBuffer = 2, // Buffer<uint> ro_instancebufferindex -> StructuredBuffer ro_instancebuffer, stride 48
      ConstantBuffer = 3, // PerInstanceCB cb_modelmatrixx/y/z
      MvpConstantBuffer = 4, // PerInstanceCB cb_mvpmatrixx/y/z/w only: world = MVP x inverse(PerViewCB cb_viewprojectionmatrix)
      Count
   };

   // What a vertex shader needs for capture, from its reflection
   struct VertexShaderInfo
   {
      MatrixSource source = MatrixSource::None;
      const char* skip_reason = "not reflected"; // Static string, null when capturable
      uint32_t instance_cb_slot = kNoOffset; // PerInstanceCB
      uint32_t instance_offset_cb = kNoOffset; // cb_instancebufferoffset, bytes
      uint32_t row_cb[3] = { kNoOffset, kNoOffset, kNoOffset }; // cb_modelmatrixx/y/z, bytes
      uint32_t depth_hack_cb = kNoOffset; // cb_modeldepthhack, bytes
      uint32_t instance_buffer_slot = kNoOffset;
      uint32_t instance_buffer_stride = 0;
      uint32_t instance_index_slot = kNoOffset;
      uint32_t record_row0 = 0; // Byte offset of the x row in an instance record
      uint32_t mvp_cb[4] = { kNoOffset, kNoOffset, kNoOffset, kNoOffset }; // cb_mvpmatrixx/y/z/w, bytes (clip.x = dot(pos, x) ...)
      uint32_t per_view_cb_slot = kNoOffset; // PerViewCB
      uint32_t view_proj_cb = kNoOffset; // cb_viewprojectionmatrix in PerViewCB, bytes
      // Vertex animated vegetation (ro_windbuffer, ro_splineinfos or cb_vegetanim_*, see RTAO_NOTES.md "Wind"). Its matrix
      // comes from one of the families above like any mesh; the wind displacement is ignored (captured in rest pose).
      bool wind = false;
      // Skinning (VERTEX_SKININDICES inputs + Buffer<float4> ro_matrices): the skinned position is in model space, then
      // goes through the matrix source above like any other vertex
      bool skinned = false;
      uint32_t skin_sets = 0; // 1: 4 bones (SKININDICES0/WEIGHTS0), 2: 8 bones (...1 too)
      uint32_t bones_slot = kNoOffset; // ro_matrices SRV slot
      uint32_t mtx_offset_cb = kNoOffset; // cb_mtxoffset in PerInstanceCB, bytes (.x = first bone row / 3)
   };
   // Uses D3DReflect (D3DCompiler_47.dll, loaded on first use). "input_signature", if given, receives the input
   // semantics (diagnostics).
   bool ReflectVertexShader(const void* code, size_t code_size, VertexShaderInfo& out, std::string* input_signature = nullptr);

   // Alpha tested ("cutout") pixel shaders: leaves, grates, fences. See RTAO_NOTES.md "Alpha tested geometry".
   struct PixelShaderInfo
   {
      bool discard = false; // The bytecode has a discard instruction
      bool alpha_tested = false; // A discard depends on a read of a diffuse/opacity/alpha texture (not dithering, not dissolve)
      // Texture exact cutouts (alpha_tested only, see rtao_alpha.h). The game's alpha test is
      // "discard if tex[alpha_slot].a at (TEXCOORD0 * scalebias.xy + scalebias.zw) < reference".
      bool texture_exact = false; // The alpha read's coordinate is one of the two forms above: the mask can be replayed
      uint32_t alpha_slot = kNoOffset; // SRV slot of the texture whose read the discard depends on
      uint32_t alpha_channel = 3; // Its channel the test reads (3: alpha)
      uint32_t material_cb_slot = kNoOffset; // The cbuffer holding the two below (PerInstanceCB)
      uint32_t alpha_ref_cb = kNoOffset; // cb_defaultalphatest / cb_hairalphatest, bytes. kNoOffset: "alpha_ref" below
      float alpha_ref = 0.5f; // Hardcoded reference ("add r, alpha, l(-0.5)" in the vegetation shaders), 0.5 if not found
      bool alpha_ref_exact = false; // The reference is a cbuffer value or a recognized hardcoded one (diagnostics)
      uint32_t uv_scalebias_cb = kNoOffset; // cb_layer0_scalebias, bytes. kNoOffset: TEXCOORD0 as is
   };
   // True if the DXBC (SHEX/SHDR chunk) contains a discard instruction. No D3DReflect needed.
   bool DxbcHasDiscard(const void* code, size_t code_size);
   // Uses D3DReflect (texture names) and a data flow pass over the bytecode, only for shaders that discard
   bool ReflectPixelShader(const void* code, size_t code_size, PixelShaderInfo& out);

   struct InputLayoutInfo
   {
      bool has_position = false;
      DXGI_FORMAT position_format = DXGI_FORMAT_UNKNOWN;
      uint32_t position_slot = 0;
      uint32_t position_offset = 0;
      bool position_per_instance = false;
      struct Element
      {
         bool present = false;
         DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
         uint32_t slot = 0;
         uint32_t offset = 0;
         bool per_instance = false;
      };
      Element skin_indices[2]; // VERTEX_SKININDICES0/1
      Element skin_weights[2]; // VERTEX_SKINWEIGHTS0/1
      Element texcoord0; // VERTEX_TEXCOORD0 (texture exact cutouts)
   };
   // Resolves D3D11_APPEND_ALIGNED_ELEMENT offsets
   InputLayoutInfo ParseInputLayout(const D3D11_INPUT_ELEMENT_DESC* elements, uint32_t count);
   // The DXR vertex format for a game position format, or DXGI_FORMAT_UNKNOWN if DXR can't read it
   DXGI_FORMAT DxrVertexFormat(DXGI_FORMAT position_format);

   // Must match CaptureParams in Shaders/Dishonored 2/Luma_RTAO.hlsl
   struct CaptureParams
   {
      uint32_t source;
      uint32_t instance_count;
      uint32_t dst_slot;
      uint32_t slot_mask;
      uint32_t instance_offset_cb;
      uint32_t row_cb[3];
      uint32_t depth_hack_cb;
      uint32_t record_row0;
      uint32_t alpha_ref_cb; // Cutouts: in the material cbuffer (b3), kNoOffset: alpha_ref_const
      uint32_t uv_scalebias_cb; // Cutouts: in the material cbuffer (b3), kNoOffset: none
      uint32_t mvp_cb[4];
      uint32_t view_proj_cb;
      float alpha_ref_const;
      uint32_t pad2[2];
      // skin_vertices_cs only
      uint32_t skin_vertex_count; // Threads: vertices written
      uint32_t skin_dst_vertex; // First vertex in the skinned ring
      uint32_t skin_soup; // 0: vertex i of the draw's range, 1: the vertex of index i (triangle soup)
      uint32_t skin_sets; // 1 or 2 (4 or 8 bones)
      uint32_t mtx_offset_cb; // cb_mtxoffset in the game's PerInstanceCB, bytes
      uint32_t skin_index_start; // Arena SRV byte address of the draw's first index (soup)
      uint32_t skin_index32; // 1: 32 bit indices
      uint32_t pad3;
      // Streams: 0 position, 1 indices0, 2 weights0, 3 indices1, 4 weights1. Vertex v of stream s is at arena SRV byte
      // address stream_base[s] + v * stream_stride[s] (base: buffer copy + IA offset + base vertex + element offset).
      uint32_t stream_base[8];
      uint32_t stream_stride[8];
      uint32_t stream_format[8]; // VertexFormatCode()
   };
   static_assert(sizeof(CaptureParams) == 208);

   // True if a 3x4 row major object to world matrix has perpendicular axes of sane lengths (rotation x scale + translation)
   bool IsRigidTransform(const float rows[12]);

   // Vertex element decoding for skin_vertices_cs: kind | component count << 4 | bits per component << 8, 0 if the
   // shader can't decode it. Must match DecodeElement() in Luma_RTAO.hlsl.
   uint32_t VertexFormatCode(DXGI_FORMAT format);
   constexpr uint32_t kVertexFormatDynamicBuffer = 1u << 16; // Stream flag: read from the dynamic vertex ring, not the arena

   // Everything the caller reads from the context at the draw
   struct DrawInfo
   {
      const VertexShaderInfo* vs = nullptr;
      ID3D11Buffer* vertex_buffer = nullptr;
      uint32_t vertex_buffer_offset = 0; // IASetVertexBuffers offset, bytes
      uint32_t vertex_stride = 0;
      DXGI_FORMAT position_format = DXGI_FORMAT_UNKNOWN;
      uint32_t position_offset = 0; // Inside a vertex
      ID3D11Buffer* index_buffer = nullptr;
      DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
      uint32_t index_buffer_offset = 0;
      uint32_t index_count = 0;
      uint32_t first_index = 0;
      int32_t base_vertex = 0;
      uint32_t instance_count = 1;
      ID3D11Buffer* instance_cb = nullptr; // The bound PerInstanceCB
      uint32_t instance_cb_first_constant = 0, instance_cb_num_constants = 0; // VSGetConstantBuffers1 (0, 0: whole buffer)
      ID3D11ShaderResourceView* instance_buffer = nullptr;
      ID3D11ShaderResourceView* instance_index = nullptr;
      ID3D11Buffer* per_view_cb = nullptr; // The bound PerViewCB (MvpConstantBuffer only)
      uint32_t per_view_cb_first_constant = 0, per_view_cb_num_constants = 0;
      bool cutout = false; // Alpha tested (the bound pixel shader): its instances go into the TLAS as cutouts (kInstanceMaskCutout)
      // Skinned shaders only (vs->skinned)
      struct Stream
      {
         ID3D11Buffer* buffer = nullptr; // IAGetVertexBuffers of the element's slot
         uint32_t stride = 0;
         uint32_t offset = 0; // IA offset, bytes
         uint32_t element_offset = 0; // Inside a vertex
         DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
      };
      Stream skin_indices[2];
      Stream skin_weights[2];
      ID3D11ShaderResourceView* bones = nullptr; // The bound ro_matrices
      // Cutouts, texture exact (ps->texture_exact): the pixel shader's alpha texture and material cbuffer, the TEXCOORD0 stream
      const PixelShaderInfo* ps = nullptr;
      ID3D11ShaderResourceView* alpha_srv = nullptr; // PS SRV at ps->alpha_slot
      ID3D11Buffer* material_cb = nullptr; // PS cbuffer at ps->material_cb_slot
      uint32_t material_cb_first_constant = 0, material_cb_num_constants = 0;
      Stream uv;
      // Changes whenever the game rewrote (Map DISCARD, on the recording thread) a buffer behind the draw's per instance
      // bindings, so a draw only counts as a duplicate of an earlier one if its data can't have changed in between
      uint64_t binding_version = 0;
      // A shadow map draw (depth only, into a shadow map array), not the main view: the casters the sun needs that the
      // camera never draws (behind or beside it). These are drawn after the AO pass, so they are kept anyway (and reach
      // the TLAS one pass later). Skinned draws are skipped (cost: once per cascade), and so are per frame meshes whose
      // matrix would come from the MVP (no CPU check possible). MVP derived static instances must pass a rigid
      // transform check at readback: world = MVP x inverse(the bound PerViewCB view projection) is only right if the
      // game bound the light's view in PerViewCB.
      bool shadow_view = false;
   };

   struct CaptureStats
   {
      // Skinned draws
      uint64_t skinned_captured = 0; // Dispatched
      uint64_t skinned_instanced = 0; // Skipped: more than one instance
      uint64_t skinned_waiting = 0; // Skipped: a buffer copy recorded on another context hasn't executed yet
      uint64_t skinned_overflow = 0; // Skipped: the frame's skinned vertex budget is used up
      // Meshes with a CPU written (DYNAMIC) vertex buffer: movable objects, copied and transformed every draw
      uint64_t dynamic_captured = 0, dynamic_instanced = 0, dynamic_waiting = 0;
      uint64_t dynamic_too_large = 0; // Skipped: more than kDynamicMaxCopyBytes from the draw's first vertex to the buffer end
      // Per frame meshes (skinned + dynamic)
      uint64_t frame_overflow = 0; // Skipped: the frame's vertex budget is used up
      uint64_t frame_soup = 0, frame_indexed = 0; // Captured as a triangle soup (one vertex per index) / with the game's indices
      uint64_t frame_draws = 0, frame_vertices = 0; // Built into the last frame's TLAS
      uint64_t frame_blas_builds = 0, frame_blas_refits = 0, frame_blas_failed = 0;
      uint64_t frame_blas_count = 0, frame_blas_bytes = 0;
      uint64_t draws_seen = 0, draws_captured = 0, instances_captured = 0;
      uint64_t skipped[8] = {}; // By CaptureSkip
      uint64_t meshes = 0, meshes_ready = 0, buffers_copied = 0, arena_vertex_bytes = 0, arena_index_bytes = 0;
      uint64_t blas_built = 0, blas_failed = 0, blas_evicted = 0;
      uint64_t frame_instances = 0; // Read back last frame
      uint64_t view_model_instances = 0, late_records = 0, readback_overflow = 0, after_ao_records = 0;
      uint64_t tlas_instances = 0;
      // Foliage: captured draws/instances of wind (vertex animated) and alpha tested (cutout) draws, cutouts read back last frame
      uint64_t wind_draws = 0, wind_instances = 0, cutout_draws = 0, cutout_instances = 0, frame_cutout_instances = 0;
      // Skipped as duplicates of a draw captured earlier in the frame (same mesh and per instance bindings, e.g. the depth
      // prepass and the G-buffer pass drawing the same objects): static draws, per frame (skinned/dynamic) meshes
      uint64_t dedup_draws = 0, dedup_frame_meshes = 0;
      // Shadow map draws (DrawInfo::shadow_view): captured draws / instances / per frame meshes, skipped (skinned, MVP
      // per frame meshes), instances rejected at readback (MVP derived, not a rigid transform)
      uint64_t shadow_draws = 0, shadow_instances = 0, shadow_frame_meshes = 0, shadow_skipped = 0, shadow_rejected = 0;
   };
   enum CaptureSkip
   {
      kSkipShader = 0, kSkipFormat, kSkipBuffer, kSkipArenaFull, kSkipRingFull, kSkipBinding, kSkipOther, kSkipCount
   };

   class GeometryCapture
   {
   public:
      ~GeometryCapture() { Shutdown(); }

      // capture_cs: Luma_RTAO.hlsl "capture_instances_cs" compiled for the game device. skin_cs: "skin_vertices_cs",
      // optional (without it, or if its buffers can't be made, skinned draws are skipped as kSkipShader).
      bool Init(ID3D11Device* device11, Interop& interop, ID3D11ComputeShader* capture_cs, ID3D11ComputeShader* skin_cs, std::string& error);
      bool IsSkinningAvailable() const { return skin_cs != nullptr; }
      std::string skinning_error; // Why skinning isn't available (set by Init)
      // Skinned draws / draws with a DYNAMIC vertex buffer are captured only while these are set
      std::atomic<bool> skinning_enabled = true;
      std::atomic<bool> dynamic_enabled = true;
      // Draws with the same identity as one captured earlier in the frame are skipped (see CaptureStats::dedup_draws)
      std::atomic<bool> dedup_enabled = true;
      bool IsInitialized() const { return ring11 != nullptr; }
      void Shutdown();
      // Texture exact cutouts (after Init). strip_cs: Luma_RTAO.hlsl "alpha_strip_cs". Without it cutouts stay stochastic.
      bool InitCutouts(ID3D11Device* device11, Interop& interop, ID3D11ComputeShader* strip_cs, std::string& error);
      bool IsCutoutsAvailable() { return cutouts.IsInitialized(); }
      std::atomic<bool> cutouts_exact = true; // Cutout draws get materials only while set (UI)
      // The trace's cutout resources for the frame (valid after Record, until the next one)
      const CutoutBindings& GetCutoutBindings() const { return cutout_bindings; }
      CutoutStats GetCutoutStats() { return cutouts.GetStats(); }

      // At a game draw, on the context recording it (any thread). Records the copies and the matrix dispatch.
      bool CaptureDraw(ID3D11DeviceContext* context, const DrawInfo& draw);

      // Deferred context bookkeeping (see the header comment). Immediate context draws are executed by definition.
      void OnFinishCommandList(ID3D11DeviceContext* deferred_context, ID3D11CommandList* command_list);
      void OnExecuteCommandList(ID3D11CommandList* command_list);
      // For the RTAO split: the records recorded so far on "deferred_context" end up in the prefix, executed before
      // the D3D12 pass. Take them here and Commit them when the prefix executes.
      struct PendingRecords;
      std::shared_ptr<PendingRecords> TakeContextRecords(ID3D11DeviceContext* deferred_context);
      void Commit(const std::shared_ptr<PendingRecords>& records);
      // Call right after the frame's AO pass ran: geometry drawn later in the frame (translucent/reflective surfaces
      // that write depth after the AO's depth was taken) is left out, until OnPresent.
      void MarkAOPassDone();
      void OnPresent();
      void OnDestroyBuffer(ID3D11Buffer* buffer);
      void OnDestroyTexture(ID3D11Resource* texture) { cutouts.OnDestroyTexture(texture); }

      // D3D12 side, inside an Interop recorder. Consumes the executed records: reads back 3 frames old matrices into
      // "tlas", builds the BLASes whose data has been copied (up to blas_budget), queues this frame's readback.
      bool Record(ID3D12GraphicsCommandList4* list, const RecordInfo& info, BlasCache& blases, TlasBuilder& tlas, Retirer& retirer, uint32_t blas_budget, std::string& error);

      CaptureStats GetStats();

      static constexpr uint32_t kRingSlots = 1u << 18; // 64 bytes each: 16 MB
      static constexpr uint32_t kReadbackSlots = 1u << 17; // Instances per frame read back: 8 MB per frame slot (main view + shadow maps)
      static constexpr uint32_t kFrameSlots = 3;
      // Skinned vertices (world space float3): ring capacity, and how many may be handed out between two D3D12 passes.
      // The budget stays well under the capacity so a draw recorded for a later frame can't wrap over vertices a pass
      // has not consumed yet.
      static constexpr uint32_t kSkinRingVertices = 1u << 22; // 48 MB
      static constexpr uint32_t kSkinFrameBudget = 1536u * 1024u;
      // Raw SRV views of the arena for the skinning shader: a view can't span the whole arena (2^27 elements max), so
      // views of arena_view_span bytes (the largest the driver accepts, from kArenaViewMaxSpan down) start every
      // span / 2 bytes, and a draw uses one that holds all its data.
      static constexpr uint64_t kArenaViewMaxSpan = 512ull << 20;
      // DYNAMIC vertex buffers: copied at each draw into a D3D11 only ring (read by skin_vertices_cs), same budget rule
      static constexpr uint64_t kDynamicRingBytes = 48ull << 20;
      static constexpr uint64_t kDynamicFrameBudget = 16ull << 20;
      static constexpr uint64_t kDynamicMaxCopyBytes = 4ull << 20;

   private:
      struct BufferEntry
      {
         uint64_t arena_offset = 0;
         uint32_t size = 0;
         bool executed = false;
         ID3D11DeviceContext* recorded_on = nullptr; // The context the copy was recorded on
      };
      struct MeshEntry
      {
         ID3D11Buffer* vertex_buffer;
         ID3D11Buffer* index_buffer;
         uint64_t vertex_start; // In the vertex buffer: offset of vertex 0 + position offset
         uint32_t vertex_count;
         uint32_t vertex_stride;
         DXGI_FORMAT vertex_format;
         uint64_t index_start; // In the index buffer
         uint32_t index_count;
         DXGI_FORMAT index_format;
         bool built = false;
         bool failed = false;
      };
      struct Rec
      {
         enum Type : uint8_t { Instances, BufferCopy, FrameMesh, AlphaStrip } type;
         uint8_t mask; // Instances: TLAS instance mask (kInstanceMaskOpaque / kInstanceMaskCutout)
         uint64_t mesh_key; // Instances, FrameMesh. AlphaStrip: the material.
         uint32_t first_slot; // Instances: capture ring slot. FrameMesh: first vertex in the skinned ring. AlphaStrip: the strip tile.
         uint32_t count; // Instances: instances. FrameMesh: vertices.
         ID3D11Buffer* buffer; // BufferCopy
         // FrameMesh, indexed (not soup) only: the game's indices in the arena
         uint64_t index_arena_offset;
         uint32_t index_count;
         DXGI_FORMAT index_format; // DXGI_FORMAT_UNKNOWN: soup
         // Instances of a texture exact cutout draw: its material (0: none) and where its triangles/uvs are in the arena
         uint64_t material;
         CutoutGeometry geometry;
         bool shadow_view = false; // Instances, FrameMesh: from a shadow map draw (kept after the AO pass)
         bool check_rigid = false; // Instances: MVP derived in a shadow map draw (see DrawInfo::shadow_view)
      };

      bool EnsureBufferCopy(ID3D11DeviceContext* context, ID3D11Buffer* buffer, bool vertex, std::vector<Rec>& records);
      // Skinned or dynamic vertex buffer draws: world space vertices for this frame (see the header comment)
      bool CaptureFrameDraw(ID3D11DeviceContext* context, const DrawInfo& draw, uint64_t key, uint64_t identity, uint64_t index_start, uint32_t index_size, bool dynamic_vertices);
      void CommitLocked(std::vector<Rec>& records);
      void QueueLocked(ID3D11DeviceContext* context, std::vector<Rec>& local);

      std::mutex mutex;
      ID3D11Device* device11 = nullptr;
      ID3D11ComputeShader* capture_cs = nullptr;
      ID3D11Buffer* params_cb = nullptr;
      ID3D11Buffer* ring11 = nullptr;
      ID3D11UnorderedAccessView* ring_uav = nullptr;
      ID3D12Resource* ring12 = nullptr;
      ID3D11Buffer* vertex_arena11 = nullptr;
      ID3D12Resource* vertex_arena12 = nullptr;
      uint64_t vertex_arena_size = 0, vertex_arena_used = 0;
      ID3D11Buffer* index_arena11 = nullptr;
      ID3D12Resource* index_arena12 = nullptr;
      uint64_t index_arena_size = 0, index_arena_used = 0;
      ID3D12Resource* readback[kFrameSlots] = {};
      // Skinning
      ID3D11ComputeShader* skin_cs = nullptr;
      ID3D11Buffer* skin_ring11 = nullptr;
      ID3D11UnorderedAccessView* skin_ring_uav = nullptr;
      ID3D12Resource* skin_ring12 = nullptr;
      std::vector<ID3D11ShaderResourceView*> arena_views; // View i starts at i * arena_view_step
      uint64_t arena_view_span = 0, arena_view_step = 0;
      uint32_t skin_head = 0; // Next free vertex in the skinned ring
      uint32_t skin_window = 0; // Vertices handed out since the last D3D12 pass
      ID3D11Buffer* dynamic_ring = nullptr;
      ID3D11ShaderResourceView* dynamic_ring_srv = nullptr;
      uint64_t dynamic_head = 0, dynamic_window = 0;
      DynamicBlasSet frame_blases;

      uint64_t ring_head = 0;
      std::unordered_map<ID3D11Buffer*, BufferEntry> buffers;
      std::unordered_map<uint64_t, MeshEntry> meshes;
      std::vector<uint64_t> evictions;
      std::unordered_map<ID3D11DeviceContext*, std::vector<Rec>> context_records;
      std::unordered_set<uint64_t> frame_identities; // Draws captured this frame (dedup), cleared at OnPresent
      struct ListRecords
      {
         std::vector<Rec> records;
         uint32_t frame;
      };
      std::unordered_map<ID3D11CommandList*, ListRecords> list_records;
      std::vector<Rec> executed; // Waiting for the next D3D12 pass
      uint32_t frame = 0;
      uint32_t last_record_frame = 0;
      bool after_ao_pass = false;
      // Per frame slot: what was copied into that readback buffer, in order
      struct ReadbackRecord
      {
         uint64_t mesh_key;
         uint32_t count;
         uint8_t mask;
         uint64_t material; // Cutouts: see Rec
         CutoutGeometry geometry;
         bool shadow_view;
         bool check_rigid;
      };
      std::vector<ReadbackRecord> readback_records[kFrameSlots];
      CaptureStats stats;
      CutoutMaterials cutouts; // Own lock, taken after "mutex" when nested
      CutoutBindings cutout_bindings;
   };

   struct GeometryCapture::PendingRecords
   {
      std::vector<GeometryCapture::Rec> records;
   };
}
