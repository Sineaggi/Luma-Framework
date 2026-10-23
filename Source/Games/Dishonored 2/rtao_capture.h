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

#include "rtao_accel.h"
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
   };
   // Uses D3DReflect (D3DCompiler_47.dll, loaded on first use)
   bool ReflectVertexShader(const void* code, size_t code_size, VertexShaderInfo& out);

   struct InputLayoutInfo
   {
      bool has_position = false;
      DXGI_FORMAT position_format = DXGI_FORMAT_UNKNOWN;
      uint32_t position_slot = 0;
      uint32_t position_offset = 0;
      bool position_per_instance = false;
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
      uint32_t pad[2];
   };
   static_assert(sizeof(CaptureParams) == 48);

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
   };

   struct CaptureStats
   {
      uint64_t draws_seen = 0, draws_captured = 0, instances_captured = 0;
      uint64_t skipped[8] = {}; // By CaptureSkip
      uint64_t meshes = 0, meshes_ready = 0, buffers_copied = 0, arena_vertex_bytes = 0, arena_index_bytes = 0;
      uint64_t blas_built = 0, blas_failed = 0, blas_evicted = 0;
      uint64_t frame_instances = 0; // Read back last frame
      uint64_t view_model_instances = 0, late_records = 0, readback_overflow = 0;
      uint64_t tlas_instances = 0;
   };
   enum CaptureSkip
   {
      kSkipShader = 0, kSkipFormat, kSkipBuffer, kSkipArenaFull, kSkipRingFull, kSkipBinding, kSkipOther, kSkipCount
   };

   class GeometryCapture
   {
   public:
      ~GeometryCapture() { Shutdown(); }

      // capture_cs: Luma_RTAO.hlsl "capture_instances_cs" compiled for the game device
      bool Init(ID3D11Device* device11, Interop& interop, ID3D11ComputeShader* capture_cs, std::string& error);
      bool IsInitialized() const { return ring11 != nullptr; }
      void Shutdown();

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
      void OnPresent();
      void OnDestroyBuffer(ID3D11Buffer* buffer);

      // D3D12 side, inside an Interop recorder. Consumes the executed records: reads back 3 frames old matrices into
      // "tlas", builds the BLASes whose data has been copied (up to blas_budget), queues this frame's readback.
      bool Record(ID3D12GraphicsCommandList4* list, const RecordInfo& info, BlasCache& blases, TlasBuilder& tlas, Retirer& retirer, uint32_t blas_budget, std::string& error);

      CaptureStats GetStats();

      static constexpr uint32_t kRingSlots = 1u << 18; // 64 bytes each: 16 MB
      static constexpr uint32_t kReadbackSlots = 1u << 16; // Instances per frame read back: 4 MB per frame slot
      static constexpr uint32_t kFrameSlots = 3;

   private:
      struct BufferEntry
      {
         uint64_t arena_offset = 0;
         uint32_t size = 0;
         bool executed = false;
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
         enum Type : uint8_t { Instances, BufferCopy } type;
         uint64_t mesh_key; // Instances
         uint32_t first_slot;
         uint32_t count;
         ID3D11Buffer* buffer; // BufferCopy
      };

      bool EnsureBufferCopy(ID3D11DeviceContext* context, ID3D11Buffer* buffer, bool vertex, std::vector<Rec>& records);
      void CommitLocked(std::vector<Rec>& records);

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

      uint64_t ring_head = 0;
      std::unordered_map<ID3D11Buffer*, BufferEntry> buffers;
      std::unordered_map<uint64_t, MeshEntry> meshes;
      std::vector<uint64_t> evictions;
      std::unordered_map<ID3D11DeviceContext*, std::vector<Rec>> context_records;
      struct ListRecords
      {
         std::vector<Rec> records;
         uint32_t frame;
      };
      std::unordered_map<ID3D11CommandList*, ListRecords> list_records;
      std::vector<Rec> executed; // Waiting for the next D3D12 pass
      uint32_t frame = 0;
      uint32_t last_record_frame = 0;
      // Per frame slot: what was copied into that readback buffer (mesh key, count), in order
      std::vector<std::pair<uint64_t, uint32_t>> readback_records[kFrameSlots];
      CaptureStats stats;
   };

   struct GeometryCapture::PendingRecords
   {
      std::vector<GeometryCapture::Rec> records;
   };
}
