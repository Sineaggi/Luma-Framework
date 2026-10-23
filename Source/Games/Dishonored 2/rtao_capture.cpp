#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rtao_capture.h"

#include <d3d11shader.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace RTAO
{
   namespace
   {
      std::string HResultString(const char* what, HRESULT hr)
      {
         char buffer[160];
         snprintf(buffer, sizeof(buffer), "%s failed (0x%08lX)", what, (unsigned long)hr);
         return buffer;
      }

      template <typename T>
      void SafeRelease(T*& p)
      {
         if (p)
         {
            p->Release();
            p = nullptr;
         }
      }

      using PFN_D3DReflect = HRESULT(WINAPI*)(LPCVOID, SIZE_T, REFIID, void**);
      // IID_ID3D11ShaderReflection (d3d11shader.h declares it with DEFINE_GUID, which needs dxguid.lib)
      const GUID kIID_ID3D11ShaderReflection = { 0x8d536ca1, 0x0cca, 0x4956, { 0xa8, 0x37, 0x78, 0x69, 0x63, 0x75, 0x55, 0x84 } };

      PFN_D3DReflect GetD3DReflect()
      {
         static PFN_D3DReflect reflect = []() -> PFN_D3DReflect {
            HMODULE module = LoadLibraryW(L"D3DCompiler_47.dll");
            return module ? reinterpret_cast<PFN_D3DReflect>(GetProcAddress(module, "D3DReflect")) : nullptr;
         }();
         return reflect;
      }

      uint32_t FormatSize(DXGI_FORMAT format)
      {
         switch (format)
         {
         case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT: case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return 16;
         case DXGI_FORMAT_R32G32B32_FLOAT: case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT: case DXGI_FORMAT_R32G32B32_TYPELESS:
            return 12;
         case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
         case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
            return 8;
         case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_SNORM: case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT: case DXGI_FORMAT_B8G8R8A8_UNORM:
         case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_UINT: case DXGI_FORMAT_R11G11B10_FLOAT:
         case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SINT:
         case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
            return 4;
         case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
         case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT:
            return 2;
         case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
            return 1;
         default:
            return 0;
         }
      }

      uint64_t HashCombine(uint64_t h, uint64_t v)
      {
         h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
         return h * 0xFF51AFD7ED558CCDull;
      }

      void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
      {
         D3D12_RESOURCE_BARRIER barrier = {};
         barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
         barrier.Transition.pResource = resource;
         barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
         barrier.Transition.StateBefore = before;
         barrier.Transition.StateAfter = after;
         list->ResourceBarrier(1, &barrier);
      }
   }

   // ---- Reflection

   bool ReflectVertexShader(const void* code, size_t code_size, VertexShaderInfo& out, std::string* input_signature)
   {
      out = {};
      auto reflect = GetD3DReflect();
      if (!reflect)
      {
         out.skip_reason = "no D3DReflect";
         return false;
      }
      ID3D11ShaderReflection* reflection = nullptr;
      if (FAILED(reflect(code, code_size, kIID_ID3D11ShaderReflection, reinterpret_cast<void**>(&reflection))) || !reflection)
      {
         out.skip_reason = "reflection failed";
         return false;
      }
      D3D11_SHADER_DESC desc = {};
      reflection->GetDesc(&desc);

      bool has_position = false;
      for (UINT i = 0; i < desc.InputParameters; i++)
      {
         D3D11_SIGNATURE_PARAMETER_DESC parameter = {};
         reflection->GetInputParameterDesc(i, &parameter);
         if (!parameter.SemanticName)
            continue;
         if (input_signature)
         {
            char item[96];
            snprintf(item, sizeof(item), "%s%s%u", input_signature->empty() ? "" : " ", parameter.SemanticName, parameter.SemanticIndex);
            *input_signature += item;
         }
         if (_stricmp(parameter.SemanticName, "VERTEX_SKININDICES") == 0)
         {
            out.skip_reason = "skinned";
            reflection->Release();
            return false;
         }
         if (_stricmp(parameter.SemanticName, "VERTEX_POSITION") == 0 && parameter.SemanticIndex == 0)
            has_position = true;
      }
      if (!has_position)
      {
         out.skip_reason = "no VERTEX_POSITION";
         reflection->Release();
         return false;
      }

      uint32_t direct_slot = kNoOffset, direct_stride = 0, indexed_slot = kNoOffset, indexed_stride = 0, index_slot = kNoOffset;
      for (UINT i = 0; i < desc.BoundResources; i++)
      {
         D3D11_SHADER_INPUT_BIND_DESC bind = {};
         reflection->GetResourceBindingDesc(i, &bind);
         if (!bind.Name)
            continue;
         if (strstr(bind.Name, "windbuffer") || strstr(bind.Name, "splineinfos"))
            out.skip_reason = "wind";
         else if (strstr(bind.Name, "particle"))
            out.skip_reason = "particles";
         else if (strstr(bind.Name, "ocean") || strstr(bind.Name, "fftwaves"))
            out.skip_reason = "water";
         else if (strcmp(bind.Name, "ro_matrices") == 0)
            out.skip_reason = "skinned";
         else if (strcmp(bind.Name, "ro_directinstancebuffer") == 0)
         {
            direct_slot = bind.BindPoint;
            direct_stride = bind.NumSamples; // Structure stride for structured buffers
         }
         else if (strcmp(bind.Name, "ro_instancebuffer") == 0)
         {
            indexed_slot = bind.BindPoint;
            indexed_stride = bind.NumSamples;
         }
         else if (strcmp(bind.Name, "ro_instancebufferindex") == 0)
            index_slot = bind.BindPoint;
         else if (strcmp(bind.Name, "PerInstanceCB") == 0 && bind.Type == D3D_SIT_CBUFFER)
            out.instance_cb_slot = bind.BindPoint;
         else if (strcmp(bind.Name, "PerViewCB") == 0 && bind.Type == D3D_SIT_CBUFFER)
            out.per_view_cb_slot = bind.BindPoint;
      }
      if (out.skip_reason && strcmp(out.skip_reason, "not reflected") != 0)
      {
         reflection->Release();
         return false;
      }

      bool rows_used = false, mvp_used = false;
      if (out.per_view_cb_slot != kNoOffset)
      {
         ID3D11ShaderReflectionConstantBuffer* cb = reflection->GetConstantBufferByName("PerViewCB");
         D3D11_SHADER_BUFFER_DESC cb_desc = {};
         if (cb && SUCCEEDED(cb->GetDesc(&cb_desc)))
         {
            for (UINT v = 0; v < cb_desc.Variables; v++)
            {
               D3D11_SHADER_VARIABLE_DESC var = {};
               if (SUCCEEDED(cb->GetVariableByIndex(v)->GetDesc(&var)) && var.Name && strcmp(var.Name, "cb_viewprojectionmatrix") == 0)
                  out.view_proj_cb = var.StartOffset;
            }
         }
      }
      if (out.instance_cb_slot != kNoOffset)
      {
         ID3D11ShaderReflectionConstantBuffer* cb = reflection->GetConstantBufferByName("PerInstanceCB");
         D3D11_SHADER_BUFFER_DESC cb_desc = {};
         if (cb && SUCCEEDED(cb->GetDesc(&cb_desc)))
         {
            uint32_t used_rows = 0, used_mvp = 0;
            for (UINT v = 0; v < cb_desc.Variables; v++)
            {
               D3D11_SHADER_VARIABLE_DESC var = {};
               if (FAILED(cb->GetVariableByIndex(v)->GetDesc(&var)) || !var.Name)
                  continue;
               const bool used = (var.uFlags & D3D_SVF_USED) != 0;
               if (strcmp(var.Name, "cb_instancebufferoffset") == 0)
                  out.instance_offset_cb = var.StartOffset;
               else if (strcmp(var.Name, "cb_modeldepthhack") == 0)
                  out.depth_hack_cb = var.StartOffset;
               else if (strcmp(var.Name, "cb_modelmatrixx") == 0)
               {
                  out.row_cb[0] = var.StartOffset;
                  used_rows += used;
               }
               else if (strcmp(var.Name, "cb_modelmatrixy") == 0)
               {
                  out.row_cb[1] = var.StartOffset;
                  used_rows += used;
               }
               else if (strcmp(var.Name, "cb_modelmatrixz") == 0)
               {
                  out.row_cb[2] = var.StartOffset;
                  used_rows += used;
               }
               else if (strncmp(var.Name, "cb_mvpmatrix", 12) == 0 && var.Name[12] != 0 && var.Name[13] == 0)
               {
                  const char c = var.Name[12];
                  const int index = c == 'x' ? 0 : (c == 'y' ? 1 : (c == 'z' ? 2 : (c == 'w' ? 3 : -1)));
                  if (index >= 0)
                  {
                     out.mvp_cb[index] = var.StartOffset;
                     used_mvp += used;
                  }
               }
            }
            rows_used = used_rows == 3;
            mvp_used = used_mvp == 4;
         }
      }
      reflection->Release();

      if (direct_slot != kNoOffset)
      {
         if (direct_stride != 96)
         {
            out.skip_reason = "unexpected direct instance stride";
            return false;
         }
         out.source = MatrixSource::DirectInstanceBuffer;
         out.instance_buffer_slot = direct_slot;
         out.instance_buffer_stride = direct_stride;
         out.record_row0 = 48;
      }
      else if (indexed_slot != kNoOffset && index_slot != kNoOffset)
      {
         if (indexed_stride != 48)
         {
            out.skip_reason = "unexpected indexed instance stride";
            return false;
         }
         out.source = MatrixSource::IndexedInstanceBuffer;
         out.instance_buffer_slot = indexed_slot;
         out.instance_buffer_stride = indexed_stride;
         out.instance_index_slot = index_slot;
         out.record_row0 = 0;
      }
      else if (rows_used && out.row_cb[0] != kNoOffset && out.row_cb[1] != kNoOffset && out.row_cb[2] != kNoOffset)
      {
         out.source = MatrixSource::ConstantBuffer;
      }
      else if (mvp_used && out.mvp_cb[0] != kNoOffset && out.mvp_cb[1] != kNoOffset && out.mvp_cb[2] != kNoOffset && out.mvp_cb[3] != kNoOffset)
      {
         out.source = MatrixSource::MvpConstantBuffer;
         if (out.view_proj_cb == kNoOffset)
         {
            // Most MVP-only shaders (~190 of ~200 in the game's dump) don't declare PerViewCB, but the game keeps it bound
            // at b1 through a pass. Read it there, at its usual layout (CBPerViewGlobals: cb_viewprojectionmatrix at 32).
            // If something else is bound, the capture shader's affine check rejects the matrix.
            out.per_view_cb_slot = 1;
            out.view_proj_cb = 32;
         }
      }
      else
      {
         out.skip_reason = mvp_used ? "MVP only, no PerViewCB" : "no model matrix";
         return false;
      }
      if (out.source != MatrixSource::ConstantBuffer && out.instance_cb_slot == kNoOffset)
         out.instance_offset_cb = kNoOffset;
      if (out.instance_cb_slot == kNoOffset)
      {
         out.depth_hack_cb = kNoOffset;
         if (out.source == MatrixSource::ConstantBuffer || out.source == MatrixSource::MvpConstantBuffer)
         {
            out.skip_reason = "no PerInstanceCB";
            out.source = MatrixSource::None;
            return false;
         }
      }
      out.skip_reason = nullptr;
      return true;
   }

   InputLayoutInfo ParseInputLayout(const D3D11_INPUT_ELEMENT_DESC* elements, uint32_t count)
   {
      InputLayoutInfo info;
      uint32_t running[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = {};
      for (uint32_t i = 0; i < count; i++)
      {
         const auto& e = elements[i];
         if (e.InputSlot >= D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT)
            continue;
         const uint32_t offset = e.AlignedByteOffset == D3D11_APPEND_ALIGNED_ELEMENT ? running[e.InputSlot] : e.AlignedByteOffset;
         running[e.InputSlot] = offset + FormatSize(e.Format);
         if (e.SemanticName && _stricmp(e.SemanticName, "VERTEX_POSITION") == 0 && e.SemanticIndex == 0)
         {
            info.has_position = true;
            info.position_format = e.Format;
            info.position_slot = e.InputSlot;
            info.position_offset = offset;
            info.position_per_instance = e.InputSlotClass == D3D11_INPUT_PER_INSTANCE_DATA;
         }
      }
      return info;
   }

   DXGI_FORMAT DxrVertexFormat(DXGI_FORMAT format)
   {
      switch (format)
      {
      case DXGI_FORMAT_R32G32B32_FLOAT:
      case DXGI_FORMAT_R32G32B32A32_FLOAT: // xyz, the stride skips w
         return DXGI_FORMAT_R32G32B32_FLOAT;
      case DXGI_FORMAT_R16G16B16A16_FLOAT:
      case DXGI_FORMAT_R16G16B16A16_SNORM:
      case DXGI_FORMAT_R16G16B16A16_UNORM: // DXR 1.1
         return format;
      default:
         return DXGI_FORMAT_UNKNOWN;
      }
   }

   // ---- GeometryCapture

   bool GeometryCapture::Init(ID3D11Device* device, Interop& interop, ID3D11ComputeShader* cs, std::string& error)
   {
      const std::lock_guard lock(mutex);
      if (ring11)
         return true;
      if (!cs)
      {
         error = "capture compute shader missing";
         return false;
      }
      device11 = device;
      capture_cs = cs;
      capture_cs->AddRef();

      D3D11_BUFFER_DESC cb_desc = { sizeof(CaptureParams), D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
      HRESULT hr = device->CreateBuffer(&cb_desc, nullptr, &params_cb);
      if (FAILED(hr))
      {
         error = HResultString("CreateBuffer (capture params)", hr);
         return false;
      }
      if (!interop.CreateSharedBuffer(device, uint64_t(kRingSlots) * 64, ring12, ring11, error))
      {
         error = "capture ring: " + error;
         return false;
      }
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
      uav.Format = DXGI_FORMAT_R32_TYPELESS;
      uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav.Buffer.NumElements = kRingSlots * 64 / 4;
      uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      hr = device->CreateUnorderedAccessView(ring11, &uav, &ring_uav);
      if (FAILED(hr))
      {
         error = HResultString("CreateUnorderedAccessView (capture ring)", hr);
         return false;
      }
      // One arena for vertex and index data (a buffer can be both). Largest that the driver accepts.
      for (uint64_t size : { 1024ull << 20, 512ull << 20, 256ull << 20, 128ull << 20 })
      {
         std::string arena_error;
         if (interop.CreateSharedBuffer(device, size, vertex_arena12, vertex_arena11, arena_error))
         {
            vertex_arena_size = size;
            break;
         }
         error = arena_error;
      }
      if (!vertex_arena11)
      {
         error = "geometry arena: " + error;
         return false;
      }
      error.clear();
      for (auto& buffer : readback)
      {
         D3D12_RESOURCE_DESC desc = {};
         desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
         desc.Width = uint64_t(kReadbackSlots) * 64;
         desc.Height = 1;
         desc.DepthOrArraySize = 1;
         desc.MipLevels = 1;
         desc.SampleDesc.Count = 1;
         desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
         D3D12_HEAP_PROPERTIES heap = {};
         heap.Type = D3D12_HEAP_TYPE_READBACK;
         hr = interop.GetDevice()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer));
         if (FAILED(hr))
         {
            error = HResultString("CreateCommittedResource (capture readback)", hr);
            return false;
         }
      }
      return true;
   }

   void GeometryCapture::Shutdown()
   {
      const std::lock_guard lock(mutex);
      // The caller makes sure the D3D12 queue is idle
      for (auto& buffer : readback)
         SafeRelease(buffer);
      SafeRelease(vertex_arena12);
      SafeRelease(vertex_arena11);
      SafeRelease(index_arena12);
      SafeRelease(index_arena11);
      SafeRelease(ring_uav);
      SafeRelease(ring12);
      SafeRelease(ring11);
      SafeRelease(params_cb);
      SafeRelease(capture_cs);
      buffers.clear();
      meshes.clear();
      context_records.clear();
      list_records.clear();
      executed.clear();
      for (auto& r : readback_records)
         r.clear();
      vertex_arena_used = index_arena_used = 0;
   }

   bool GeometryCapture::EnsureBufferCopy(ID3D11DeviceContext* context, ID3D11Buffer* buffer, bool, std::vector<Rec>& records)
   {
      if (buffers.find(buffer) != buffers.end())
         return true;
      D3D11_BUFFER_DESC desc = {};
      buffer->GetDesc(&desc);
      if (desc.Usage == D3D11_USAGE_DYNAMIC || desc.Usage == D3D11_USAGE_STAGING)
         return false; // Rewritten by the CPU: a copy would go stale
      const uint64_t offset = (vertex_arena_used + 255) & ~uint64_t(255);
      if (offset + desc.ByteWidth > vertex_arena_size)
         return false;
      vertex_arena_used = offset + desc.ByteWidth;
      context->CopySubresourceRegion(vertex_arena11, 0, UINT(offset), 0, 0, buffer, 0, nullptr);
      BufferEntry entry;
      entry.arena_offset = offset;
      entry.size = desc.ByteWidth;
      buffers.emplace(buffer, entry);
      stats.buffers_copied++;
      stats.arena_vertex_bytes = vertex_arena_used;
      Rec record = {};
      record.type = Rec::BufferCopy;
      record.buffer = buffer;
      records.push_back(record);
      return true;
   }

   bool GeometryCapture::CaptureDraw(ID3D11DeviceContext* context, const DrawInfo& draw)
   {
      if (!ring11)
         return false;
      auto skip = [&](CaptureSkip reason) {
         const std::lock_guard lock(mutex);
         stats.draws_seen++;
         stats.skipped[reason]++;
         return false;
      };
      if (!draw.vs || draw.vs->source == MatrixSource::None)
         return skip(kSkipShader);
      if (draw.index_count < 3 || draw.instance_count == 0 || !draw.vertex_buffer || !draw.index_buffer || draw.vertex_stride == 0)
         return skip(kSkipOther);
      const DXGI_FORMAT vertex_format = DxrVertexFormat(draw.position_format);
      const uint32_t index_size = draw.index_format == DXGI_FORMAT_R16_UINT ? 2 : (draw.index_format == DXGI_FORMAT_R32_UINT ? 4 : 0);
      if (vertex_format == DXGI_FORMAT_UNKNOWN || index_size == 0)
         return skip(kSkipFormat);
      const MatrixSource source = draw.vs->source;
      if ((source == MatrixSource::DirectInstanceBuffer && !draw.instance_buffer) || (source == MatrixSource::IndexedInstanceBuffer && (!draw.instance_buffer || !draw.instance_index)) ||
         (source == MatrixSource::ConstantBuffer && !draw.instance_cb) || (source == MatrixSource::MvpConstantBuffer && (!draw.instance_cb || !draw.per_view_cb)) ||
         (draw.vs->instance_offset_cb != kNoOffset && !draw.instance_cb))
         return skip(kSkipBinding);
      if (draw.instance_count > kRingSlots / 8)
         return skip(kSkipRingFull);
      const int64_t vertex_zero = int64_t(draw.vertex_buffer_offset) + int64_t(draw.base_vertex) * int64_t(draw.vertex_stride);
      if (vertex_zero < 0)
         return skip(kSkipOther);
      const uint64_t vertex_start = uint64_t(vertex_zero) + draw.position_offset;
      const uint64_t index_start = uint64_t(draw.index_buffer_offset) + uint64_t(draw.first_index) * index_size;

      uint64_t key = HashCombine(uint64_t(uintptr_t(draw.vertex_buffer)), uint64_t(uintptr_t(draw.index_buffer)));
      key = HashCombine(key, vertex_start);
      key = HashCombine(key, (uint64_t(draw.vertex_stride) << 32) | uint64_t(vertex_format));
      key = HashCombine(key, index_start);
      key = HashCombine(key, (uint64_t(draw.index_count) << 32) | uint64_t(draw.index_format));

      const bool immediate = context->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE;
      std::vector<Rec> local;
      uint32_t dst_slot;
      {
         const std::lock_guard lock(mutex);
         stats.draws_seen++;
         auto it = meshes.find(key);
         if (it == meshes.end())
         {
            D3D11_BUFFER_DESC vb_desc = {};
            draw.vertex_buffer->GetDesc(&vb_desc);
            D3D11_BUFFER_DESC ib_desc = {};
            draw.index_buffer->GetDesc(&ib_desc);
            const uint64_t vertex_zero_u = uint64_t(vertex_zero);
            if (vertex_start + FormatSize(draw.position_format) > vb_desc.ByteWidth || index_start + uint64_t(draw.index_count) * index_size > ib_desc.ByteWidth)
            {
               stats.skipped[kSkipBuffer]++;
               return false;
            }
            if (!EnsureBufferCopy(context, draw.vertex_buffer, true, local) || !EnsureBufferCopy(context, draw.index_buffer, false, local))
            {
               const bool dynamic = vb_desc.Usage == D3D11_USAGE_DYNAMIC || ib_desc.Usage == D3D11_USAGE_DYNAMIC;
               stats.skipped[dynamic ? kSkipBuffer : kSkipArenaFull]++;
               // A copy may have been recorded for one of the two: keep its record
               if (!local.empty())
               {
                  if (immediate)
                     CommitLocked(local);
                  else
                     context_records[context].insert(context_records[context].end(), local.begin(), local.end());
               }
               return false;
            }
            MeshEntry mesh = {};
            mesh.vertex_buffer = draw.vertex_buffer;
            mesh.index_buffer = draw.index_buffer;
            mesh.vertex_start = vertex_start;
            mesh.vertex_count = uint32_t((vb_desc.ByteWidth - vertex_zero_u) / draw.vertex_stride);
            mesh.vertex_stride = draw.vertex_stride;
            mesh.vertex_format = vertex_format;
            mesh.index_start = index_start;
            mesh.index_count = draw.index_count;
            mesh.index_format = draw.index_format;
            meshes.emplace(key, mesh);
            stats.meshes = meshes.size();
         }
         dst_slot = uint32_t(ring_head % kRingSlots);
         ring_head += draw.instance_count;
      }

      // The matrix dispatch, with the draw's own bindings. Only the compute state it touches is saved and restored.
      ID3D11DeviceContext1* context1 = nullptr;
      context->QueryInterface(IID_PPV_ARGS(&context1));
      ID3D11ComputeShader* old_cs = nullptr;
      context->CSGetShader(&old_cs, nullptr, nullptr);
      ID3D11Buffer* old_cbs[3] = {};
      UINT old_cb_first[3] = {}, old_cb_num[3] = {};
      if (context1)
         context1->CSGetConstantBuffers1(0, 3, old_cbs, old_cb_first, old_cb_num);
      else
         context->CSGetConstantBuffers(0, 3, old_cbs);
      ID3D11ShaderResourceView* old_srvs[3] = {};
      context->CSGetShaderResources(0, 3, old_srvs);
      ID3D11UnorderedAccessView* old_uav = nullptr;
      context->CSGetUnorderedAccessViews(0, 1, &old_uav);

      D3D11_MAPPED_SUBRESOURCE mapped = {};
      bool dispatched = false;
      if (SUCCEEDED(context->Map(params_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
      {
         CaptureParams params = {};
         params.source = uint32_t(source);
         params.instance_count = draw.instance_count;
         params.dst_slot = dst_slot;
         params.slot_mask = kRingSlots - 1;
         params.instance_offset_cb = draw.vs->instance_offset_cb;
         std::memcpy(params.row_cb, draw.vs->row_cb, sizeof(params.row_cb));
         params.depth_hack_cb = draw.vs->depth_hack_cb;
         params.record_row0 = draw.vs->record_row0;
         std::memcpy(params.mvp_cb, draw.vs->mvp_cb, sizeof(params.mvp_cb));
         params.view_proj_cb = draw.vs->view_proj_cb;
         std::memcpy(mapped.pData, &params, sizeof(params));
         context->Unmap(params_cb, 0);

         context->CSSetShader(capture_cs, nullptr, 0);
         ID3D11Buffer* cbs[3] = { params_cb, draw.instance_cb, source == MatrixSource::MvpConstantBuffer ? draw.per_view_cb : nullptr };
         if (context1 && (draw.instance_cb_num_constants != 0 || draw.per_view_cb_num_constants != 0))
         {
            // A buffer bound whole (num 0) gets its own size, in the multiples of 16 constants D3D11.1 wants
            auto whole = [](ID3D11Buffer* buffer) -> UINT {
               if (!buffer)
                  return 16;
               D3D11_BUFFER_DESC desc = {};
               buffer->GetDesc(&desc);
               return (std::min)(4096u, ((desc.ByteWidth / 16) + 15) & ~15u);
            };
            UINT first[3] = { 0, draw.instance_cb_first_constant, draw.per_view_cb_first_constant };
            UINT num[3] = { 16, draw.instance_cb_num_constants ? draw.instance_cb_num_constants : whole(cbs[1]), draw.per_view_cb_num_constants ? draw.per_view_cb_num_constants : whole(cbs[2]) };
            context1->CSSetConstantBuffers1(0, 3, cbs, first, num);
         }
         else
         {
            context->CSSetConstantBuffers(0, 3, cbs);
         }
         ID3D11ShaderResourceView* srvs[3] = {
            source == MatrixSource::IndexedInstanceBuffer ? draw.instance_index : nullptr,
            source == MatrixSource::IndexedInstanceBuffer ? draw.instance_buffer : nullptr,
            source == MatrixSource::DirectInstanceBuffer ? draw.instance_buffer : nullptr,
         };
         context->CSSetShaderResources(0, 3, srvs);
         UINT initial_count = UINT(-1);
         context->CSSetUnorderedAccessViews(0, 1, &ring_uav, &initial_count);
         context->Dispatch((draw.instance_count + 63) / 64, 1, 1);
         dispatched = true;
      }

      context->CSSetShader(old_cs, nullptr, 0);
      if (context1)
         context1->CSSetConstantBuffers1(0, 3, old_cbs, old_cb_first, old_cb_num);
      else
         context->CSSetConstantBuffers(0, 3, old_cbs);
      context->CSSetShaderResources(0, 3, old_srvs);
      UINT keep_count = UINT(-1);
      context->CSSetUnorderedAccessViews(0, 1, &old_uav, &keep_count);
      SafeRelease(old_cs);
      for (auto*& p : old_cbs)
         SafeRelease(p);
      for (auto*& p : old_srvs)
         SafeRelease(p);
      SafeRelease(old_uav);
      SafeRelease(context1);

      const std::lock_guard lock(mutex);
      if (dispatched)
      {
         Rec record = {};
         record.type = Rec::Instances;
         record.mesh_key = key;
         record.first_slot = dst_slot;
         record.count = draw.instance_count;
         local.push_back(record);
         stats.draws_captured++;
         stats.instances_captured += draw.instance_count;
      }
      else
      {
         stats.skipped[kSkipOther]++;
      }
      if (immediate)
         CommitLocked(local);
      else
      {
         auto& pending = context_records[context];
         pending.insert(pending.end(), local.begin(), local.end());
      }
      return dispatched;
   }

   void GeometryCapture::CommitLocked(std::vector<Rec>& records)
   {
      for (const Rec& record : records)
      {
         if (record.type == Rec::BufferCopy)
         {
            auto it = buffers.find(record.buffer);
            if (it != buffers.end())
               it->second.executed = true;
         }
         else if (after_ao_pass)
         {
            stats.after_ao_records++; // Drawn after the AO took its depth: not in that depth, so not an occluder
         }
         else
         {
            executed.push_back(record);
         }
      }
      records.clear();
   }

   void GeometryCapture::OnFinishCommandList(ID3D11DeviceContext* deferred_context, ID3D11CommandList* command_list)
   {
      const std::lock_guard lock(mutex);
      auto it = context_records.find(deferred_context);
      if (it == context_records.end() || it->second.empty())
         return;
      auto& target = list_records[command_list];
      target.frame = frame;
      target.records.insert(target.records.end(), it->second.begin(), it->second.end());
      it->second.clear();
   }

   void GeometryCapture::OnExecuteCommandList(ID3D11CommandList* command_list)
   {
      const std::lock_guard lock(mutex);
      auto it = list_records.find(command_list);
      if (it == list_records.end())
         return;
      CommitLocked(it->second.records);
      list_records.erase(it);
   }

   std::shared_ptr<GeometryCapture::PendingRecords> GeometryCapture::TakeContextRecords(ID3D11DeviceContext* deferred_context)
   {
      auto records = std::make_shared<PendingRecords>();
      const std::lock_guard lock(mutex);
      auto it = context_records.find(deferred_context);
      if (it != context_records.end())
         records->records.swap(it->second);
      return records;
   }

   void GeometryCapture::Commit(const std::shared_ptr<PendingRecords>& records)
   {
      if (!records)
         return;
      const std::lock_guard lock(mutex);
      CommitLocked(records->records);
   }

   void GeometryCapture::MarkAOPassDone()
   {
      const std::lock_guard lock(mutex);
      after_ao_pass = true;
   }

   void GeometryCapture::OnPresent()
   {
      const std::lock_guard lock(mutex);
      frame++;
      after_ao_pass = false;
      // Command lists the game finished but never executed (or released)
      for (auto it = list_records.begin(); it != list_records.end();)
      {
         if (frame - it->second.frame > 8)
         {
            // Their buffer copies never ran: forget those buffers so they get copied again
            for (const Rec& record : it->second.records)
               if (record.type == Rec::BufferCopy)
               {
                  auto b = buffers.find(record.buffer);
                  if (b != buffers.end() && !b->second.executed)
                     buffers.erase(b);
               }
            it = list_records.erase(it);
         }
         else
         {
            ++it;
         }
      }
      // Nobody consumed the executed records (RTAO not running): drop them
      if (frame - last_record_frame > 2 && !executed.empty())
      {
         stats.late_records += executed.size();
         executed.clear();
      }
   }

   void GeometryCapture::OnDestroyBuffer(ID3D11Buffer* buffer)
   {
      const std::lock_guard lock(mutex);
      if (buffers.erase(buffer) == 0)
         return;
      for (auto it = meshes.begin(); it != meshes.end();)
      {
         if (it->second.vertex_buffer == buffer || it->second.index_buffer == buffer)
         {
            evictions.push_back(it->first);
            it = meshes.erase(it);
         }
         else
         {
            ++it;
         }
      }
      stats.meshes = meshes.size();
   }

   bool GeometryCapture::Record(ID3D12GraphicsCommandList4* list, const RecordInfo& info, BlasCache& blases, TlasBuilder& tlas, Retirer& retirer, uint32_t blas_budget, std::string& error)
   {
      struct Build
      {
         uint64_t key;
         GpuMesh mesh;
      };
      std::vector<GeometryCapture::Rec> work;
      std::vector<uint64_t> evict;
      std::vector<Build> builds;
      {
         const std::lock_guard lock(mutex);
         if (!ring12)
         {
            error = "capture not initialized";
            return false;
         }
         work.swap(executed);
         evict.swap(evictions);
         last_record_frame = frame;
         const D3D12_GPU_VIRTUAL_ADDRESS arena = vertex_arena12->GetGPUVirtualAddress();
         for (auto& pair : meshes)
         {
            if (builds.size() >= blas_budget)
               break;
            MeshEntry& mesh = pair.second;
            if (mesh.built || mesh.failed)
               continue;
            auto vb = buffers.find(mesh.vertex_buffer);
            auto ib = buffers.find(mesh.index_buffer);
            if (vb == buffers.end() || ib == buffers.end() || !vb->second.executed || !ib->second.executed)
               continue;
            Build build;
            build.key = pair.first;
            build.mesh.vertices = arena + vb->second.arena_offset + mesh.vertex_start;
            build.mesh.vertex_count = mesh.vertex_count;
            build.mesh.vertex_stride = mesh.vertex_stride;
            build.mesh.vertex_format = mesh.vertex_format;
            build.mesh.indices = arena + ib->second.arena_offset + mesh.index_start;
            build.mesh.index_count = mesh.index_count;
            build.mesh.index_format = mesh.index_format;
            builds.push_back(build);
            mesh.built = true;
         }
      }

      const uint32_t slot = info.frame_slot % kFrameSlots;

      // 1) Matrices captured 3 frames ago (this slot's last use has completed: Interop waited for its allocator)
      tlas.BeginFrame();
      uint64_t frame_instances = 0, view_model = 0;
      if (!readback_records[slot].empty())
      {
         uint64_t total = 0;
         for (const auto& r : readback_records[slot])
            total += r.second;
         const uint8_t* data = nullptr;
         D3D12_RANGE range = { 0, SIZE_T(total * 64) };
         if (SUCCEEDED(readback[slot]->Map(0, &range, (void**)&data)) && data)
         {
            uint64_t index = 0;
            for (const auto& r : readback_records[slot])
            {
               for (uint32_t i = 0; i < r.second; i++, index++)
               {
                  const float* rows = reinterpret_cast<const float*>(data + index * 64);
                  const uint32_t flags = *reinterpret_cast<const uint32_t*>(data + index * 64 + 48);
                  if (!(flags & 1u))
                     continue;
                  if (flags & 2u)
                  {
                     view_model++;
                     continue;
                  }
                  bool finite = true;
                  for (int k = 0; k < 12; k++)
                     finite = finite && std::isfinite(rows[k]) && std::fabs(rows[k]) < 1e7f;
                  if (!finite || (rows[0] == 0.f && rows[1] == 0.f && rows[2] == 0.f))
                     continue;
                  // Identity: mesh + transform quantized to ~1 cm / 1e-3 (instance buffers carry no stable id)
                  uint64_t key = r.first;
                  for (int k = 0; k < 12; k++)
                  {
                     const bool translation = (k % 4) == 3;
                     key = HashCombine(key, uint64_t(int64_t(std::llround(rows[k] * (translation ? 100.f : 1000.f)))));
                  }
                  float transform[3][4];
                  std::memcpy(transform, rows, sizeof(transform));
                  tlas.AddInstance(key, r.first, transform);
                  frame_instances++;
               }
            }
            D3D12_RANGE written = { 0, 0 };
            readback[slot]->Unmap(0, &written);
         }
         readback_records[slot].clear();
      }

      // 2) This frame's matrices: ring -> readback slot (contiguous ring ranges merged into one copy)
      uint64_t overflow = 0;
      if (!work.empty())
      {
         Transition(list, ring12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
         uint64_t dst = 0;
         uint64_t run_src = 0, run_dst = 0, run_len = 0;
         auto flush = [&]() {
            if (run_len)
               list->CopyBufferRegion(readback[slot], run_dst * 64, ring12, run_src * 64, run_len * 64);
            run_len = 0;
         };
         for (const auto& record : work)
         {
            if (record.type != GeometryCapture::Rec::Instances)
               continue;
            if (dst + record.count > kReadbackSlots)
            {
               overflow += record.count;
               continue;
            }
            // A record may wrap around the end of the ring: split it
            uint32_t remaining = record.count, src = record.first_slot;
            while (remaining)
            {
               const uint32_t len = std::min(remaining, kRingSlots - src);
               if (run_len && run_src + run_len == src && run_dst + run_len == dst)
                  run_len += len;
               else
               {
                  flush();
                  run_src = src;
                  run_dst = dst;
                  run_len = len;
               }
               dst += len;
               remaining -= len;
               src = (src + len) % kRingSlots;
            }
            readback_records[slot].emplace_back(record.mesh_key, record.count);
         }
         flush();
         Transition(list, ring12, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
      }

      // 3) Meshes whose buffers were freed (the pointer may come back as another buffer)
      for (uint64_t key : evict)
         blases.Evict(key, &retirer, info.submit_fence_value);

      // 4) New BLASes, straight from the arena
      uint32_t built = 0, failed = 0;
      std::vector<uint64_t> failed_keys;
      if (!builds.empty())
      {
         Transition(list, vertex_arena12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
         const BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         for (const Build& build : builds)
         {
            std::string build_error;
            if (blases.BuildFromGpu(ctx, build.key, build.mesh, build_error))
               built++;
            else
            {
               failed++;
               failed_keys.push_back(build.key);
            }
         }
         blases.FlushBarrier(list);
         Transition(list, vertex_arena12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
      }

      const std::lock_guard lock(mutex);
      for (uint64_t key : failed_keys)
      {
         auto it = meshes.find(key);
         if (it != meshes.end())
            it->second.failed = true;
      }
      uint64_t ready = 0;
      for (const auto& pair : meshes)
         ready += pair.second.built && !pair.second.failed;
      stats.meshes_ready = ready;
      stats.blas_built += built;
      stats.blas_failed += failed;
      stats.blas_evicted += evict.size();
      stats.frame_instances = frame_instances;
      stats.view_model_instances = view_model;
      stats.readback_overflow += overflow;
      stats.tlas_instances = tlas.RetainedCount();
      return true;
   }

   CaptureStats GeometryCapture::GetStats()
   {
      const std::lock_guard lock(mutex);
      return stats;
   }
}
