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

      // Records one capture dispatch with the draw's own bindings (b1 PerInstanceCB, b2 PerViewCB, t0..t2 instance data)
      // plus ours. Only the compute state it touches is saved and restored. Fills the matrix source part of "params".
      bool DispatchCapture(ID3D11DeviceContext* context, ID3D11Buffer* params_cb, const DrawInfo& draw, CaptureParams& params, ID3D11ComputeShader* shader, ID3D11ShaderResourceView* bones, ID3D11ShaderResourceView* arena,
         ID3D11ShaderResourceView* dynamic_vertices, ID3D11UnorderedAccessView* uav, UINT uav_slot, UINT groups)
      {
         const MatrixSource source = draw.vs->source;
         params.source = uint32_t(source);
         params.instance_offset_cb = draw.vs->instance_offset_cb;
         std::memcpy(params.row_cb, draw.vs->row_cb, sizeof(params.row_cb));
         params.depth_hack_cb = draw.vs->depth_hack_cb;
         params.record_row0 = draw.vs->record_row0;
         std::memcpy(params.mvp_cb, draw.vs->mvp_cb, sizeof(params.mvp_cb));
         params.view_proj_cb = draw.vs->view_proj_cb;

         constexpr UINT kSrvs = 6, kUavs = 2;
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
         ID3D11ShaderResourceView* old_srvs[kSrvs] = {};
         context->CSGetShaderResources(0, kSrvs, old_srvs);
         ID3D11UnorderedAccessView* old_uavs[kUavs] = {};
         context->CSGetUnorderedAccessViews(0, kUavs, old_uavs);

         D3D11_MAPPED_SUBRESOURCE mapped = {};
         bool dispatched = false;
         if (SUCCEEDED(context->Map(params_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
         {
            std::memcpy(mapped.pData, &params, sizeof(params));
            context->Unmap(params_cb, 0);

            context->CSSetShader(shader, nullptr, 0);
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
            ID3D11ShaderResourceView* srvs[kSrvs] = {
               source == MatrixSource::IndexedInstanceBuffer ? draw.instance_index : nullptr,
               source == MatrixSource::IndexedInstanceBuffer ? draw.instance_buffer : nullptr,
               source == MatrixSource::DirectInstanceBuffer ? draw.instance_buffer : nullptr,
               bones,
               arena,
               dynamic_vertices,
            };
            context->CSSetShaderResources(0, kSrvs, srvs);
            ID3D11UnorderedAccessView* uavs[kUavs] = {};
            uavs[uav_slot] = uav;
            UINT initial_counts[kUavs] = { UINT(-1), UINT(-1) };
            context->CSSetUnorderedAccessViews(0, kUavs, uavs, initial_counts);
            context->Dispatch(groups, 1, 1);
            dispatched = true;
         }

         context->CSSetShader(old_cs, nullptr, 0);
         if (context1)
            context1->CSSetConstantBuffers1(0, 3, old_cbs, old_cb_first, old_cb_num);
         else
            context->CSSetConstantBuffers(0, 3, old_cbs);
         context->CSSetShaderResources(0, kSrvs, old_srvs);
         UINT keep_counts[kUavs] = { UINT(-1), UINT(-1) };
         context->CSSetUnorderedAccessViews(0, kUavs, old_uavs, keep_counts);
         SafeRelease(old_cs);
         for (auto*& p : old_cbs)
            SafeRelease(p);
         for (auto*& p : old_srvs)
            SafeRelease(p);
         for (auto*& p : old_uavs)
            SafeRelease(p);
         SafeRelease(context1);
         return dispatched;
      }
   }

   // ---- Reflection

   bool ReflectVertexShader(const void* code, size_t code_size, VertexShaderInfo& out)
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
      uint32_t skin_index_sets = 0, skin_weight_sets = 0;
      for (UINT i = 0; i < desc.InputParameters; i++)
      {
         D3D11_SIGNATURE_PARAMETER_DESC parameter = {};
         reflection->GetInputParameterDesc(i, &parameter);
         if (!parameter.SemanticName)
            continue;
         if (_stricmp(parameter.SemanticName, "VERTEX_SKININDICES") == 0)
            skin_index_sets = (std::max)(skin_index_sets, uint32_t(parameter.SemanticIndex) + 1);
         else if (_stricmp(parameter.SemanticName, "VERTEX_SKINWEIGHTS") == 0)
            skin_weight_sets = (std::max)(skin_weight_sets, uint32_t(parameter.SemanticIndex) + 1);
         else if (_stricmp(parameter.SemanticName, "VERTEX_POSITION") == 0 && parameter.SemanticIndex == 0)
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
         {
            if (bind.Type == D3D_SIT_TEXTURE && bind.Dimension == D3D_SRV_DIMENSION_BUFFER)
               out.bones_slot = bind.BindPoint;
            else
               out.skip_reason = "skinned: unexpected ro_matrices type";
         }
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
               else if (strcmp(var.Name, "cb_mtxoffset") == 0)
                  out.mtx_offset_cb = var.StartOffset;
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
      // Skinned: every skinned vertex shader in the game's dump (1930 of them) reads ro_matrices at
      // (VERTEX_SKININDICESn + cb_mtxoffset.x) * 3 + {0, 1, 2}, weighted by VERTEX_SKINWEIGHTSn (n = 0, or 0 and 1), and
      // then applies its family's object to world (instance buffer, cb_modelmatrix or MVP) to the skinned position
      if (skin_index_sets != 0 || out.bones_slot != kNoOffset)
      {
         const char* skin_error = nullptr;
         if (out.bones_slot == kNoOffset || skin_index_sets == 0)
            skin_error = "skinned: bones or skin inputs missing";
         else if (skin_index_sets > 2 || skin_weight_sets != skin_index_sets)
            skin_error = "skinned: unexpected skin inputs";
         else if (out.instance_cb_slot == kNoOffset || out.mtx_offset_cb == kNoOffset)
            skin_error = "skinned: no cb_mtxoffset";
         if (skin_error)
         {
            out.skip_reason = skin_error;
            out.source = MatrixSource::None;
            return false;
         }
         out.skinned = true;
         out.skin_sets = skin_index_sets;
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
         if (!e.SemanticName)
            continue;
         if (_stricmp(e.SemanticName, "VERTEX_POSITION") == 0 && e.SemanticIndex == 0)
         {
            info.has_position = true;
            info.position_format = e.Format;
            info.position_slot = e.InputSlot;
            info.position_offset = offset;
            info.position_per_instance = e.InputSlotClass == D3D11_INPUT_PER_INSTANCE_DATA;
         }
         const bool skin_indices = _stricmp(e.SemanticName, "VERTEX_SKININDICES") == 0;
         if ((skin_indices || _stricmp(e.SemanticName, "VERTEX_SKINWEIGHTS") == 0) && e.SemanticIndex < 2)
         {
            InputLayoutInfo::Element& element = skin_indices ? info.skin_indices[e.SemanticIndex] : info.skin_weights[e.SemanticIndex];
            element.present = true;
            element.format = e.Format;
            element.slot = e.InputSlot;
            element.offset = offset;
            element.per_instance = e.InputSlotClass == D3D11_INPUT_PER_INSTANCE_DATA;
         }
      }
      return info;
   }

   uint32_t VertexFormatCode(DXGI_FORMAT format)
   {
      // kind | components << 4 | bits << 8 (kinds: RTAO_KIND_* in Luma_RTAO.hlsl)
      enum : uint32_t { Float = 1, Unorm, Snorm, Uint, Sint, Half, Rgb10A2Unorm, Rgb10A2Uint, Bgra8Unorm };
      auto code = [](uint32_t kind, uint32_t count, uint32_t bits) { return kind | (count << 4) | (bits << 8); };
      switch (format)
      {
      case DXGI_FORMAT_R32G32B32A32_FLOAT: return code(Float, 4, 32);
      case DXGI_FORMAT_R32G32B32_FLOAT: return code(Float, 3, 32);
      case DXGI_FORMAT_R32G32_FLOAT: return code(Float, 2, 32);
      case DXGI_FORMAT_R32_FLOAT: return code(Float, 1, 32);
      case DXGI_FORMAT_R32G32B32A32_UINT: return code(Uint, 4, 32);
      case DXGI_FORMAT_R32G32B32_UINT: return code(Uint, 3, 32);
      case DXGI_FORMAT_R32G32_UINT: return code(Uint, 2, 32);
      case DXGI_FORMAT_R32_UINT: return code(Uint, 1, 32);
      case DXGI_FORMAT_R32G32B32A32_SINT: return code(Sint, 4, 32);
      case DXGI_FORMAT_R16G16B16A16_FLOAT: return code(Half, 4, 16);
      case DXGI_FORMAT_R16G16_FLOAT: return code(Half, 2, 16);
      case DXGI_FORMAT_R16G16B16A16_UNORM: return code(Unorm, 4, 16);
      case DXGI_FORMAT_R16G16_UNORM: return code(Unorm, 2, 16);
      case DXGI_FORMAT_R16G16B16A16_SNORM: return code(Snorm, 4, 16);
      case DXGI_FORMAT_R16G16_SNORM: return code(Snorm, 2, 16);
      case DXGI_FORMAT_R16G16B16A16_UINT: return code(Uint, 4, 16);
      case DXGI_FORMAT_R16G16_UINT: return code(Uint, 2, 16);
      case DXGI_FORMAT_R16G16B16A16_SINT: return code(Sint, 4, 16);
      case DXGI_FORMAT_R8G8B8A8_UNORM: return code(Unorm, 4, 8);
      case DXGI_FORMAT_R8G8_UNORM: return code(Unorm, 2, 8);
      case DXGI_FORMAT_R8G8B8A8_SNORM: return code(Snorm, 4, 8);
      case DXGI_FORMAT_R8G8B8A8_UINT: return code(Uint, 4, 8);
      case DXGI_FORMAT_R8G8_UINT: return code(Uint, 2, 8);
      case DXGI_FORMAT_R8G8B8A8_SINT: return code(Sint, 4, 8);
      case DXGI_FORMAT_B8G8R8A8_UNORM: return code(Bgra8Unorm, 4, 8);
      case DXGI_FORMAT_R10G10B10A2_UNORM: return code(Rgb10A2Unorm, 4, 10);
      case DXGI_FORMAT_R10G10B10A2_UINT: return code(Rgb10A2Uint, 4, 10);
      default: return 0;
      }
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

   bool GeometryCapture::Init(ID3D11Device* device, Interop& interop, ID3D11ComputeShader* cs, ID3D11ComputeShader* skin_shader, std::string& error)
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

      // Skinning: optional, a failure here only leaves skinned draws out
      if (skin_shader)
      {
         std::string skin_error;
         bool ok = interop.CreateSharedBuffer(device, uint64_t(kSkinRingVertices) * 12, skin_ring12, skin_ring11, skin_error);
         if (!ok)
            skinning_error = "skinned vertex ring: " + skin_error;
         if (ok)
         {
            D3D11_UNORDERED_ACCESS_VIEW_DESC skin_uav = {};
            skin_uav.Format = DXGI_FORMAT_R32_TYPELESS;
            skin_uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            skin_uav.Buffer.NumElements = kSkinRingVertices * 3;
            skin_uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
            hr = device->CreateUnorderedAccessView(skin_ring11, &skin_uav, &skin_ring_uav);
            ok = SUCCEEDED(hr);
            if (!ok)
               skinning_error = HResultString("CreateUnorderedAccessView (skinned vertex ring)", hr);
         }
         // Raw views of the arena. Measured (RTX 4090, Win11) on a 512 MB+ buffer: only views of exactly 2^27 elements
         // (512 MB) at multiples of 512 MB are accepted, anything else returns E_INVALIDARG; smaller buffers take any
         // view. So: the largest span that works at offset 0, then views every span / 2 if those work, else every span.
         auto make_view = [&](uint64_t first, uint64_t bytes, ID3D11ShaderResourceView** view) {
            D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
            srv.BufferEx.FirstElement = UINT(first / 4);
            srv.BufferEx.NumElements = UINT(bytes / 4);
            srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
            return device->CreateShaderResourceView(vertex_arena11, &srv, view);
         };
         arena_view_span = 0;
         for (uint64_t span = (std::min)(kArenaViewMaxSpan, vertex_arena_size); ok && span >= (16ull << 20); span /= 2)
         {
            ID3D11ShaderResourceView* view = nullptr;
            if (SUCCEEDED(make_view(0, span, &view)))
            {
               view->Release();
               arena_view_span = span;
               break;
            }
         }
         if (ok && arena_view_span == 0)
         {
            ok = false;
            skinning_error = "no raw view of the geometry arena is accepted";
         }
         if (ok)
         {
            arena_view_step = arena_view_span;
            ID3D11ShaderResourceView* half_view = nullptr;
            if (arena_view_span < vertex_arena_size && SUCCEEDED(make_view(arena_view_span / 2, arena_view_span, &half_view)))
            {
               half_view->Release();
               arena_view_step = arena_view_span / 2;
            }
         }
         for (uint64_t first = 0; ok && first < vertex_arena_size; first += arena_view_step)
         {
            ID3D11ShaderResourceView* view = nullptr;
            hr = make_view(first, (std::min)(arena_view_span, vertex_arena_size - first), &view);
            ok = SUCCEEDED(hr);
            if (!ok)
               skinning_error = HResultString("CreateShaderResourceView (arena)", hr);
            else
               arena_views.push_back(view);
            if (first + arena_view_span >= vertex_arena_size)
               break; // The last view reaches the end
         }
         if (ok)
         {
            D3D11_BUFFER_DESC dynamic_desc = { UINT(kDynamicRingBytes), D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0 };
            ok = SUCCEEDED(device->CreateBuffer(&dynamic_desc, nullptr, &dynamic_ring));
            if (ok)
            {
               D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
               srv.Format = DXGI_FORMAT_R32_TYPELESS;
               srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
               srv.BufferEx.NumElements = UINT(kDynamicRingBytes / 4);
               srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
               ok = SUCCEEDED(device->CreateShaderResourceView(dynamic_ring, &srv, &dynamic_ring_srv));
            }
            if (!ok)
               skinning_error = "dynamic vertex ring failed";
         }
         if (ok)
         {
            skin_cs = skin_shader;
            skin_cs->AddRef();
         }
         else
         {
            SafeRelease(dynamic_ring_srv);
            SafeRelease(dynamic_ring);
            for (auto*& view : arena_views)
               SafeRelease(view);
            arena_views.clear();
            SafeRelease(skin_ring_uav);
            SafeRelease(skin_ring12);
            SafeRelease(skin_ring11);
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
      frame_blases.Clear(nullptr, 0);
      for (auto*& view : arena_views)
         SafeRelease(view);
      arena_views.clear();
      SafeRelease(skin_ring_uav);
      SafeRelease(skin_ring12);
      SafeRelease(skin_ring11);
      SafeRelease(skin_cs);
      SafeRelease(dynamic_ring_srv);
      SafeRelease(dynamic_ring);
      skin_head = skin_window = 0;
      dynamic_head = dynamic_window = 0;
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
      entry.recorded_on = context;
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
      if (!draw.vs || draw.vs->source == MatrixSource::None || (draw.vs->skinned && (!skin_cs || !skinning_enabled.load(std::memory_order_relaxed))))
         return skip(kSkipShader);
      if (draw.index_count < 3 || draw.instance_count == 0 || !draw.vertex_buffer || !draw.index_buffer || draw.vertex_stride == 0)
         return skip(kSkipOther);
      // A vertex buffer the CPU rewrites (movable objects: doors, windows...) only holds this frame's vertices: it goes
      // through the per frame path (copied at the draw) like skinned meshes
      bool dynamic_vertices = false;
      if (skin_cs && dynamic_enabled.load(std::memory_order_relaxed))
      {
         D3D11_BUFFER_DESC vb_desc = {};
         draw.vertex_buffer->GetDesc(&vb_desc);
         dynamic_vertices = vb_desc.Usage == D3D11_USAGE_DYNAMIC;
      }
      const bool frame_mesh = draw.vs->skinned || dynamic_vertices;
      // Per frame vertices are decoded by our shader (any VertexFormatCode format), the others go to DXR as they are
      const DXGI_FORMAT vertex_format = frame_mesh ? (VertexFormatCode(draw.position_format) ? draw.position_format : DXGI_FORMAT_UNKNOWN) : DxrVertexFormat(draw.position_format);
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

      if (frame_mesh)
         return CaptureFrameDraw(context, draw, key, index_start, index_size, dynamic_vertices);

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
               QueueLocked(context, local);
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

      // The matrix dispatch, with the draw's own bindings
      CaptureParams params = {};
      params.instance_count = draw.instance_count;
      params.dst_slot = dst_slot;
      params.slot_mask = kRingSlots - 1;
      const bool dispatched = DispatchCapture(context, params_cb, draw, params, capture_cs, nullptr, nullptr, nullptr, ring_uav, 0, (draw.instance_count + 63) / 64);

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
      QueueLocked(context, local);
      return dispatched;
   }

   void GeometryCapture::QueueLocked(ID3D11DeviceContext* context, std::vector<Rec>& local)
   {
      if (local.empty())
         return;
      if (context->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE)
         CommitLocked(local);
      else
      {
         auto& pending = context_records[context];
         pending.insert(pending.end(), local.begin(), local.end());
         local.clear();
      }
   }

   bool GeometryCapture::CaptureFrameDraw(ID3D11DeviceContext* context, const DrawInfo& draw, uint64_t key, uint64_t index_start, uint32_t index_size, bool dynamic_vertices)
   {
      const bool skinned = draw.vs->skinned;
      auto skip = [&](CaptureSkip reason) {
         const std::lock_guard lock(mutex);
         stats.draws_seen++;
         stats.skipped[reason]++;
         return false;
      };
      if (draw.instance_count != 1)
      {
         const std::lock_guard lock(mutex);
         stats.draws_seen++;
         (skinned ? stats.skinned_instanced : stats.dynamic_instanced)++;
         return false;
      }
      if (!draw.instance_cb || (skinned && !draw.bones))
         return skip(kSkipBinding);

      // Streams: 0 position, then (indices, weights) per skin set. "first" = byte offset of vertex 0's element in its buffer.
      struct Stream
      {
         ID3D11Buffer* buffer;
         uint32_t stride;
         int64_t first;
         DXGI_FORMAT format;
         uint32_t code;
         uint64_t buffer_size;
         D3D11_USAGE usage;
      };
      const uint32_t sets = skinned ? draw.vs->skin_sets : 0;
      const uint32_t stream_count = 1 + 2 * sets;
      Stream streams[5] = {};
      streams[0] = { draw.vertex_buffer, draw.vertex_stride, int64_t(draw.vertex_buffer_offset) + int64_t(draw.base_vertex) * draw.vertex_stride + draw.position_offset, draw.position_format };
      for (uint32_t set = 0; set < sets; set++)
      {
         const DrawInfo::Stream* sources[2] = { &draw.skin_indices[set], &draw.skin_weights[set] };
         for (uint32_t k = 0; k < 2; k++)
         {
            const DrawInfo::Stream& source = *sources[k];
            if (!source.buffer || source.stride == 0)
               return skip(kSkipBinding);
            streams[1 + set * 2 + k] = { source.buffer, source.stride, int64_t(source.offset) + int64_t(draw.base_vertex) * source.stride + source.element_offset, source.format };
         }
      }
      // Vertex range: from the base vertex to the end of the shortest stream (whole elements)
      uint64_t range = UINT64_MAX;
      for (uint32_t s = 0; s < stream_count; s++)
      {
         Stream& stream = streams[s];
         stream.code = VertexFormatCode(stream.format);
         if (stream.code == 0)
            return skip(kSkipFormat);
         if (stream.first < 0)
            return skip(kSkipOther);
         D3D11_BUFFER_DESC desc = {};
         stream.buffer->GetDesc(&desc);
         stream.buffer_size = desc.ByteWidth;
         stream.usage = desc.Usage;
         // Only the position stream may be dynamic (copied every draw), and only when the caller routed it here for that
         if ((desc.Usage == D3D11_USAGE_DYNAMIC || desc.Usage == D3D11_USAGE_STAGING) && !(s == 0 && dynamic_vertices))
            return skip(kSkipBuffer);
         const uint64_t element_size = FormatSize(stream.format);
         if (uint64_t(stream.first) + element_size > desc.ByteWidth)
            return skip(kSkipBuffer);
         range = (std::min)(range, (desc.ByteWidth - uint64_t(stream.first) - element_size) / stream.stride + 1);
      }
      D3D11_BUFFER_DESC ib_desc = {};
      draw.index_buffer->GetDesc(&ib_desc);
      if (index_start + uint64_t(draw.index_count) * index_size > ib_desc.ByteWidth || ib_desc.Usage == D3D11_USAGE_DYNAMIC || ib_desc.Usage == D3D11_USAGE_STAGING)
         return skip(kSkipBuffer);
      const uint32_t index_count = draw.index_count - draw.index_count % 3;
      // Bounded by the index count: skin the draw's vertex range if it is not larger, else one vertex per index (soup)
      const bool soup = range > index_count;
      const uint32_t vertex_count = soup ? index_count : uint32_t(range);
      if (vertex_count < 3)
         return skip(kSkipOther);
      // Dynamic vertices: the bytes from vertex 0 to the end of the buffer are copied (the indices may reach any of them)
      uint64_t dynamic_copy_start = 0, dynamic_copy_bytes = 0;
      if (dynamic_vertices)
      {
         dynamic_copy_start = uint64_t(streams[0].first) & ~uint64_t(3); // 4 byte aligned for the raw view
         dynamic_copy_bytes = streams[0].buffer_size - dynamic_copy_start;
         if (dynamic_copy_bytes > kDynamicMaxCopyBytes)
         {
            const std::lock_guard lock(mutex);
            stats.draws_seen++;
            stats.dynamic_too_large++;
            return false;
         }
      }

      std::vector<Rec> local;
      CaptureParams params = {};
      ID3D11ShaderResourceView* arena_view = nullptr;
      uint64_t index_arena_offset = 0;
      uint32_t dst = 0;
      uint64_t dynamic_offset = 0;
      {
         const std::lock_guard lock(mutex);
         stats.draws_seen++;
         // Arena copies (once per buffer). They must be in place when our dispatch runs: executed already, or recorded
         // on this same context (in order with it).
         ID3D11Buffer* needed[6] = {};
         uint32_t needed_count = 0;
         for (uint32_t s = dynamic_vertices ? 1 : 0; s < stream_count; s++)
            needed[needed_count++] = streams[s].buffer;
         needed[needed_count++] = draw.index_buffer;
         for (uint32_t n = 0; n < needed_count; n++)
         {
            if (!EnsureBufferCopy(context, needed[n], true, local))
            {
               stats.skipped[kSkipArenaFull]++;
               QueueLocked(context, local);
               return false;
            }
         }
         uint64_t low = UINT64_MAX, high = 0;
         for (uint32_t n = 0; n < needed_count; n++)
         {
            const BufferEntry& entry = buffers[needed[n]];
            if (!entry.executed && entry.recorded_on != context)
            {
               (skinned ? stats.skinned_waiting : stats.dynamic_waiting)++;
               QueueLocked(context, local);
               return false;
            }
            low = (std::min)(low, entry.arena_offset);
            high = (std::max)(high, entry.arena_offset + entry.size);
         }
         // One arena view must hold all of them
         const uint64_t view = arena_view_step ? low / arena_view_step : 0;
         const uint64_t view_start = view * arena_view_step;
         if (view >= arena_views.size() || high > view_start + arena_view_span)
         {
            stats.skipped[kSkipBuffer]++;
            QueueLocked(context, local);
            return false;
         }
         arena_view = arena_views[size_t(view)];
         // Budget and ring space
         if (skin_window + vertex_count > kSkinFrameBudget || (dynamic_vertices && dynamic_window + dynamic_copy_bytes > kDynamicFrameBudget))
         {
            stats.frame_overflow++;
            QueueLocked(context, local);
            return false;
         }
         if (skin_head + vertex_count > kSkinRingVertices)
         {
            skin_window += kSkinRingVertices - skin_head;
            skin_head = 0;
         }
         dst = skin_head;
         skin_head += vertex_count;
         skin_window += vertex_count;
         if (dynamic_vertices)
         {
            const uint64_t size = (dynamic_copy_bytes + 15) & ~uint64_t(15);
            if (dynamic_head + size > kDynamicRingBytes)
            {
               dynamic_window += kDynamicRingBytes - dynamic_head;
               dynamic_head = 0;
            }
            dynamic_offset = dynamic_head;
            dynamic_head += size;
            dynamic_window += size;
         }
         (soup ? stats.frame_soup : stats.frame_indexed)++;

         params.skin_vertex_count = vertex_count;
         params.skin_dst_vertex = dst;
         params.skin_soup = soup ? 1 : 0;
         params.skin_sets = sets;
         params.mtx_offset_cb = skinned ? draw.vs->mtx_offset_cb : kNoOffset;
         const BufferEntry& ib = buffers[draw.index_buffer];
         index_arena_offset = ib.arena_offset + index_start;
         params.skin_index_start = uint32_t(index_arena_offset - view_start);
         params.skin_index32 = index_size == 4 ? 1 : 0;
         for (uint32_t s = 0; s < stream_count; s++)
         {
            if (s == 0 && dynamic_vertices)
            {
               params.stream_base[s] = uint32_t(dynamic_offset + (uint64_t(streams[s].first) - dynamic_copy_start));
               params.stream_format[s] = streams[s].code | kVertexFormatDynamicBuffer;
            }
            else
            {
               params.stream_base[s] = uint32_t(buffers[streams[s].buffer].arena_offset + uint64_t(streams[s].first) - view_start);
               params.stream_format[s] = streams[s].code;
            }
            params.stream_stride[s] = streams[s].stride;
         }
      }

      if (dynamic_vertices)
      {
         // This frame's contents of the dynamic buffer, next to the draw that uses them
         const D3D11_BOX box = { UINT(dynamic_copy_start), 0, 0, UINT(dynamic_copy_start + dynamic_copy_bytes), 1, 1 };
         context->CopySubresourceRegion(dynamic_ring, 0, UINT(dynamic_offset), 0, 0, streams[0].buffer, 0, &box);
      }
      params.instance_count = 1;
      const bool dispatched = DispatchCapture(context, params_cb, draw, params, skin_cs, skinned ? draw.bones : nullptr, arena_view, dynamic_ring_srv, skin_ring_uav, 1, (vertex_count + 63) / 64);

      const std::lock_guard lock(mutex);
      if (dispatched)
      {
         Rec record = {};
         record.type = Rec::FrameMesh;
         record.mesh_key = key;
         record.first_slot = dst;
         record.count = vertex_count;
         record.index_arena_offset = index_arena_offset;
         record.index_count = index_count;
         record.index_format = soup ? DXGI_FORMAT_UNKNOWN : draw.index_format;
         local.push_back(record);
         stats.draws_captured++;
         stats.instances_captured++;
         (skinned ? stats.skinned_captured : stats.dynamic_captured)++;
      }
      else
      {
         stats.skipped[kSkipOther]++;
      }
      QueueLocked(context, local);
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
         skin_window = 0; // What was handed out so far is consumed by this pass (or dropped)
         dynamic_window = 0;
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

      // 5) Per frame meshes (skinned, dynamic vertex buffers): world space vertices written this frame by
      //    skin_vertices_cs, one BLAS per draw built or refit now, added as current-frame-only TLAS instances.
      //    Identity: the draw's mesh key + its ordinal among this frame's draws of that mesh (the CPU can't tell two
      //    characters sharing a mesh apart; if they swap order the refit still gives the right triangles, only a looser
      //    BVH until the periodic rebuild).
      uint64_t frame_draws = 0, frame_vertices = 0, frame_builds = 0, frame_refits = 0, frame_failed = 0;
      {
         const BuildContext ctx = { info.device, list, &retirer, info.submit_fence_value };
         frame_blases.BeginFrame();
         bool any = false, any_indexed = false;
         for (const auto& record : work)
         {
            if (record.type != GeometryCapture::Rec::FrameMesh)
               continue;
            any = true;
            any_indexed = any_indexed || record.index_format != DXGI_FORMAT_UNKNOWN;
         }
         if (any && skin_ring12)
         {
            Transition(list, skin_ring12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (any_indexed)
               Transition(list, vertex_arena12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            const D3D12_GPU_VIRTUAL_ADDRESS vertices = skin_ring12->GetGPUVirtualAddress();
            const D3D12_GPU_VIRTUAL_ADDRESS arena = vertex_arena12->GetGPUVirtualAddress();
            static const float identity[3][4] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
            std::unordered_map<uint64_t, uint32_t> ordinals;
            for (const auto& record : work)
            {
               if (record.type != GeometryCapture::Rec::FrameMesh)
                  continue;
               const uint64_t key = HashCombine(record.mesh_key, 0x5C1A000000000000ull + ordinals[record.mesh_key]++);
               GpuMesh mesh;
               mesh.vertices = vertices + uint64_t(record.first_slot) * 12;
               mesh.vertex_count = record.count;
               mesh.vertex_stride = 12;
               mesh.vertex_format = DXGI_FORMAT_R32G32B32_FLOAT;
               if (record.index_format != DXGI_FORMAT_UNKNOWN)
               {
                  mesh.indices = arena + record.index_arena_offset;
                  mesh.index_count = record.index_count;
                  mesh.index_format = record.index_format;
               }
               bool refit = false;
               std::string build_error;
               const D3D12_GPU_VIRTUAL_ADDRESS blas = frame_blases.Update(ctx, key, mesh, refit, build_error);
               if (!blas)
               {
                  frame_failed++;
                  continue;
               }
               tlas.AddFrameInstance(blas, identity);
               frame_draws++;
               frame_vertices += record.count;
               (refit ? frame_refits : frame_builds)++;
            }
            frame_blases.EndFrame(ctx);
            if (any_indexed)
               Transition(list, vertex_arena12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            Transition(list, skin_ring12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
         }
         else
         {
            frame_blases.EndFrame(ctx); // Evicts the BLASes of draws that went away
         }
      }

      const std::lock_guard lock(mutex);
      stats.frame_draws = frame_draws;
      stats.frame_vertices = frame_vertices;
      stats.frame_blas_builds += frame_builds;
      stats.frame_blas_refits += frame_refits;
      stats.frame_blas_failed += frame_failed;
      stats.frame_blas_count = frame_blases.Count();
      stats.frame_blas_bytes = frame_blases.ResultBytes();
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
