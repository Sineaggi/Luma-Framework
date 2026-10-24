#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rtao_capture.h"

#include <d3d11shader.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

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

      float HalfToFloat(uint32_t h)
      {
         const uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
         float value;
         if (exponent == 0)
            value = std::ldexp(float(mantissa), -24);
         else if (exponent == 31)
            value = mantissa ? NAN : INFINITY;
         else
            value = std::ldexp(float(mantissa | 1024), int(exponent) - 25);
         return sign ? -value : value;
      }

   }

   // A 3x4 row major object to world matrix that is a rotation times a (possibly non uniform) scale, plus a translation:
   // its three columns (the model axes in world space) are perpendicular and of sane lengths. An MVP derived matrix
   // computed with the wrong view projection (MVP x inverse(another view's VP)) carries that view's skew and fails.
   bool IsRigidTransform(const float rows[12])
   {
      float axis[3][3];
      float length[3];
      for (int c = 0; c < 3; c++)
      {
         for (int r = 0; r < 3; r++)
            axis[c][r] = rows[r * 4 + c];
         length[c] = std::sqrt(axis[c][0] * axis[c][0] + axis[c][1] * axis[c][1] + axis[c][2] * axis[c][2]);
         if (!(length[c] > 1e-4f && length[c] < 1e4f))
            return false;
      }
      const float shortest = (std::min)(length[0], (std::min)(length[1], length[2]));
      const float longest = (std::max)(length[0], (std::max)(length[1], length[2]));
      if (longest > shortest * 1000.f)
         return false;
      for (int a = 0; a < 3; a++)
         for (int b = a + 1; b < 3; b++)
         {
            const float cosine = (axis[a][0] * axis[b][0] + axis[a][1] * axis[b][1] + axis[a][2] * axis[b][2]) / (length[a] * length[b]);
            if (std::fabs(cosine) > 0.02f)
               return false;
         }
      return true;
   }

   namespace
   {
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
         // Cutouts: the draw's alpha reference and uv scale/bias, from its pixel shader's cbuffer (b3)
         const bool material = draw.ps && draw.material_cb;
         params.alpha_ref_cb = material ? draw.ps->alpha_ref_cb : kNoOffset;
         params.uv_scalebias_cb = material ? draw.ps->uv_scalebias_cb : kNoOffset;
         params.alpha_ref_const = draw.ps ? draw.ps->alpha_ref : 0.5f;

         constexpr UINT kCbs = 4, kSrvs = 6, kUavs = 2;
         ID3D11DeviceContext1* context1 = nullptr;
         context->QueryInterface(IID_PPV_ARGS(&context1));
         ID3D11ComputeShader* old_cs = nullptr;
         context->CSGetShader(&old_cs, nullptr, nullptr);
         ID3D11Buffer* old_cbs[kCbs] = {};
         UINT old_cb_first[kCbs] = {}, old_cb_num[kCbs] = {};
         if (context1)
            context1->CSGetConstantBuffers1(0, kCbs, old_cbs, old_cb_first, old_cb_num);
         else
            context->CSGetConstantBuffers(0, kCbs, old_cbs);
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
            ID3D11Buffer* cbs[kCbs] = { params_cb, draw.instance_cb, source == MatrixSource::MvpConstantBuffer ? draw.per_view_cb : nullptr, material ? draw.material_cb : nullptr };
            if (context1 && (draw.instance_cb_num_constants != 0 || draw.per_view_cb_num_constants != 0 || (material && draw.material_cb_num_constants != 0)))
            {
               // A buffer bound whole (num 0) gets its own size, in the multiples of 16 constants D3D11.1 wants
               auto whole = [](ID3D11Buffer* buffer) -> UINT {
                  if (!buffer)
                     return 16;
                  D3D11_BUFFER_DESC desc = {};
                  buffer->GetDesc(&desc);
                  return (std::min)(4096u, ((desc.ByteWidth / 16) + 15) & ~15u);
               };
               UINT first[kCbs] = { 0, draw.instance_cb_first_constant, draw.per_view_cb_first_constant, material ? draw.material_cb_first_constant : 0 };
               UINT num[kCbs] = { 16, draw.instance_cb_num_constants ? draw.instance_cb_num_constants : whole(cbs[1]), draw.per_view_cb_num_constants ? draw.per_view_cb_num_constants : whole(cbs[2]),
                  (material && draw.material_cb_num_constants) ? draw.material_cb_num_constants : whole(cbs[3]) };
               context1->CSSetConstantBuffers1(0, kCbs, cbs, first, num);
            }
            else
            {
               context->CSSetConstantBuffers(0, kCbs, cbs);
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
            context1->CSSetConstantBuffers1(0, kCbs, old_cbs, old_cb_first, old_cb_num);
         else
            context->CSSetConstantBuffers(0, kCbs, old_cbs);
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
      // The game's scene shaders are all SM 5.0; its UI shaders are SM 4.0 (and use POSITION, not VERTEX_POSITION).
      // UI draws can have the main depth target bound (Luma keeps it bound for UI stencil masks).
      if (D3D11_SHVER_GET_MAJOR(desc.Version) < 5)
      {
         out.skip_reason = "UI (shader model 4)";
         reflection->Release();
         return false;
      }

      bool has_position = false;
      uint32_t skin_index_sets = 0, skin_weight_sets = 0;
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
            out.wind = true; // Captured in rest pose, through its matrix family (RTAO_NOTES.md "Wind")
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
               if (used && strncmp(var.Name, "cb_vegetanim_", 13) == 0)
                  out.wind = true; // Vegetation bending with a constant wind (no ro_windbuffer)
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
         if (_stricmp(e.SemanticName, "VERTEX_TEXCOORD") == 0 && e.SemanticIndex == 0)
         {
            info.texcoord0.present = true;
            info.texcoord0.format = e.Format;
            info.texcoord0.slot = e.InputSlot;
            info.texcoord0.offset = offset;
            info.texcoord0.per_instance = e.InputSlotClass == D3D11_INPUT_PER_INSTANCE_DATA;
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

   // ---- Pixel shaders: alpha test detection

   bool DxbcHasDiscard(const void* code, size_t code_size)
   {
      // DXBC container: "DXBC", 16 byte checksum, version, total size, chunk count, chunk offsets. Each chunk: fourcc,
      // size, data. The shader program chunk (SHEX, or SHDR for SM4) holds the version token, the length in tokens,
      // then instructions: opcode in bits 0..10, instruction length in tokens in bits 24..30 (customdata, opcode 53,
      // stores its length in the next token instead).
      const auto* bytes = static_cast<const uint8_t*>(code);
      auto u32 = [&](size_t offset) {
         uint32_t v;
         std::memcpy(&v, bytes + offset, 4);
         return v;
      };
      if (!code || code_size < 32 || std::memcmp(bytes, "DXBC", 4) != 0)
         return false;
      const uint32_t chunk_count = u32(28);
      if (chunk_count > 64 || 32 + size_t(chunk_count) * 4 > code_size)
         return false;
      for (uint32_t c = 0; c < chunk_count; c++)
      {
         const size_t chunk = u32(32 + c * 4);
         if (chunk + 8 > code_size)
            continue;
         if (std::memcmp(bytes + chunk, "SHEX", 4) != 0 && std::memcmp(bytes + chunk, "SHDR", 4) != 0)
            continue;
         const size_t data = chunk + 8;
         const size_t data_size = (std::min)(size_t(u32(chunk + 4)), code_size - data);
         if (data_size < 8)
            return false;
         const size_t tokens = (std::min)(size_t(u32(data + 4)), data_size / 4);
         constexpr uint32_t kOpcodeDiscard = 13; // D3D10_SB_OPCODE_DISCARD
         constexpr uint32_t kOpcodeCustomData = 53; // D3D10_SB_OPCODE_CUSTOMDATA
         for (size_t i = 2; i < tokens;)
         {
            const uint32_t token = u32(data + i * 4);
            const uint32_t opcode = token & 0x7FF;
            uint32_t length = (token >> 24) & 0x7F;
            if (opcode == kOpcodeCustomData)
               length = i + 1 < tokens ? u32(data + (i + 1) * 4) : 0;
            if (opcode == kOpcodeDiscard)
               return true;
            if (length == 0)
               return false; // Malformed
            i += length;
         }
         return false;
      }
      return false;
   }

   namespace
   {
      // The first read of an "alpha_slots" texture in program order, and how its coordinate is formed (texture exact
      // cutouts replay it: see rtao_alpha.h)
      struct AlphaRead
      {
         bool found = false;
         bool sampled = false; // sample*/gather4* (normalized uv), not ld (texel coordinates)
         uint32_t address_type = 0; // Operand type of the coordinate: 1 input (v#), 0 temp (r#), other: unknown
         // A temp coordinate's producer: "mov r, v#" (direct) or "mad r, v#, cbS[k], cbS[k]" (scale/bias at cbS[k])
         bool direct = false;
         bool scalebias = false;
         uint32_t scalebias_cb_slot = 0, scalebias_register = 0;
         // The reference, if the first use of the value read is "add r, value, l(-reference)" (vegetation: 0.5)
         bool reference_found = false;
         float reference = 0.5f;
         // Internal: where the counted channel landed (temp register component), and whether its first use was seen
         bool value_tracked = false, value_checked = false;
         uint32_t value_register = 0, value_component = 0;
      };

      // Forward data flow over the shader program: which temp register components hold a value derived from a read of
      // one of the "alpha_slots" textures, and whether a discard's condition (or an enclosing if) is such a value.
      // Per component for temps (r#), one flag per indexable temp array (x#). Conservative where it doesn't matter
      // (all components a source operand selects taint every written component). "channel_mask": which channels of the
      // texture read count (bit c: channel c, through the resource swizzle). "read": receives the first alpha read.
      bool DiscardReadsTextures(const uint8_t* bytes, size_t code_size, const bool (&alpha_slots)[128], uint32_t channel_mask = 0xF, AlphaRead* read = nullptr)
      {
         auto u32 = [&](size_t offset) {
            uint32_t v;
            std::memcpy(&v, bytes + offset, 4);
            return v;
         };
         if (code_size < 32 || std::memcmp(bytes, "DXBC", 4) != 0)
            return false;
         const uint32_t chunk_count = u32(28);
         if (chunk_count > 64 || 32 + size_t(chunk_count) * 4 > code_size)
            return false;
         size_t data = 0, tokens = 0;
         for (uint32_t c = 0; c < chunk_count && !data; c++)
         {
            const size_t chunk = u32(32 + c * 4);
            if (chunk + 16 > code_size || (std::memcmp(bytes + chunk, "SHEX", 4) != 0 && std::memcmp(bytes + chunk, "SHDR", 4) != 0))
               continue;
            data = chunk + 8;
            tokens = (std::min)(size_t(u32(data + 4)), (std::min)(size_t(u32(chunk + 4)), code_size - data) / 4);
         }
         if (!data)
            return false;
         auto token = [&](size_t i) { return u32(data + i * 4); };

         enum : uint32_t
         {
            kTemp = 0, kInput = 1, kIndexableTemp = 3, kImmediate32 = 4, kImmediate64 = 5, kResource = 7, kConstantBuffer = 8,
         };
         struct Operand
         {
            uint32_t type;
            uint32_t index; // First index (register number), if any
            uint32_t components; // Bit mask: write mask (dest) or components read (source)
            uint32_t index2; // Second index (cb#[index2]), if any
            uint32_t swizzle; // Source component c reads component (swizzle >> 2c) & 3
            uint32_t immediate; // Immediate32: the first value
         };
         // Decodes one operand at "i" (advanced past it). False if malformed.
         std::function<bool(size_t&, size_t, Operand&)> parse = [&](size_t& i, size_t end, Operand& op) -> bool {
            if (i >= end)
               return false;
            const uint32_t t = token(i++);
            for (uint32_t ext = t >> 31; ext;)
            {
               if (i >= end)
                  return false;
               ext = token(i++) >> 31;
            }
            const uint32_t num_components = t & 3, selection = (t >> 2) & 3;
            op = { (t >> 12) & 0xFF, 0, 0, 0, 0xE4, 0 };
            if (num_components == 2)
            {
               if (selection == 0)
                  op.components = (t >> 4) & 0xF;
               else if (selection == 1)
               {
                  op.swizzle = (t >> 4) & 0xFF;
                  for (uint32_t c = 0; c < 4; c++)
                     op.components |= 1u << ((t >> (4 + 2 * c)) & 3);
               }
               else
               {
                  op.components = 1u << ((t >> 4) & 3);
                  op.swizzle = ((t >> 4) & 3) * 0x55;
               }
            }
            else if (num_components == 1)
            {
               op.components = 1;
            }
            if (op.type == kImmediate32 || op.type == kImmediate64)
            {
               const size_t count = num_components == 1 ? 1 : (num_components == 2 ? 4 : 0);
               if (count && i < end)
                  op.immediate = token(i);
               i += count * (op.type == kImmediate64 ? 2 : 1);
               return i <= end;
            }
            const uint32_t index_dimension = (t >> 20) & 3;
            for (uint32_t d = 0; d < index_dimension; d++)
            {
               const uint32_t representation = (t >> (22 + 3 * d)) & 7;
               if (representation == 0 || representation == 3) // Immediate32 (+ relative)
               {
                  if (i >= end)
                     return false;
                  if (d == 0)
                     op.index = token(i);
                  else if (d == 1)
                     op.index2 = token(i);
                  i++;
               }
               else if (representation == 1 || representation == 4) // Immediate64 (+ relative)
               {
                  i += 2;
               }
               if (representation == 2 || representation == 3 || representation == 4) // Relative: an operand follows
               {
                  Operand relative;
                  if (!parse(i, end, relative))
                     return false;
               }
            }
            return i <= end;
         };

         std::vector<uint8_t> temps; // 4 bits per temp register: tainted components
         std::vector<uint8_t> indexable; // Per x# array
         std::vector<bool> if_stack;
         auto taint_of = [&](const Operand& op) -> bool {
            if (op.type == kTemp)
               return op.index < temps.size() && (temps[op.index] & op.components) != 0;
            if (op.type == kIndexableTemp)
               return op.index < indexable.size() && indexable[op.index] != 0;
            return false;
         };
         constexpr uint32_t kOpDiscard = 13, kOpIf = 31, kOpEndIf = 21, kOpCustomData = 53;
         auto is_declaration = [](uint32_t op) { return (op >= 88 && op <= 106) || (op >= 143 && op <= 162) || (op >= 113 && op <= 116); };
         auto dest_count = [](uint32_t op) -> uint32_t {
            switch (op)
            {
            // No destination: flow control, discard, emit, stores
            case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9: case 10: case 13: case 18: case 19: case 20: case 21: case 22:
            case 23: case 31: case 44: case 48: case 58: case 62: case 63: case 76: case 117: case 118: case 119: case 164: case 166: case 168:
            case 169: case 170: case 171: case 172: case 173: case 174: case 175: case 176: case 177:
               return 0;
            // Two destinations: imul, sincos, udiv, umul, uaddc, usubb, swapc
            case 38: case 77: case 78: case 81: case 132: case 133: case 142:
               return 2;
            default:
               return 1;
            }
         };
         auto reads_texture = [](uint32_t op) {
            // ld, ld_ms, sample (all variants), gather4 (all variants)
            return op == 45 || op == 46 || (op >= 69 && op <= 74) || op == 109 || (op >= 126 && op <= 128);
         };

         // Decodes the instruction at "i": opcode, operands, end. False if malformed.
         auto decode = [&](size_t i, uint32_t& op, size_t& end, Operand (&operands)[8], uint32_t& count) -> bool {
            const uint32_t t = token(i);
            op = t & 0x7FF;
            const size_t length = op == kOpCustomData ? (i + 1 < tokens ? token(i + 1) : 0) : ((t >> 24) & 0x7F);
            if (length == 0 || i + length > tokens)
               return false;
            end = i + length;
            count = 0;
            if (op == kOpCustomData || is_declaration(op))
               return true;
            size_t j = i + 1;
            for (uint32_t ext = t >> 31; ext && j < end;)
               ext = token(j++) >> 31;
            while (j < end && count < 8)
               if (!parse(j, end, operands[count++]))
                  return false;
            return true;
         };
         std::vector<size_t> writers; // Per temp register component (4 per register): the last instruction writing it
         bool read_recorded = false;

         for (size_t i = 2; i < tokens;)
         {
            uint32_t op = 0, count = 0;
            size_t end = 0;
            Operand operands[8];
            if (!decode(i, op, end, operands, count))
               return false; // Malformed
            if (op == kOpCustomData || is_declaration(op))
            {
               i = end;
               continue;
            }
            const uint32_t dests = (std::min)(dest_count(op), count);
            if (read && read->value_tracked && !read->value_checked)
            {
               // The first instruction reading the value: "add r, value, l(-reference)"?
               for (uint32_t k = dests; k < count; k++)
               {
                  const Operand& source = operands[k];
                  if (source.type != kTemp || source.index != read->value_register || (source.components & (1u << read->value_component)) == 0)
                     continue;
                  read->value_checked = true;
                  const Operand* other = (op == 0 && count == dests + 2) ? &operands[k == dests ? dests + 1 : dests] : nullptr;
                  if (other && other->type == kImmediate32)
                  {
                     float value;
                     std::memcpy(&value, &other->immediate, 4);
                     if (-value > 0.f && -value < 1.f)
                     {
                        read->reference_found = true;
                        read->reference = -value;
                     }
                  }
                  break;
               }
               for (uint32_t k = 0; k < dests && !read->value_checked; k++)
                  if (operands[k].type == kTemp && operands[k].index == read->value_register && (operands[k].components & (1u << read->value_component)) != 0)
                     read->value_checked = true; // Overwritten unused
            }
            bool tainted = false;
            uint32_t read_mask = 0; // Dest components holding a counted channel of an alpha texture read
            for (uint32_t k = dests; k < count; k++)
            {
               tainted = tainted || taint_of(operands[k]);
               if (reads_texture(op) && operands[k].type == kResource && operands[k].index < 128 && alpha_slots[operands[k].index] && dests > 0)
               {
                  for (uint32_t c = 0; c < 4; c++)
                     if (channel_mask & (1u << ((operands[k].swizzle >> (2 * c)) & 3)))
                        read_mask |= 1u << c;
                  if (read && !read_recorded && count > dests)
                  {
                     // The coordinate (first source) and, for a temp, its producer
                     read_recorded = true;
                     const Operand& address = operands[dests];
                     read->found = true;
                     read->sampled = op != 45 && op != 46;
                     const uint32_t value_components = read_mask & operands[0].components;
                     if (operands[0].type == kTemp && value_components != 0)
                     {
                        read->value_tracked = true;
                        read->value_register = operands[0].index;
                        for (uint32_t c = 4; c-- > 0;)
                           if (value_components & (1u << c))
                              read->value_component = c;
                     }
                     read->address_type = address.type;
                     const size_t writer_index = address.type == kTemp ? size_t(address.index) * 4 + (address.swizzle & 3) : SIZE_MAX;
                     if (writer_index < writers.size() && writers[writer_index] != 0)
                     {
                        uint32_t w_op = 0, w_count = 0;
                        size_t w_end = 0;
                        Operand w[8];
                        if (decode(writers[writer_index], w_op, w_end, w, w_count))
                        {
                           constexpr uint32_t kOpMad = 50, kOpMov = 54;
                           if (w_op == kOpMov && w_count == 2 && w[1].type == kInput)
                              read->direct = true;
                           else if (w_op == kOpMad && w_count == 4 && w[1].type == kInput && w[2].type == kConstantBuffer && w[3].type == kConstantBuffer && w[2].index == w[3].index &&
                              w[2].index2 == w[3].index2)
                           {
                              read->scalebias = true;
                              read->scalebias_cb_slot = w[2].index;
                              read->scalebias_register = w[2].index2;
                           }
                        }
                     }
                  }
               }
            }
            for (uint32_t k = 0; k < dests; k++)
            {
               const Operand& d = operands[k];
               if (d.type == kTemp && d.index < 4096)
               {
                  if (d.index >= temps.size())
                  {
                     temps.resize(d.index + 1, 0);
                     writers.resize(size_t(d.index + 1) * 4, 0);
                  }
                  const uint32_t taint_mask = tainted ? d.components : (d.components & read_mask);
                  temps[d.index] = uint8_t((temps[d.index] & ~d.components) | taint_mask);
                  for (uint32_t c = 0; c < 4; c++)
                     if (d.components & (1u << c))
                        writers[size_t(d.index) * 4 + c] = i;
               }
               else if (d.type == kIndexableTemp && d.index < 4096)
               {
                  if (d.index >= indexable.size())
                     indexable.resize(d.index + 1, 0);
                  indexable[d.index] |= (tainted || read_mask) ? 1 : 0;
               }
            }
            tainted = tainted || read_mask != 0; // For a texture read used as an if/discard condition directly
            if (op == kOpIf)
               if_stack.push_back(tainted);
            else if (op == kOpEndIf && !if_stack.empty())
               if_stack.pop_back();
            else if (op == kOpDiscard && (tainted || std::find(if_stack.begin(), if_stack.end(), true) != if_stack.end()))
               return true;
            i = end;
         }
         return false;
      }
   }

   bool ReflectPixelShader(const void* code, size_t code_size, PixelShaderInfo& out)
   {
      out = {};
      out.discard = DxbcHasDiscard(code, code_size);
      if (!out.discard)
         return true;
      // The game discards for several reasons: alpha test (diffuse alpha against cb_defaultalphatest, cb_hairalphatest,
      // or a hardcoded 0.5 in the vegetation shaders), but also LOD dithering (cb_ditheringdistance, from the pixel
      // position) and the dissolve/immolation effects (noise textures against a threshold), which leave opaque meshes
      // opaque. So a shader is a cutout if a discard depends on a read of its diffuse/opacity/alpha texture.
      auto reflect = GetD3DReflect();
      if (!reflect)
         return false;
      ID3D11ShaderReflection* reflection = nullptr;
      if (FAILED(reflect(code, code_size, kIID_ID3D11ShaderReflection, reinterpret_cast<void**>(&reflection))) || !reflection)
         return false;
      D3D11_SHADER_DESC desc = {};
      reflection->GetDesc(&desc);
      bool alpha_slots[128] = {};
      bool any = false;
      for (UINT r = 0; r < desc.BoundResources; r++)
      {
         D3D11_SHADER_INPUT_BIND_DESC bind = {};
         if (FAILED(reflection->GetResourceBindingDesc(r, &bind)) || !bind.Name || bind.Type != D3D_SIT_TEXTURE)
            continue;
         if (!strstr(bind.Name, "diffuse") && !strstr(bind.Name, "opacity") && !strstr(bind.Name, "alpha"))
            continue;
         for (UINT k = 0; k < (std::max)(bind.BindCount, 1u) && bind.BindPoint + k < 128; k++)
            alpha_slots[bind.BindPoint + k] = any = true;
      }
      out.alpha_tested = any && DiscardReadsTextures(static_cast<const uint8_t*>(code), code_size, alpha_slots);
      if (out.alpha_tested)
      {
         // Which texture: the first slot that alone makes a discard depend on it. Which channel: alpha first.
         AlphaRead read;
         for (uint32_t slot = 0; slot < 128 && out.alpha_slot == kNoOffset; slot++)
         {
            if (!alpha_slots[slot])
               continue;
            bool single[128] = {};
            single[slot] = true;
            for (uint32_t channel : { 3u, 0u, 1u, 2u })
            {
               AlphaRead candidate;
               if (DiscardReadsTextures(static_cast<const uint8_t*>(code), code_size, single, 1u << channel, &candidate))
               {
                  out.alpha_slot = slot;
                  out.alpha_channel = channel;
                  read = candidate;
                  break;
               }
            }
         }
         // The alpha reference and the uv scale/bias (used variables only): same cbuffer in every alpha tested shader of
         // the game's dump (PerInstanceCB, see RTAO_NOTES.md "Texture exact cutouts")
         for (UINT c = 0; c < desc.ConstantBuffers; c++)
         {
            ID3D11ShaderReflectionConstantBuffer* cb = reflection->GetConstantBufferByIndex(c);
            D3D11_SHADER_BUFFER_DESC cb_desc = {};
            if (!cb || FAILED(cb->GetDesc(&cb_desc)) || !cb_desc.Name || cb_desc.Type != D3D_CT_CBUFFER)
               continue;
            uint32_t ref = kNoOffset, scalebias = kNoOffset;
            for (UINT v = 0; v < cb_desc.Variables; v++)
            {
               D3D11_SHADER_VARIABLE_DESC var = {};
               if (FAILED(cb->GetVariableByIndex(v)->GetDesc(&var)) || !var.Name || (var.uFlags & D3D_SVF_USED) == 0)
                  continue;
               if (strcmp(var.Name, "cb_defaultalphatest") == 0 || strcmp(var.Name, "cb_hairalphatest") == 0)
                  ref = var.StartOffset;
               else if (strcmp(var.Name, "cb_layer0_scalebias") == 0)
                  scalebias = var.StartOffset;
            }
            if (ref == kNoOffset && scalebias == kNoOffset)
               continue;
            D3D11_SHADER_INPUT_BIND_DESC bind = {};
            if (FAILED(reflection->GetResourceBindingDescByName(cb_desc.Name, &bind)))
               continue;
            if (out.material_cb_slot != kNoOffset && out.material_cb_slot != bind.BindPoint)
               continue; // Spread over two cbuffers: keep the first (never seen in the game)
            out.material_cb_slot = bind.BindPoint;
            if (ref != kNoOffset)
               out.alpha_ref_cb = ref;
            if (scalebias != kNoOffset)
               out.uv_scalebias_cb = scalebias;
         }
         // The coordinate of the alpha read: an interpolator as is, or "interpolator * cb_layer0_scalebias.xy + .zw" (the two
         // forms of the 1406 non FX alpha tested shaders of the dump but 8). Anything else stays on the stochastic fallback.
         if (read.found && read.sampled)
         {
            if (read.address_type == 1 || read.direct)
            {
               out.uv_scalebias_cb = kNoOffset;
               out.texture_exact = true;
            }
            else if (read.scalebias && out.uv_scalebias_cb != kNoOffset && read.scalebias_cb_slot == out.material_cb_slot && read.scalebias_register * 16 == out.uv_scalebias_cb)
            {
               out.texture_exact = true;
            }
         }
         if (!out.texture_exact)
            out.uv_scalebias_cb = kNoOffset;
         // Without a reference constant: the immediate the value is compared to (vegetation 0.5). Other forms (alpha against
         // a dither pattern or a dissolve noise, ~160 shaders) keep 0.5, which approximates their coverage.
         if (out.alpha_ref_cb == kNoOffset && read.reference_found)
            out.alpha_ref = read.reference;
         out.alpha_ref_exact = out.alpha_ref_cb != kNoOffset || read.reference_found;
      }
      reflection->Release();
      return true;
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
      frame_identities.clear();
      executed.clear();
      for (auto& r : readback_records)
         r.clear();
      vertex_arena_used = index_arena_used = 0;
      cutouts.Shutdown();
      cutout_bindings = {};
   }

   bool GeometryCapture::InitCutouts(ID3D11Device* device, Interop& interop, ID3D11ComputeShader* strip_cs, std::string& error)
   {
      if (!ring11)
      {
         error = "capture not initialized";
         return false;
      }
      return cutouts.Init(device, interop, strip_cs, error);
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
      // Shadow map draws: no skinned meshes (once per cascade, the most expensive capture), see DrawInfo::shadow_view.
      // MVP per frame meshes (openable windows, doors) are taken: Luma_RTAO.hlsl ResolveInstance rejects a non rigid MVP
      // derived matrix on the GPU (NaN vertices, inactive in DXR).
      if (draw.shadow_view && draw.vs->skinned)
      {
         const std::lock_guard lock(mutex);
         stats.shadow_skipped++;
         stats.shadow_skipped_skinned++;
         return false;
      }
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

      // Per frame identity: the mesh plus everything that selects its instances' data. A second draw of it in the same
      // frame (another pass) would capture the same instances again (or another per frame BLAS of the same pose).
      uint64_t identity = 0;
      if (dedup_enabled.load(std::memory_order_relaxed))
      {
         identity = HashCombine(key, uint64_t(uintptr_t(draw.instance_cb)));
         identity = HashCombine(identity, (uint64_t(draw.instance_cb_first_constant) << 32) | draw.instance_cb_num_constants);
         identity = HashCombine(identity, uint64_t(uintptr_t(draw.instance_buffer)));
         identity = HashCombine(identity, uint64_t(uintptr_t(draw.instance_index)));
         identity = HashCombine(identity, uint64_t(uintptr_t(draw.per_view_cb)) ^ (uint64_t(draw.per_view_cb_first_constant) << 44));
         identity = HashCombine(identity, uint64_t(uintptr_t(draw.bones)));
         identity = HashCombine(identity, uint64_t(draw.instance_count) | (uint64_t(draw.cutout) << 32) | (uint64_t(frame_mesh) << 33));
         identity = HashCombine(identity, draw.binding_version);
         identity |= 1; // Never 0
         const std::lock_guard lock(mutex);
         if (frame_identities.count(identity))
         {
            (frame_mesh ? stats.dedup_frame_meshes : stats.dedup_draws)++;
            return true; // Its instances are in this frame's records already
         }
      }

      if (frame_mesh)
         return CaptureFrameDraw(context, draw, key, identity, index_start, index_size, dynamic_vertices);

      // Texture exact cutouts: the draw's material (its alpha texture), whose mask is rendered at its first draw
      uint64_t material = 0;
      if (draw.cutout && draw.ps && draw.ps->texture_exact && draw.alpha_srv && draw.uv.buffer && draw.uv.stride && cutouts_exact.load(std::memory_order_relaxed))
      {
         const uint32_t uv_code = VertexFormatCode(draw.uv.format);
         const uint32_t kind = uv_code & 15, components = (uv_code >> 4) & 15;
         if ((kind == 1 || kind == 2 || kind == 3 || kind == 6) && components >= 2) // float, unorm, snorm, half (rtao_cutout.hlsli)
         {
            uint32_t tile = CutoutMaterials::kNoTile;
            material = cutouts.AcquireMaterial(draw.alpha_srv, draw.ps->alpha_channel, tile);
            if (tile != CutoutMaterials::kNoTile)
            {
               if (cutouts.RecordStrip(context, draw.alpha_srv, draw.ps->alpha_channel, tile))
               {
                  std::vector<Rec> strip(1, Rec{});
                  strip[0].type = Rec::AlphaStrip;
                  strip[0].mesh_key = material;
                  strip[0].first_slot = tile;
                  const std::lock_guard lock(mutex);
                  QueueLocked(context, strip);
               }
               else
               {
                  cutouts.AbandonStrip(material, tile);
               }
            }
         }
      }
      CutoutGeometry geometry;

      std::vector<Rec> local;
      uint32_t dst_slot;
      uint32_t record_tag = 0;
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
         if (material)
         {
            // TEXCOORD0 in the arena too (usually the same buffer as the positions), next to the mesh's indices
            const int64_t uv_zero = int64_t(draw.uv.offset) + int64_t(draw.base_vertex) * int64_t(draw.uv.stride) + int64_t(draw.uv.element_offset);
            auto ib = buffers.find(draw.index_buffer);
            if (uv_zero < 0 || ib == buffers.end() || !EnsureBufferCopy(context, draw.uv.buffer, true, local))
               material = 0;
            else
            {
               const uint64_t index_offset = ib->second.arena_offset + index_start;
               const uint64_t uv_offset = buffers[draw.uv.buffer].arena_offset + uint64_t(uv_zero);
               geometry.index_offset = uint32_t(index_offset);
               geometry.uv_offset = uint32_t(uv_offset);
               geometry.uv_stride = draw.uv.stride;
               geometry.uv_code = VertexFormatCode(draw.uv.format);
               geometry.index32 = index_size == 4 ? 1 : 0;
               if (index_offset > UINT32_MAX || uv_offset > UINT32_MAX)
                  material = 0;
            }
         }
         dst_slot = uint32_t(ring_head % kRingSlots);
         ring_head += draw.instance_count;
         record_tag = next_record_tag++ & 0xFFFFFF;
      }

      // The matrix dispatch, with the draw's own bindings
      CaptureParams params = {};
      params.instance_count = draw.instance_count;
      params.dst_slot = dst_slot;
      params.record_tag = record_tag;
      params.slot_mask = kRingSlots - 1;
      RTAO::DrawInfo capture_draw = draw;
      if (!material)
         capture_draw.ps = nullptr; // No reference/scale-bias reads
      const bool dispatched = DispatchCapture(context, params_cb, capture_draw, params, capture_cs, nullptr, nullptr, nullptr, ring_uav, 0, (draw.instance_count + 63) / 64);

      const std::lock_guard lock(mutex);
      if (dispatched)
      {
         Rec record = {};
         record.type = Rec::Instances;
         record.mask = draw.cutout ? kInstanceMaskCutout : kInstanceMaskOpaque;
         record.mesh_key = key;
         record.first_slot = dst_slot;
         record.count = draw.instance_count;
         record.material = draw.cutout ? material : 0;
         record.geometry = geometry;
         record.shadow_view = draw.shadow_view;
         record.check_rigid = draw.shadow_view && source == MatrixSource::MvpConstantBuffer;
         record.mvp = source == MatrixSource::MvpConstantBuffer;
         record.tag = record_tag;
         local.push_back(record);
         if (identity)
            frame_identities.insert(identity);
         stats.draws_captured++;
         stats.instances_captured += draw.instance_count;
         if (draw.shadow_view)
         {
            stats.shadow_draws++;
            stats.shadow_instances += draw.instance_count;
         }
         if (draw.vs->wind)
         {
            stats.wind_draws++;
            stats.wind_instances += draw.instance_count;
         }
         if (draw.cutout)
         {
            stats.cutout_draws++;
            stats.cutout_instances += draw.instance_count;
            cutouts.CountDraw(material != 0);
         }
      }
      else
      {
         stats.skipped[kSkipOther]++;
      }
      QueueLocked(context, local);
      return dispatched;
   }

   uint64_t GeometryCapture::FuzzyKey(uint64_t mesh_key, const float rows[12], uint64_t exact_key)
   {
      const float t[3] = { rows[3], rows[7], rows[11] };
      int32_t cell[3], lo[3], hi[3];
      for (int a = 0; a < 3; a++)
      {
         cell[a] = int32_t(std::floor(t[a] / kFuzzyCell));
         lo[a] = int32_t(std::floor((t[a] - kFuzzyTranslation) / kFuzzyCell));
         hi[a] = int32_t(std::floor((t[a] + kFuzzyTranslation) / kFuzzyCell));
      }
      auto cell_key = [&](int32_t x, int32_t y, int32_t z) {
         return HashCombine(HashCombine(HashCombine(mesh_key, uint64_t(uint32_t(x))), uint64_t(uint32_t(y))), uint64_t(uint32_t(z)));
      };
      // The cells within the tolerance of the translation (1 to 8)
      for (int32_t x = lo[0]; x <= hi[0]; x++)
         for (int32_t y = lo[1]; y <= hi[1]; y++)
            for (int32_t z = lo[2]; z <= hi[2]; z++)
            {
               auto it = fuzzy_cells.find(cell_key(x, y, z));
               if (it == fuzzy_cells.end())
                  continue;
               for (FuzzyEntry& e : it->second)
               {
                  bool same = true;
                  for (int k = 0; k < 12 && same; k++)
                     same = std::fabs(e.rows[k] - rows[k]) <= ((k % 4) == 3 ? kFuzzyTranslation : kFuzzyRotation * (std::max)(1.f, std::fabs(e.rows[k])));
                  if (!same)
                     continue;
                  fuzzy_matched_other_key += e.key != exact_key;
                  e.last_frame = fuzzy_frame;
                  return e.key; // The stored transform stays the first one seen (the matches are within the tolerance)
               }
            }
      FuzzyEntry entry;
      std::memcpy(entry.rows, rows, sizeof(entry.rows));
      entry.key = exact_key;
      entry.last_frame = fuzzy_frame;
      fuzzy_cells[cell_key(cell[0], cell[1], cell[2])].push_back(entry);
      return exact_key;
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

   bool GeometryCapture::CaptureFrameDraw(ID3D11DeviceContext* context, const DrawInfo& draw, uint64_t key, uint64_t identity, uint64_t index_start, uint32_t index_size, bool dynamic_vertices)
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
         record.shadow_view = draw.shadow_view;
         record.skinned = skinned;
         local.push_back(record);
         if (identity)
            frame_identities.insert(identity);
         stats.draws_captured++;
         stats.instances_captured++;
         stats.shadow_frame_meshes += draw.shadow_view;
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
         else if (record.type == Rec::AlphaStrip)
         {
            cutouts.OnStripExecuted(record.mesh_key, record.first_slot); // Copied at the next D3D12 pass, whenever drawn
         }
         else if (after_ao_pass && !record.shadow_view) // Shadow maps are drawn after the AO, and are opaque casters
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
      frame_identities.clear();
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
               else if (record.type == Rec::AlphaStrip)
               {
                  cutouts.AbandonStrip(record.mesh_key, record.first_slot);
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
      cutouts.OnPresent();
   }

   void GeometryCapture::OnBufferWritten(ID3D11Buffer* buffer, bool forget)
   {
      {
         const std::lock_guard lock(mutex);
         auto it = buffers.find(buffer);
         if (it == buffers.end() || !it->second.executed)
            return;
         stats.written_after_copy++;
         if (written_seen.insert(buffer).second)
            stats.written_buffers++;
         if (!forget)
            return;
         stats.written_forgotten++;
      }
      OnDestroyBuffer(buffer);
   }

   void GeometryCapture::OnDestroyBuffer(ID3D11Buffer* buffer)
   {
      const std::lock_guard lock(mutex);
      written_seen.erase(buffer);
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
      const RetentionStats retention = tlas.GetRetentionStats();
      uint64_t frame_instances = 0, view_model = 0, frame_cutouts = 0, shadow_rejected = 0, mvp_instances = 0, tag_mismatch = 0, tag_checked = 0;
      fuzzy_frame++;
      fuzzy_matched_other_key = 0;
      if ((fuzzy_frame & 255) == 0) // Forget entries the TLAS has dropped too
      {
         for (auto it = fuzzy_cells.begin(); it != fuzzy_cells.end();)
         {
            auto& entries = it->second;
            entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const FuzzyEntry& e) { return fuzzy_frame - e.last_frame > TlasBuilder::kShadowRetainFrames + 16; }), entries.end());
            it = entries.empty() ? fuzzy_cells.erase(it) : std::next(it);
         }
      }
      if (!readback_records[slot].empty())
      {
         uint64_t total = 0;
         for (const auto& r : readback_records[slot])
            total += r.count;
         const uint8_t* data = nullptr;
         D3D12_RANGE range = { 0, SIZE_T(total * 64) };
         if (SUCCEEDED(readback[slot]->Map(0, &range, (void**)&data)) && data)
         {
            uint64_t index = 0;
            for (const auto& r : readback_records[slot])
            {
               // Texture exact cutouts: the draw's uv scale/bias and alpha reference (capture_instances_cs wrote them next to
               // every instance; they come from the draw's cbuffer, so its first instance has them) select its record
               uint32_t cutout_record = kNoCutoutRecord;
               if (r.material && (r.mask & kInstanceMaskCutout) && r.count)
               {
                  uint32_t words[3];
                  std::memcpy(words, data + index * 64 + 52, sizeof(words));
                  const float scalebias[4] = { HalfToFloat(words[0] & 0xFFFF), HalfToFloat(words[0] >> 16), HalfToFloat(words[1] & 0xFFFF), HalfToFloat(words[1] >> 16) };
                  float alpha_ref;
                  std::memcpy(&alpha_ref, &words[2], 4);
                  bool finite = std::isfinite(alpha_ref);
                  for (float v : scalebias)
                     finite = finite && std::isfinite(v);
                  if (finite)
                     cutout_record = cutouts.ResolveRecord(r.material, r.geometry, scalebias, alpha_ref);
               }
               for (uint32_t i = 0; i < r.count; i++, index++)
               {
                  const float* rows = reinterpret_cast<const float*>(data + index * 64);
                  const uint32_t flags = *reinterpret_cast<const uint32_t*>(data + index * 64 + 48);
                  tag_checked++;
                  if ((flags >> 8) != r.tag)
                  {
                     tag_mismatch++; // This entry was written by another draw: never use it
                     continue;
                  }
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
                  if (r.check_rigid && !IsRigidTransform(rows))
                  {
                     shadow_rejected++;
                     continue;
                  }
                  // Identity: mesh + transform quantized to ~1 cm / 1e-3 (instance buffers carry no stable id)
                  uint64_t key = r.mesh_key;
                  for (int k = 0; k < 12; k++)
                  {
                     const bool translation = (k % 4) == 3;
                     key = HashCombine(key, uint64_t(int64_t(std::llround(rows[k] * (translation ? 100.f : 1000.f)))));
                  }
                  if (r.mvp)
                  {
                     key = FuzzyKey(r.mesh_key, rows, key);
                     mvp_instances++;
                  }
                  float transform[3][4];
                  std::memcpy(transform, rows, sizeof(transform));
                  tlas.AddInstance(key, r.mesh_key, transform, r.mask, cutout_record, uint8_t((r.shadow_view ? kInstanceSourceShadow : kInstanceSourceMain) | (r.mvp ? kInstanceFlagMvp : 0)));
                  frame_instances++;
                  frame_cutouts += (r.mask & kInstanceMaskCutout) != 0;
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
            readback_records[slot].push_back({ record.mesh_key, record.count, record.mask, record.material, record.geometry, record.shadow_view, record.check_rigid, record.mvp, record.tag });
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
               // Movable objects (dynamic vertex buffers: openable windows, doors) keep their last pose while the game skips
               // drawing them (DynamicBlasSet::kRetainFrames); characters (skinned) are current-frame only
               const uint8_t source = uint8_t((record.shadow_view ? kInstanceSourceShadow : kInstanceSourceMain) | (record.skinned ? kInstanceFlagSkinned : 0));
               const D3D12_GPU_VIRTUAL_ADDRESS blas = frame_blases.Update(ctx, key, mesh, refit, build_error, !record.skinned, source);
               if (!blas)
               {
                  frame_failed++;
                  continue;
               }
               tlas.AddFrameInstance(blas, identity, kInstanceMaskOpaque, source);
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
         // Movable objects not drawn this frame: their last pose
         frame_blases.ForEachRetained([&](D3D12_GPU_VIRTUAL_ADDRESS blas, uint8_t source) {
            static const float identity[3][4] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
            tlas.AddFrameInstance(blas, identity, kInstanceMaskOpaque, source);
         });
      }

      // 6) Texture exact cutouts: new masks into the atlas, the record table (after this frame's records were resolved above)
      if (!cutouts.Record(list, info, retirer, vertex_arena12, cutout_bindings, error))
         return false;

      const std::lock_guard lock(mutex);
      stats.frame_draws = frame_draws;
      stats.frame_vertices = frame_vertices;
      stats.frame_blas_builds += frame_builds;
      stats.frame_blas_refits += frame_refits;
      stats.frame_blas_failed += frame_failed;
      stats.frame_blas_count = frame_blases.Count();
      stats.frame_retained = frame_blases.RetainedCount();
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
      stats.frame_cutout_instances = frame_cutouts;
      stats.view_model_instances = view_model;
      stats.shadow_rejected += shadow_rejected;
      stats.mvp_instances = mvp_instances;
      stats.tag_mismatch += tag_mismatch;
      stats.tag_checked += tag_checked;
      stats.mvp_rekeyed = fuzzy_matched_other_key;
      stats.retention = retention;
      size_t fuzzy_entries = 0;
      for (const auto& pair : fuzzy_cells)
         fuzzy_entries += pair.second.size();
      stats.fuzzy_entries = fuzzy_entries;
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
