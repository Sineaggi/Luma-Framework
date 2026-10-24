#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rtao_alpha.h"

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

      uint64_t HashCombine(uint64_t h, uint64_t v)
      {
         h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
         return h * 0xFF51AFD7ED558CCDull;
      }

      uint64_t MaterialKey(ID3D11Resource* texture, uint32_t channel)
      {
         return HashCombine(uint64_t(uintptr_t(texture)), channel + 1);
      }

      uint32_t FloatBits(float f)
      {
         uint32_t u;
         std::memcpy(&u, &f, 4);
         return u;
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

      // Must match AlphaStripParams in Shaders/Dishonored 2/Luma_RTAO.hlsl
      struct StripParams
      {
         uint32_t channel;
         uint32_t source_mips;
         float source_size[2];
         float glass_density;
         float pad[3];
      };
      static_assert(sizeof(StripParams) == 32);
   }

   bool CutoutMaterials::Init(ID3D11Device* device, Interop& interop, ID3D11ComputeShader* cs, std::string& error)
   {
      const std::lock_guard lock(mutex);
      if (atlas12)
         return true;
      if (!cs)
      {
         error = "alpha strip shader missing";
         return false;
      }
      auto fail = [&](std::string message) {
         error = std::move(message);
         SafeRelease(atlas12);
         SafeRelease(strip11);
         SafeRelease(strip12);
         SafeRelease(tile_uav);
         SafeRelease(tile11);
         SafeRelease(sampler);
         SafeRelease(params_cb);
         return false;
      };
      device11 = device;
      D3D11_BUFFER_DESC cb_desc = { 16, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
      HRESULT hr = device->CreateBuffer(&cb_desc, nullptr, &params_cb);
      if (FAILED(hr))
         return fail(HResultString("CreateBuffer (alpha strip params)", hr));
      D3D11_SAMPLER_DESC sampler_desc = {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
      sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
      hr = device->CreateSamplerState(&sampler_desc, &sampler);
      if (FAILED(hr))
         return fail(HResultString("CreateSamplerState (alpha strip)", hr));
      D3D11_TEXTURE2D_DESC tile_desc = { kTileSize, kTileSize, 1, 1, DXGI_FORMAT_R8_UNORM, { 1, 0 }, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, 0 };
      hr = device->CreateTexture2D(&tile_desc, nullptr, &tile11);
      if (SUCCEEDED(hr))
         hr = device->CreateUnorderedAccessView(tile11, nullptr, &tile_uav);
      if (FAILED(hr))
         return fail(HResultString("alpha tile texture", hr));
      std::string shared_error;
      if (!interop.CreateSharedTexture2D(device, DXGI_FORMAT_R8_UNORM, kTileSize, kTileSize * kStripTiles, strip12, strip11, shared_error))
         return fail("alpha strip (shared R8): " + shared_error);

      D3D12_RESOURCE_DESC desc = {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = kTileSize;
      desc.Height = kTileSize;
      desc.DepthOrArraySize = uint16_t(kAtlasSlices);
      desc.MipLevels = 1;
      desc.Format = DXGI_FORMAT_R8_UNORM;
      desc.SampleDesc.Count = 1;
      D3D12_HEAP_PROPERTIES heap = {};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      hr = interop.GetDevice()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&atlas12));
      if (FAILED(hr))
         return fail(HResultString("CreateCommittedResource (alpha atlas)", hr));
      atlas_written = false;
      strip_cs = cs;
      strip_cs->AddRef();
      return true;
   }

   void CutoutMaterials::Shutdown()
   {
      const std::lock_guard lock(mutex);
      SafeRelease(atlas12);
      SafeRelease(strip11);
      SafeRelease(strip12);
      SafeRelease(tile_uav);
      SafeRelease(tile11);
      SafeRelease(sampler);
      SafeRelease(params_cb);
      SafeRelease(strip_cs);
      for (uint32_t i = 0; i < kSlots; i++)
      {
         SafeRelease(upload[i]);
         upload_capacity[i] = 0;
         upload_version[i] = 0;
      }
      materials.clear();
      executed.clear();
      free_slices.clear();
      next_slice = 0;
      record_ids.clear();
      records.clear();
      free_records.clear();
      strip_head = strip_window = 0;
      version++;
   }

   uint64_t CutoutMaterials::AcquireMaterial(ID3D11ShaderResourceView* srv, uint32_t channel, uint32_t& tile, ID3D11ShaderResourceView* srv2, float param)
   {
      tile = kNoTile;
      if (!srv)
         return 0;
      ID3D11Resource* texture = nullptr;
      srv->GetResource(&texture);
      if (!texture)
         return 0;
      texture->Release(); // Identity only (the SRV keeps it alive for the draw)
      uint64_t key = MaterialKey(texture, channel);
      if (srv2)
      {
         ID3D11Resource* texture2 = nullptr;
         srv2->GetResource(&texture2);
         if (texture2)
            texture2->Release();
         key = HashCombine(HashCombine(key, uint64_t(uintptr_t(texture2))), uint64_t(FloatBits(param)));
      }

      const std::lock_guard lock(mutex);
      if (!atlas12)
         return 0;
      auto it = materials.find(key);
      bool render = false;
      if (it == materials.end())
      {
         D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
         srv->GetDesc(&srv_desc);
         if (srv_desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D)
         {
            // alpha_strip_cs reads a Texture2D: never rendered, stays on the fallback (not retried at every draw)
            Material material;
            material.texture = texture;
            material.state = State::NoSlice;
            materials.emplace(key, material);
            stats.unsupported++;
            return key;
         }
         render = true;
      }
      else if (it->second.state == State::Ready && !it->second.refreshed && frame - it->second.frame > kRefreshFrames)
         render = true; // Once more, later: textures stream their larger mips in after the first sight
      if (!render)
         return key;
      if (strip_window >= kStripBudget)
      {
         stats.strips_deferred++;
         return it == materials.end() ? 0 : key; // A later draw renders it
      }
      tile = strip_head;
      strip_head = (strip_head + 1) % kStripTiles;
      strip_window++;
      if (it == materials.end())
      {
         Material material;
         material.texture = texture;
         material.tile = tile;
         material.frame = frame;
         materials.emplace(key, material);
      }
      else
      {
         // Refresh: stays Ready (its slice keeps the old mask until the copy)
         it->second.tile = tile;
         it->second.refreshed = true;
      }
      return key;
   }

   bool CutoutMaterials::RecordStrip(ID3D11DeviceContext* context, ID3D11ShaderResourceView* srv, uint32_t channel, uint32_t tile, ID3D11ShaderResourceView* srv2, float param)
   {
      if (!strip_cs || tile >= kStripTiles)
         return false;
      // Mip aware: the source is sampled at the mip closest to the tile size (alpha_strip_cs supersamples past the last mip)
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
      srv->GetDesc(&srv_desc);
      if (srv_desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D) // Checked by AcquireMaterial
         return false;
      ID3D11Resource* resource = nullptr;
      srv->GetResource(&resource);
      ID3D11Texture2D* texture = nullptr;
      if (resource)
      {
         resource->QueryInterface(IID_PPV_ARGS(&texture));
         resource->Release();
      }
      if (!texture)
         return false;
      D3D11_TEXTURE2D_DESC texture_desc = {};
      texture->GetDesc(&texture_desc);
      texture->Release();
      const uint32_t most_detailed = srv_desc.Texture2D.MostDetailedMip;
      const uint32_t available = texture_desc.MipLevels > most_detailed ? texture_desc.MipLevels - most_detailed : 1;
      StripParams params = {};
      params.channel = channel;
      params.source_mips = srv_desc.Texture2D.MipLevels == UINT(-1) ? available : (std::min)(uint32_t(srv_desc.Texture2D.MipLevels), available);
      params.source_size[0] = float((std::max)(texture_desc.Width >> most_detailed, 1u));
      params.source_size[1] = float((std::max)(texture_desc.Height >> most_detailed, 1u));
      params.glass_density = param;

      ID3D11ComputeShader* old_cs = nullptr;
      context->CSGetShader(&old_cs, nullptr, nullptr);
      ID3D11Buffer* old_cb = nullptr;
      context->CSGetConstantBuffers(0, 1, &old_cb);
      ID3D11ShaderResourceView* old_srv = nullptr;
      context->CSGetShaderResources(0, 1, &old_srv);
      ID3D11ShaderResourceView* old_srv2 = nullptr;
      context->CSGetShaderResources(1, 1, &old_srv2);
      ID3D11SamplerState* old_sampler = nullptr;
      context->CSGetSamplers(0, 1, &old_sampler);
      ID3D11UnorderedAccessView* old_uav = nullptr;
      context->CSGetUnorderedAccessViews(0, 1, &old_uav);

      bool dispatched = false;
      D3D11_MAPPED_SUBRESOURCE mapped = {};
      if (SUCCEEDED(context->Map(params_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
      {
         std::memcpy(mapped.pData, &params, sizeof(params));
         context->Unmap(params_cb, 0);
         context->CSSetShader(strip_cs, nullptr, 0);
         context->CSSetConstantBuffers(0, 1, &params_cb);
         context->CSSetShaderResources(0, 1, &srv);
         ID3D11ShaderResourceView* second = srv2 ? srv2 : srv;
         context->CSSetShaderResources(1, 1, &second);
         context->CSSetSamplers(0, 1, &sampler);
         UINT initial_count = UINT(-1);
         context->CSSetUnorderedAccessViews(0, 1, &tile_uav, &initial_count);
         context->Dispatch(kTileSize / 8, kTileSize / 8, 1);
         dispatched = true;
      }

      context->CSSetShader(old_cs, nullptr, 0);
      context->CSSetConstantBuffers(0, 1, &old_cb);
      context->CSSetShaderResources(0, 1, &old_srv);
      context->CSSetShaderResources(1, 1, &old_srv2);
      context->CSSetSamplers(0, 1, &old_sampler);
      UINT keep_count = UINT(-1);
      context->CSSetUnorderedAccessViews(0, 1, &old_uav, &keep_count);
      SafeRelease(old_cs);
      SafeRelease(old_cb);
      SafeRelease(old_srv);
      SafeRelease(old_srv2);
      SafeRelease(old_sampler);
      SafeRelease(old_uav);
      if (!dispatched)
         return false;
      // The private tile into the shared strip (a render target only texture: no UAV on it)
      context->CopySubresourceRegion(strip11, 0, 0, tile * kTileSize, 0, tile11, 0, nullptr);
      const std::lock_guard lock(mutex);
      stats.strips_rendered++;
      return true;
   }

   void CutoutMaterials::AbandonStrip(uint64_t material, uint32_t tile)
   {
      const std::lock_guard lock(mutex);
      auto it = materials.find(material);
      if (it == materials.end() || it->second.tile != tile)
         return;
      stats.strips_abandoned++;
      if (it->second.state == State::Pending)
         materials.erase(it);
      else
         it->second.tile = kNoTile;
   }

   void CutoutMaterials::OnStripExecuted(uint64_t material, uint32_t tile)
   {
      const std::lock_guard lock(mutex);
      executed.push_back({ material, tile });
   }

   void CutoutMaterials::OnDestroyTexture(ID3D11Resource* texture)
   {
      const std::lock_guard lock(mutex);
      if (materials.empty())
         return;
      for (uint32_t channel = 0; channel < 4; channel++)
      {
         auto it = materials.find(MaterialKey(texture, channel));
         if (it == materials.end())
            continue;
         if (it->second.slice != 0xFFFF)
            free_slices.push_back(it->second.slice); // Its records resolve to "no mask" at the next upload
         materials.erase(it);
         version++;
      }
   }

   void CutoutMaterials::OnPresent()
   {
      const std::lock_guard lock(mutex);
      frame++;
      for (auto it = materials.begin(); it != materials.end();)
      {
         Material& material = it->second;
         if (material.tile != kNoTile && frame - material.frame > kPendingTimeoutFrames && material.state == State::Pending)
         {
            // Its strip never executed (or got lost): forget it, a later draw renders it again
            stats.strips_abandoned++;
            it = materials.erase(it);
            continue;
         }
         ++it;
      }
   }

   void CutoutMaterials::CountDraw(bool with_material)
   {
      const std::lock_guard lock(mutex);
      (with_material ? stats.draws_exact : stats.draws_no_material)++;
   }

   uint32_t CutoutMaterials::ResolveRecord(uint64_t material, const CutoutGeometry& geometry, const float scalebias[4], float alpha_ref)
   {
      uint64_t key = HashCombine(material, (uint64_t(geometry.index_offset) << 32) | geometry.uv_offset);
      key = HashCombine(key, (uint64_t(geometry.uv_stride) << 32) | (uint64_t(geometry.uv_code) << 1) | geometry.index32);
      key = HashCombine(key, (uint64_t(FloatBits(scalebias[0])) << 32) | FloatBits(scalebias[1]));
      key = HashCombine(key, (uint64_t(FloatBits(scalebias[2])) << 32) | FloatBits(scalebias[3]));
      key = HashCombine(key, FloatBits(alpha_ref));

      const std::lock_guard lock(mutex);
      auto it = record_ids.find(key);
      if (it != record_ids.end())
      {
         records[it->second].last_used = record_frame;
         return it->second;
      }
      uint32_t id = kNoCutoutRecord;
      if (free_records.empty() && records.size() >= kMaxRecords && record_frame - last_sweep > 64)
      {
         // Full: recycle the ids no TLAS instance can still reference
         last_sweep = record_frame;
         for (uint32_t i = 0; i < records.size(); i++)
         {
            RecordEntry& entry = records[i];
            if (entry.key != 0 && record_frame - entry.last_used > kRecordStaleFrames)
            {
               record_ids.erase(entry.key);
               entry.key = 0;
               entry.material = 0;
               free_records.push_back(i);
            }
         }
      }
      if (!free_records.empty())
      {
         id = free_records.back();
         free_records.pop_back();
      }
      else if (records.size() < kMaxRecords)
      {
         id = uint32_t(records.size());
         records.emplace_back();
      }
      if (id == kNoCutoutRecord)
      {
         stats.record_overflow++;
         return kNoCutoutRecord;
      }
      RecordEntry& entry = records[id];
      entry.key = key;
      entry.material = material;
      entry.last_used = record_frame;
      entry.gpu = {};
      entry.gpu.index_offset = geometry.index_offset;
      entry.gpu.uv_offset = geometry.uv_offset;
      entry.gpu.uv_stride_index32 = (geometry.uv_stride & 0x7FFFFFFFu) | (geometry.index32 ? 0x80000000u : 0u);
      entry.gpu.uv_code_slice = (geometry.uv_code & 0xFFFFu) | 0xFFFF0000u;
      std::memcpy(entry.gpu.scalebias, scalebias, sizeof(entry.gpu.scalebias));
      entry.gpu.alpha_ref = alpha_ref;
      record_ids.emplace(key, id);
      version++;
      return id;
   }

   bool CutoutMaterials::Record(ID3D12GraphicsCommandList4* list, const RecordInfo& info, Retirer& retirer, ID3D12Resource* arena, CutoutBindings& out, std::string& error)
   {
      out = {};
      std::vector<ExecutedStrip> work;
      struct Copy
      {
         uint32_t tile, slice;
      };
      std::vector<Copy> copies;
      {
         const std::lock_guard lock(mutex);
         if (!atlas12)
            return true;
         record_frame++;
         strip_window = 0; // Every tile handed out so far was executed (copied below) or will be by a later pass
         work.swap(executed);
         for (const ExecutedStrip& strip : work)
         {
            auto it = materials.find(strip.material);
            if (it == materials.end() || it->second.tile != strip.tile)
               continue; // Destroyed or abandoned meanwhile (its tile may hold another mask now)
            Material& material = it->second;
            material.tile = kNoTile;
            if (material.slice == 0xFFFF)
            {
               if (!free_slices.empty())
               {
                  material.slice = free_slices.back();
                  free_slices.pop_back();
               }
               else if (next_slice < kAtlasSlices)
               {
                  material.slice = next_slice++;
               }
               else
               {
                  material.state = State::NoSlice;
                  stats.atlas_overflow++;
                  continue;
               }
            }
            if (material.state != State::Ready)
            {
               material.state = State::Ready;
               material.frame = frame;
               version++;
            }
            copies.push_back({ strip.tile, material.slice });
         }
         stats.strips_copied += copies.size();
      }

      // Strip tiles -> atlas slices
      if (!copies.empty())
      {
         Transition(list, strip12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
         if (atlas_written)
            Transition(list, atlas12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
         for (const Copy& copy : copies)
         {
            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = atlas12;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = copy.slice; // One mip: subresource = array slice
            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = strip12;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.SubresourceIndex = 0;
            const D3D12_BOX box = { 0, copy.tile * kTileSize, 0, kTileSize, (copy.tile + 1) * kTileSize, 1 };
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
         }
         Transition(list, strip12, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
         Transition(list, atlas12, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
         atlas_written = true;
      }
      if (!atlas_written)
      {
         // The trace samples it (never a slice without a copy, but the state must be right)
         Transition(list, atlas12, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
         atlas_written = true;
      }

      // Record table: one upload buffer per frame slot (this slot's previous use has completed), rewritten whole when
      // anything changed since it was last written
      const uint32_t slot = info.frame_slot % kSlots;
      {
         const std::lock_guard lock(mutex);
         const uint64_t count = (std::max<uint64_t>)(records.size(), 1);
         const uint64_t bytes = count * sizeof(CutoutGpuRecord);
         if (upload_capacity[slot] < bytes)
         {
            if (upload[slot])
            {
               retirer.Retire(upload[slot], info.submit_fence_value);
               upload[slot] = nullptr;
            }
            const uint64_t capacity = (std::min<uint64_t>)((std::max<uint64_t>)(bytes * 2, 64 * sizeof(CutoutGpuRecord)), uint64_t(kMaxRecords) * sizeof(CutoutGpuRecord));
            D3D12_RESOURCE_DESC desc = {};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = (std::max)(capacity, bytes);
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            D3D12_HEAP_PROPERTIES heap = {};
            heap.Type = D3D12_HEAP_TYPE_UPLOAD;
            HRESULT hr = info.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload[slot]));
            if (FAILED(hr))
            {
               upload_capacity[slot] = 0;
               error = HResultString("CreateCommittedResource (cutout records)", hr);
               return false;
            }
            upload_capacity[slot] = desc.Width;
            upload_version[slot] = 0;
         }
         if (upload_version[slot] != version)
         {
            staging.resize(count);
            std::memset(staging.data(), 0, size_t(bytes));
            for (size_t i = 0; i < records.size(); i++)
            {
               const RecordEntry& entry = records[i];
               if (entry.key == 0)
                  continue;
               CutoutGpuRecord gpu = entry.gpu;
               auto material = materials.find(entry.material);
               const uint32_t slice = (material != materials.end() && material->second.state == State::Ready) ? material->second.slice : 0xFFFFu;
               gpu.uv_code_slice = (gpu.uv_code_slice & 0xFFFFu) | (slice << 16);
               staging[i] = gpu;
            }
            if (records.empty())
               staging[0].uv_code_slice = 0xFFFF0000u;
            void* mapped = nullptr;
            D3D12_RANGE no_read = { 0, 0 };
            if (FAILED(upload[slot]->Map(0, &no_read, &mapped)) || !mapped)
            {
               error = "Map (cutout records) failed";
               return false;
            }
            std::memcpy(mapped, staging.data(), size_t(bytes));
            upload[slot]->Unmap(0, nullptr);
            upload_version[slot] = version;
            stats.record_uploads++;
         }
      }

      out.exact = true;
      out.records = upload[slot]->GetGPUVirtualAddress();
      out.arena = arena;
      out.atlas = atlas12;
      out.atlas_slices = kAtlasSlices;
      return true;
   }

   CutoutStats CutoutMaterials::GetStats()
   {
      const std::lock_guard lock(mutex);
      CutoutStats s = stats;
      s.materials = materials.size();
      s.materials_ready = 0;
      s.materials_pending = 0;
      for (const auto& pair : materials)
      {
         s.materials_ready += pair.second.state == State::Ready;
         s.materials_pending += pair.second.state == State::Pending;
      }
      s.atlas_slices_used = next_slice - free_slices.size();
      s.records = records.size() - free_records.size();
      return s;
   }
}
