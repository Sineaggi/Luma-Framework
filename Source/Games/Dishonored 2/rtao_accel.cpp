#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rtao_accel.h"

#include <algorithm>
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

      uint64_t Align(uint64_t value, uint64_t alignment)
      {
         return (value + alignment - 1) & ~(alignment - 1);
      }

      ID3D12Resource* CreateBuffer(ID3D12Device* device, uint64_t size, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags, HRESULT& hr)
      {
         D3D12_RESOURCE_DESC desc = {};
         desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
         desc.Width = std::max<uint64_t>(size, 256);
         desc.Height = 1;
         desc.DepthOrArraySize = 1;
         desc.MipLevels = 1;
         desc.SampleDesc.Count = 1;
         desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
         desc.Flags = flags;
         D3D12_HEAP_PROPERTIES heap = {};
         heap.Type = heap_type;
         ID3D12Resource* resource = nullptr;
         hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource));
         return SUCCEEDED(hr) ? resource : nullptr;
      }

      void Release(IUnknown*& object, Retirer* retirer, uint64_t fence_value)
      {
         if (!object)
            return;
         if (retirer)
            retirer->Retire(object, fence_value);
         else
            object->Release();
         object = nullptr;
      }

      template <typename T>
      void Release(T*& object, Retirer* retirer, uint64_t fence_value)
      {
         IUnknown* unknown = object;
         Release(unknown, retirer, fence_value);
         object = nullptr;
      }

      void UavBarrier(ID3D12GraphicsCommandList4* list, ID3D12Resource* resource)
      {
         D3D12_RESOURCE_BARRIER barrier = {};
         barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
         barrier.UAV.pResource = resource; // null: all UAV accesses
         list->ResourceBarrier(1, &barrier);
      }
   }

   // ---- Retirer

   void Retirer::Retire(IUnknown* object, uint64_t fence_value)
   {
      if (object)
         pending.emplace_back(fence_value, object);
   }

   void Retirer::Collect(uint64_t completed_fence_value)
   {
      auto it = std::remove_if(pending.begin(), pending.end(), [&](const auto& p) {
         if (p.first > completed_fence_value)
            return false;
         p.second->Release();
         return true;
      });
      pending.erase(it, pending.end());
   }

   void Retirer::ReleaseAll()
   {
      for (auto& p : pending)
         p.second->Release();
      pending.clear();
   }

   // ---- BlasCache

   D3D12_GPU_VIRTUAL_ADDRESS BlasCache::Get(uint64_t key) const
   {
      auto it = entries.find(key);
      return it != entries.end() ? it->second.result->GetGPUVirtualAddress() : 0;
   }

   bool BlasCache::BuildFromCpu(const BuildContext& ctx, uint64_t key, const CpuMesh& mesh, std::string& error)
   {
      if (Has(key))
         return true;
      if (!mesh.positions || mesh.vertex_count == 0 || mesh.vertex_stride < 12 || (mesh.indices && mesh.index_count < 3))
      {
         error = "invalid mesh";
         return false;
      }
      // One upload buffer: vertices, then indices. Upload heap memory is a valid build input; it is retired with
      // this submission's fence, so it lives exactly until the build has run.
      const uint64_t vertex_bytes = uint64_t(mesh.vertex_count) * mesh.vertex_stride;
      const uint64_t index_offset = Align(vertex_bytes, 16);
      const uint64_t index_bytes = mesh.indices ? uint64_t(mesh.index_count) * 4 : 0;
      HRESULT hr;
      ID3D12Resource* upload = CreateBuffer(ctx.device, index_offset + index_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, hr);
      if (!upload)
      {
         error = HResultString("CreateCommittedResource (BLAS upload)", hr);
         return false;
      }
      uint8_t* mapped = nullptr;
      D3D12_RANGE no_read = { 0, 0 };
      hr = upload->Map(0, &no_read, reinterpret_cast<void**>(&mapped));
      if (FAILED(hr))
      {
         upload->Release();
         error = HResultString("Map (BLAS upload)", hr);
         return false;
      }
      std::memcpy(mapped, mesh.positions, vertex_bytes);
      if (index_bytes)
         std::memcpy(mapped + index_offset, mesh.indices, index_bytes);
      upload->Unmap(0, nullptr);

      D3D12_RAYTRACING_GEOMETRY_DESC geometry = {};
      geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
      geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
      geometry.Triangles.VertexBuffer.StartAddress = upload->GetGPUVirtualAddress();
      geometry.Triangles.VertexBuffer.StrideInBytes = mesh.vertex_stride;
      geometry.Triangles.VertexCount = mesh.vertex_count;
      geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
      if (index_bytes)
      {
         geometry.Triangles.IndexBuffer = upload->GetGPUVirtualAddress() + index_offset;
         geometry.Triangles.IndexCount = mesh.index_count - mesh.index_count % 3;
         geometry.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
      }
      const bool ok = Build(ctx, key, geometry, error);
      ctx.retirer->Retire(upload, ctx.submit_fence_value);
      return ok;
   }

   bool BlasCache::BuildFromGpu(const BuildContext& ctx, uint64_t key, const GpuMesh& mesh, std::string& error)
   {
      if (Has(key))
         return true;
      if (!mesh.vertices || mesh.vertex_count == 0 || (mesh.indices && mesh.index_count < 3))
      {
         error = "invalid mesh";
         return false;
      }
      D3D12_RAYTRACING_GEOMETRY_DESC geometry = {};
      geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
      geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
      geometry.Triangles.VertexBuffer.StartAddress = mesh.vertices;
      geometry.Triangles.VertexBuffer.StrideInBytes = mesh.vertex_stride;
      geometry.Triangles.VertexCount = mesh.vertex_count;
      geometry.Triangles.VertexFormat = mesh.vertex_format;
      if (mesh.indices)
      {
         geometry.Triangles.IndexBuffer = mesh.indices;
         geometry.Triangles.IndexCount = mesh.index_count - mesh.index_count % 3;
         geometry.Triangles.IndexFormat = mesh.index_format;
      }
      return Build(ctx, key, geometry, error);
   }

   bool BlasCache::Build(const BuildContext& ctx, uint64_t key, const D3D12_RAYTRACING_GEOMETRY_DESC& geometry, std::string& error)
   {
      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
      inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
      inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
      inputs.NumDescs = 1;
      inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
      inputs.pGeometryDescs = &geometry;
      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
      ctx.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);
      if (prebuild.ResultDataMaxSizeInBytes == 0)
      {
         error = "BLAS prebuild info returned 0 bytes";
         return false;
      }
      HRESULT hr;
      ID3D12Resource* result = CreateBuffer(ctx.device, Align(prebuild.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
      if (!result)
      {
         error = HResultString("CreateCommittedResource (BLAS)", hr);
         return false;
      }
      ID3D12Resource* scratch = CreateBuffer(ctx.device, Align(prebuild.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
      if (!scratch)
      {
         result->Release();
         error = HResultString("CreateCommittedResource (BLAS scratch)", hr);
         return false;
      }
      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = {};
      build.Inputs = inputs;
      build.DestAccelerationStructureData = result->GetGPUVirtualAddress();
      build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
      ctx.list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
      ctx.retirer->Retire(scratch, ctx.submit_fence_value);

      Entry entry;
      entry.result = result;
      entry.bytes = prebuild.ResultDataMaxSizeInBytes;
      entries.emplace(key, entry);
      result_bytes += entry.bytes;
      barrier_owed = true;
      return true;
   }

   void BlasCache::FlushBarrier(ID3D12GraphicsCommandList4* list)
   {
      if (!barrier_owed)
         return;
      UavBarrier(list, nullptr); // One barrier for the whole batch of builds
      barrier_owed = false;
   }

   void BlasCache::Evict(uint64_t key, Retirer* retirer, uint64_t fence_value)
   {
      auto it = entries.find(key);
      if (it == entries.end())
         return;
      result_bytes -= it->second.bytes;
      Release(it->second.result, retirer, fence_value);
      entries.erase(it);
   }

   void BlasCache::Clear(Retirer* retirer, uint64_t fence_value)
   {
      for (auto& entry : entries)
         Release(entry.second.result, retirer, fence_value);
      entries.clear();
      result_bytes = 0;
      barrier_owed = false;
   }

   // ---- DynamicBlasSet

   void DynamicBlasSet::BeginFrame()
   {
      ++frame;
      scratch_used = 0;
      scratch_peak = 0;
   }

   bool DynamicBlasSet::AllocateScratch(const BuildContext& ctx, uint64_t size, D3D12_GPU_VIRTUAL_ADDRESS& out, std::string& error)
   {
      size = Align(std::max<uint64_t>(size, 256), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
      scratch_peak += size;
      if (scratch && scratch_used + size <= scratch_capacity)
      {
         out = scratch->GetGPUVirtualAddress() + scratch_used;
         scratch_used += size;
         return true;
      }
      // Doesn't fit this frame: a temporary buffer (the persistent one grows in EndFrame)
      HRESULT hr;
      ID3D12Resource* temp = CreateBuffer(ctx.device, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
      if (!temp)
      {
         error = HResultString("CreateCommittedResource (dynamic BLAS scratch)", hr);
         return false;
      }
      out = temp->GetGPUVirtualAddress();
      ctx.retirer->Retire(temp, ctx.submit_fence_value);
      return true;
   }

   D3D12_GPU_VIRTUAL_ADDRESS DynamicBlasSet::Update(const BuildContext& ctx, uint64_t key, const GpuMesh& mesh, bool& refit, std::string& error)
   {
      refit = false;
      if (!mesh.vertices || mesh.vertex_count < 3 || (mesh.indices && mesh.index_count < 3))
      {
         error = "invalid mesh";
         return 0;
      }
      D3D12_RAYTRACING_GEOMETRY_DESC geometry = {};
      geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
      geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
      geometry.Triangles.VertexBuffer.StartAddress = mesh.vertices;
      geometry.Triangles.VertexBuffer.StrideInBytes = mesh.vertex_stride;
      geometry.Triangles.VertexCount = mesh.vertex_count;
      geometry.Triangles.VertexFormat = mesh.vertex_format;
      if (mesh.indices)
      {
         geometry.Triangles.IndexBuffer = mesh.indices;
         geometry.Triangles.IndexCount = mesh.index_count - mesh.index_count % 3;
         geometry.Triangles.IndexFormat = mesh.index_format;
      }
      else
      {
         geometry.Triangles.VertexCount = mesh.vertex_count - mesh.vertex_count % 3;
      }
      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
      inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
      inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
      inputs.NumDescs = 1;
      inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
      inputs.pGeometryDescs = &geometry;

      auto it = entries.find(key);
      if (it != entries.end() && it->second.last_frame == frame)
      {
         error = "dynamic BLAS updated twice in a frame";
         return 0;
      }
      // Same topology: the BLAS can be refit, or rebuilt in its own memory
      const bool same_topology = it != entries.end() && it->second.vertex_count == mesh.vertex_count && it->second.index_count == (mesh.indices ? mesh.index_count : 0) &&
         it->second.indices == mesh.indices && it->second.index_format == (mesh.indices ? mesh.index_format : DXGI_FORMAT_UNKNOWN);

      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = {};
      D3D12_GPU_VIRTUAL_ADDRESS scratch_address = 0;
      if (same_topology && it->second.refits < kMaxRefits)
      {
         if (!AllocateScratch(ctx, it->second.update_scratch, scratch_address, error))
            return 0;
         inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
         build.SourceAccelerationStructureData = it->second.result->GetGPUVirtualAddress();
         it->second.refits++;
         refit = true;
      }
      else
      {
         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
         ctx.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);
         if (prebuild.ResultDataMaxSizeInBytes == 0)
         {
            error = "dynamic BLAS prebuild info returned 0 bytes";
            return 0;
         }
         if (!same_topology)
         {
            if (it != entries.end())
            {
               result_bytes -= it->second.bytes;
               Release(it->second.result, ctx.retirer, ctx.submit_fence_value);
               entries.erase(it);
            }
            HRESULT hr;
            ID3D12Resource* result = CreateBuffer(ctx.device, Align(prebuild.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
            if (!result)
            {
               error = HResultString("CreateCommittedResource (dynamic BLAS)", hr);
               return 0;
            }
            Entry entry;
            entry.result = result;
            entry.bytes = prebuild.ResultDataMaxSizeInBytes;
            entry.update_scratch = prebuild.UpdateScratchDataSizeInBytes;
            entry.vertex_count = mesh.vertex_count;
            entry.index_count = mesh.indices ? mesh.index_count : 0;
            entry.index_format = mesh.indices ? mesh.index_format : DXGI_FORMAT_UNKNOWN;
            entry.indices = mesh.indices;
            it = entries.emplace(key, entry).first;
            result_bytes += entry.bytes;
         }
         if (!AllocateScratch(ctx, prebuild.ScratchDataSizeInBytes, scratch_address, error))
         {
            // Nothing was recorded into this entry: drop it, so it is never refit from an unbuilt BLAS
            result_bytes -= it->second.bytes;
            Release(it->second.result, ctx.retirer, ctx.submit_fence_value);
            entries.erase(it);
            return 0;
         }
         it->second.refits = 0;
      }
      it->second.last_frame = frame;
      build.Inputs = inputs;
      build.DestAccelerationStructureData = it->second.result->GetGPUVirtualAddress();
      build.ScratchAccelerationStructureData = scratch_address;
      ctx.list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
      barrier_owed = true;
      return it->second.result->GetGPUVirtualAddress();
   }

   void DynamicBlasSet::EndFrame(const BuildContext& ctx)
   {
      if (barrier_owed)
      {
         UavBarrier(ctx.list, nullptr);
         barrier_owed = false;
      }
      for (auto it = entries.begin(); it != entries.end();)
      {
         if (frame - it->second.last_frame > kEvictFrames)
         {
            result_bytes -= it->second.bytes;
            Release(it->second.result, ctx.retirer, ctx.submit_fence_value);
            it = entries.erase(it);
         }
         else
         {
            ++it;
         }
      }
      // Grow the persistent scratch to what this frame needed (the frame's builds used temporaries for the rest)
      if (scratch_peak > scratch_capacity)
      {
         Release(scratch, ctx.retirer, ctx.submit_fence_value);
         const uint64_t capacity = Align(scratch_peak + scratch_peak / 4, 65536);
         HRESULT hr;
         scratch = CreateBuffer(ctx.device, capacity, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
         scratch_capacity = scratch ? capacity : 0;
      }
   }

   void DynamicBlasSet::Clear(Retirer* retirer, uint64_t fence_value)
   {
      for (auto& entry : entries)
         Release(entry.second.result, retirer, fence_value);
      entries.clear();
      result_bytes = 0;
      Release(scratch, retirer, fence_value);
      scratch_capacity = scratch_used = scratch_peak = 0;
      barrier_owed = false;
   }

   // ---- TlasBuilder

   void TlasBuilder::BeginFrame()
   {
      const std::lock_guard lock(mutex);
      ++frame;
      frame_instances.clear();
      for (auto it = instances.begin(); it != instances.end();)
      {
         const uint32_t max_age = it->second.seen_frames >= promote_frames ? retain_frames : young_retain_frames;
         if (frame - it->second.last_frame > max_age)
            it = instances.erase(it);
         else
            ++it;
      }
   }

   void TlasBuilder::AddInstance(uint64_t instance_key, uint64_t blas_key, const float object_to_world[3][4], uint8_t mask, uint32_t cutout_record)
   {
      const std::lock_guard lock(mutex);
      auto [it, inserted] = instances.try_emplace(instance_key);
      Instance& instance = it->second;
      if (inserted)
         instance.seen_frames = 1;
      else if (instance.last_frame != frame)
         instance.seen_frames++;
      std::memcpy(instance.transform, object_to_world, sizeof(instance.transform));
      instance.blas_key = blas_key;
      instance.mask = mask;
      instance.cutout_record = cutout_record;
      instance.last_frame = frame; // Re-adding refreshes the transform and the age
   }

   void TlasBuilder::RemoveInstance(uint64_t instance_key)
   {
      const std::lock_guard lock(mutex);
      instances.erase(instance_key);
   }

   void TlasBuilder::AddFrameInstance(D3D12_GPU_VIRTUAL_ADDRESS blas, const float object_to_world[3][4], uint8_t mask)
   {
      if (!blas)
         return;
      D3D12_RAYTRACING_INSTANCE_DESC desc = {};
      std::memcpy(desc.Transform, object_to_world, sizeof(desc.Transform));
      desc.InstanceMask = mask;
      desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE | D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
      desc.AccelerationStructure = blas;
      frame_instances.push_back(desc);
   }

   size_t TlasBuilder::RetainedCount()
   {
      const std::lock_guard lock(mutex);
      return instances.size();
   }

   bool TlasBuilder::Build(const BuildContext& ctx, const BlasCache& blases, uint32_t frame_slot, D3D12_GPU_VIRTUAL_ADDRESS& out, uint32_t& instance_count, std::string& error)
   {
      out = 0;
      instance_count = 0;
      frame_slot %= kSlots;
      last_cutout_count = 0;
      {
         const std::lock_guard lock(mutex);
         descs.clear();
         descs.reserve(instances.size() + frame_instances.size());
         for (const auto& pair : instances)
         {
            const D3D12_GPU_VIRTUAL_ADDRESS blas = blases.Get(pair.second.blas_key);
            if (!blas)
               continue; // Not built (yet)
            D3D12_RAYTRACING_INSTANCE_DESC desc = {};
            std::memcpy(desc.Transform, pair.second.transform, sizeof(desc.Transform));
            desc.InstanceID = uint32_t(descs.size()) & (kInstanceIdCutoutBit - 1);
            desc.InstanceMask = pair.second.mask;
            desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE | D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
            if (pair.second.mask & kInstanceMaskCutout)
            {
               // Same BLAS as an opaque draw of the mesh (built OPAQUE), overridden per instance: the trace's candidate
               // loop decides each hit. Opaque instances stay forced opaque, so they cost nothing extra.
               desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE | D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
               desc.InstanceID = kInstanceIdCutoutBit | (pair.second.cutout_record & kNoCutoutRecord); // The trace reads the record from it
               last_cutout_count++;
            }
            desc.AccelerationStructure = blas;
            descs.push_back(desc);
         }
         for (D3D12_RAYTRACING_INSTANCE_DESC desc : frame_instances)
         {
            desc.InstanceID = uint32_t(descs.size()) & 0xFFFFFF;
            descs.push_back(desc);
         }
      }

      // Instance descs: one upload buffer per frame slot, so the CPU never writes one the GPU may still read
      const uint64_t desc_bytes = std::max<uint64_t>(descs.size(), 1) * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
      HRESULT hr;
      if (instance_upload_capacity[frame_slot] < desc_bytes)
      {
         Release(instance_upload[frame_slot], ctx.retirer, ctx.submit_fence_value);
         const uint64_t capacity = Align(desc_bytes + desc_bytes / 2, 4096);
         instance_upload[frame_slot] = CreateBuffer(ctx.device, capacity, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, hr);
         if (!instance_upload[frame_slot])
         {
            instance_upload_capacity[frame_slot] = 0;
            error = HResultString("CreateCommittedResource (TLAS instances)", hr);
            return false;
         }
         instance_upload_capacity[frame_slot] = capacity;
      }
      if (!descs.empty())
      {
         void* mapped = nullptr;
         D3D12_RANGE no_read = { 0, 0 };
         hr = instance_upload[frame_slot]->Map(0, &no_read, &mapped);
         if (FAILED(hr))
         {
            error = HResultString("Map (TLAS instances)", hr);
            return false;
         }
         std::memcpy(mapped, descs.data(), descs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
         instance_upload[frame_slot]->Unmap(0, nullptr);
      }

      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
      inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
      inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
      inputs.NumDescs = uint32_t(descs.size());
      inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
      inputs.InstanceDescs = descs.empty() ? 0 : instance_upload[frame_slot]->GetGPUVirtualAddress();
      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
      ctx.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);

      // Result and scratch are reused frame to frame (the queue runs submissions in order); grown with headroom
      const uint64_t result_needed = Align(std::max<uint64_t>(prebuild.ResultDataMaxSizeInBytes, 256), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
      if (result_capacity < result_needed)
      {
         Release(result, ctx.retirer, ctx.submit_fence_value);
         const uint64_t capacity = Align(result_needed + result_needed / 2, 65536);
         result = CreateBuffer(ctx.device, capacity, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
         result_capacity = result ? capacity : 0;
         if (!result)
         {
            error = HResultString("CreateCommittedResource (TLAS)", hr);
            return false;
         }
      }
      const uint64_t scratch_needed = Align(std::max<uint64_t>(prebuild.ScratchDataSizeInBytes, 256), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
      if (scratch_capacity < scratch_needed)
      {
         Release(scratch, ctx.retirer, ctx.submit_fence_value);
         const uint64_t capacity = Align(scratch_needed + scratch_needed / 2, 65536);
         scratch = CreateBuffer(ctx.device, capacity, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hr);
         scratch_capacity = scratch ? capacity : 0;
         if (!scratch)
         {
            error = HResultString("CreateCommittedResource (TLAS scratch)", hr);
            return false;
         }
      }

      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = {};
      build.Inputs = inputs;
      build.DestAccelerationStructureData = result->GetGPUVirtualAddress();
      build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
      ctx.list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
      UavBarrier(ctx.list, result);

      out = result->GetGPUVirtualAddress();
      instance_count = uint32_t(descs.size());
      return true;
   }

   void TlasBuilder::Clear(Retirer* retirer, uint64_t fence_value)
   {
      {
         const std::lock_guard lock(mutex);
         instances.clear();
         frame_instances.clear();
      }
      for (uint32_t i = 0; i < kSlots; i++)
      {
         Release(instance_upload[i], retirer, fence_value);
         instance_upload_capacity[i] = 0;
      }
      Release(scratch, retirer, fence_value);
      Release(result, retirer, fence_value);
      scratch_capacity = result_capacity = 0;
   }
}
