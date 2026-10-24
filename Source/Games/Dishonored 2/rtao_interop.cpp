#include "rtao_interop.h"

#include <d3dcompiler.h>
#include <cstdio>
#include <cstring>

// d3d12.dll, dxgi.dll and D3DCompiler_47.dll are loaded at runtime so the addon still loads (with RTAO
// unavailable) on systems without them, and so the project needs no extra import libraries.

namespace RTAO
{
   namespace
   {
      template <typename T>
      void SafeRelease(T*& p)
      {
         if (p)
         {
            p->Release();
            p = nullptr;
         }
      }

      std::string HResultString(const char* what, HRESULT hr)
      {
         char buffer[160];
         snprintf(buffer, sizeof(buffer), "%s failed (0x%08lX)", what, (unsigned long)hr);
         return buffer;
      }

      using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
      using PFN_D3DCompile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

      // Test kernel. Writes one AO value per pixel of the game's SSAO render target.
      // InvProj has the same bytes (and the same default column major packing) as "cb_inverseprojectionmatrix"
      // in the game's PerViewCB, so the view depth formula is the one Luma_XeGTAO.hlsl uses.
      const char kComputeShader[] = R"(
cbuffer RootConstants : register(b0)
{
   float4x4 InvProj;
   uint2 AoSize;
   uint2 DepthSize;
   uint DepthPacked;
   uint DebugMode;
   float Strength;
   uint Frame;
};

Texture2D<float4> DepthTex : register(t0);
RWTexture2D<unorm float> AoOut : register(u0);

float LoadDeviceDepth(int2 p)
{
   p = clamp(p, int2(0, 0), int2(DepthSize) - 1);
   float4 v = DepthTex.Load(int3(p, 0));
   uint4 b = uint4(round(saturate(v) * 255.0));
   float unpacked = asfloat(b.x | (b.y << 8) | (b.z << 16) | (b.w << 24));
   return DepthPacked != 0 ? unpacked : v.x;
}

float ViewDepth(float device_depth)
{
   return abs(-InvProj._m23 / (InvProj._m32 * device_depth + InvProj._m33));
}

int2 AoToDepth(int2 p)
{
   return int2((float2(p) + 0.5) * float2(DepthSize) / float2(AoSize));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
   if (any(id.xy >= AoSize))
      return;

   if (DebugMode == 1)
   {
      // Checkerboard plus a bar that sweeps across the screen once per 256 frames:
      // proves the D3D12 result reaches the game, every frame.
      uint2 cell = id.xy / 32;
      float v = ((cell.x + cell.y) & 1) ? 1.0 : 0.35;
      uint bar = (Frame * (AoSize.x / 256 + 1)) % AoSize.x;
      if (id.x >= bar && id.x < bar + 16)
         v = 0.0;
      AoOut[id.xy] = v;
      return;
   }

   // Depth only crease darkening. Not real AO, just enough to show the depth crossed over correctly.
   int2 dp = AoToDepth(int2(id.xy));
   float center = ViewDepth(LoadDeviceDepth(dp));
   if (!(center > 0.0) || center > 10000.0) // sky / invalid
   {
      AoOut[id.xy] = 1.0;
      return;
   }
   static const int2 offsets[8] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, -1), int2(1, -1), int2(-1, 1) };
   float occlusion = 0.0;
   [unroll]
   for (int r = 1; r <= 2; r++)
   {
      [unroll]
      for (int i = 0; i < 8; i++)
      {
         float neighbour = ViewDepth(LoadDeviceDepth(dp + offsets[i] * (r * 6)));
         float delta = center - neighbour; // > 0: the neighbour is in front of us
         occlusion += saturate(delta / (center * 0.05)) * saturate(1.0 - delta / (center * 0.5));
      }
   }
   AoOut[id.xy] = saturate(1.0 - Strength * occlusion / 16.0);
}
)";

      bool CreateSharedTexture(ID3D12Device* device12, ID3D11Device* device11, DXGI_FORMAT format, uint32_t width, uint32_t height, ID3D12Resource*& out12, ID3D11Texture2D*& out11, std::string& error)
      {
         out12 = nullptr;
         out11 = nullptr;
         D3D12_RESOURCE_DESC desc = {};
         desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
         desc.Width = width;
         desc.Height = height;
         desc.DepthOrArraySize = 1;
         desc.MipLevels = 1;
         desc.Format = format;
         desc.SampleDesc.Count = 1;
         desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET; // The only flag combination D3D11 can open (see header)
         D3D12_HEAP_PROPERTIES heap = {};
         heap.Type = D3D12_HEAP_TYPE_DEFAULT;
         HRESULT hr = device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out12));
         if (FAILED(hr))
         {
            error = HResultString("CreateCommittedResource (shared)", hr);
            return false;
         }
         HANDLE handle = nullptr;
         hr = device12->CreateSharedHandle(out12, nullptr, GENERIC_ALL, nullptr, &handle);
         if (FAILED(hr))
         {
            error = HResultString("CreateSharedHandle (texture)", hr);
            SafeRelease(out12);
            return false;
         }
         ID3D11Device1* device11_1 = nullptr;
         hr = device11->QueryInterface(IID_PPV_ARGS(&device11_1));
         if (SUCCEEDED(hr))
         {
            hr = device11_1->OpenSharedResource1(handle, IID_PPV_ARGS(&out11));
            device11_1->Release();
         }
         CloseHandle(handle);
         if (FAILED(hr) || !out11)
         {
            error = HResultString("OpenSharedResource1", hr);
            SafeRelease(out11);
            SafeRelease(out12);
            return false;
         }
         return true;
      }

      void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after, D3D12_RESOURCE_BARRIER& out)
      {
         out = {};
         out.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
         out.Transition.pResource = resource;
         out.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
         out.Transition.StateBefore = before;
         out.Transition.StateAfter = after;
      }
   }

   std::string Interop::DeviceRemovedReport()
   {
      if (!device12)
         return std::string();
      const HRESULT reason = device12->GetDeviceRemovedReason();
      if (reason == S_OK)
         return std::string();
      std::string out = "D3D12 device removed: " + HResultString("reason", reason);
      ID3D12DeviceRemovedExtendedData* dred = nullptr;
      if (FAILED(device12->QueryInterface(IID_PPV_ARGS(&dred))) || !dred)
         return out + " (no DRED)";
      D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs = {};
      if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs)))
      {
         for (const D3D12_AUTO_BREADCRUMB_NODE* node = breadcrumbs.pHeadAutoBreadcrumbNode; node; node = node->pNext)
         {
            const uint32_t count = node->BreadcrumbCount, done = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
            if (done >= count)
               continue; // Completed list
            char line[256];
            snprintf(line, sizeof(line), " | list '%ls' on queue '%ls': %u of %u ops done;", node->pCommandListDebugNameW ? node->pCommandListDebugNameW : L"?",
               node->pCommandQueueDebugNameW ? node->pCommandQueueDebugNameW : L"?", done, count);
            out += line;
            for (uint32_t i = (done > 3 ? done - 3 : 0); i < count && i < done + 4; i++)
            {
               snprintf(line, sizeof(line), " %s%d", i == done ? "[next>]" : "", int(node->pCommandHistory[i]));
               out += line;
            }
         }
      }
      D3D12_DRED_PAGE_FAULT_OUTPUT fault = {};
      if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&fault)) && fault.PageFaultVA)
      {
         char line[128];
         snprintf(line, sizeof(line), " | page fault at 0x%llX, allocations there:", (unsigned long long)fault.PageFaultVA);
         out += line;
         for (const D3D12_DRED_ALLOCATION_NODE* n = fault.pHeadExistingAllocationNode; n; n = n->pNext)
         {
            snprintf(line, sizeof(line), " live '%ls'", n->ObjectNameW ? n->ObjectNameW : L"?");
            out += line;
         }
         for (const D3D12_DRED_ALLOCATION_NODE* n = fault.pHeadRecentFreedAllocationNode; n; n = n->pNext)
         {
            snprintf(line, sizeof(line), " FREED '%ls'", n->ObjectNameW ? n->ObjectNameW : L"?");
            out += line;
         }
      }
      dred->Release();
      return out;
   }

   bool Interop::Init(ID3D11Device* device11, std::string& error)
   {
      const std::lock_guard lock(mutex);
      if (device12)
         return true;

      // Match the game's adapter by LUID
      IDXGIDevice* dxgi_device = nullptr;
      HRESULT hr = device11->QueryInterface(IID_PPV_ARGS(&dxgi_device));
      if (FAILED(hr))
      {
         error = HResultString("QueryInterface(IDXGIDevice)", hr);
         return false;
      }
      IDXGIAdapter* game_adapter = nullptr;
      hr = dxgi_device->GetAdapter(&game_adapter);
      dxgi_device->Release();
      if (FAILED(hr))
      {
         error = HResultString("IDXGIDevice::GetAdapter", hr);
         return false;
      }
      DXGI_ADAPTER_DESC game_adapter_desc = {};
      game_adapter->GetDesc(&game_adapter_desc);
      game_adapter->Release();
      {
         char name[128] = {};
         WideCharToMultiByte(CP_UTF8, 0, game_adapter_desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
         adapter_name = name;
      }

      HMODULE dxgi_module = LoadLibraryW(L"dxgi.dll");
      HMODULE d3d12_module = LoadLibraryW(L"d3d12.dll");
      if (!dxgi_module || !d3d12_module)
      {
         error = "d3d12.dll or dxgi.dll not available";
         return false;
      }
      auto create_factory = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(dxgi_module, "CreateDXGIFactory1"));
      auto create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(d3d12_module, "D3D12CreateDevice"));
      if (!create_factory || !create_device)
      {
         error = "CreateDXGIFactory1 or D3D12CreateDevice not found";
         return false;
      }
      IDXGIFactory4* factory = nullptr;
      hr = create_factory(IID_PPV_ARGS(&factory));
      if (FAILED(hr))
      {
         error = HResultString("CreateDXGIFactory1(IDXGIFactory4)", hr);
         return false;
      }
      IDXGIAdapter1* adapter = nullptr;
      hr = factory->EnumAdapterByLuid(game_adapter_desc.AdapterLuid, IID_PPV_ARGS(&adapter));
      factory->Release();
      if (FAILED(hr))
      {
         error = HResultString("EnumAdapterByLuid", hr);
         return false;
      }
      // Device Removed Extended Data: if the GPU faults, which of our operations was running and which allocation the
      // faulting address was in (DeviceRemovedReport). Must be set before the device is created; cheap.
      if (auto get_debug = reinterpret_cast<PFN_D3D12_GET_DEBUG_INTERFACE>(GetProcAddress(d3d12_module, "D3D12GetDebugInterface")))
      {
         ID3D12DeviceRemovedExtendedDataSettings* dred = nullptr;
         if (SUCCEEDED(get_debug(IID_PPV_ARGS(&dred))) && dred)
         {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->Release();
         }
      }
      hr = create_device(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device12));
      adapter->Release();
      if (FAILED(hr) || !device12)
      {
         device12 = nullptr;
         error = HResultString("D3D12CreateDevice(ID3D12Device5, FL 12_0)", hr);
         return false;
      }

      auto fail = [&](std::string message) {
         error = std::move(message);
         ShutdownLocked();
         return false;
      };

      D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
      if (FAILED(device12->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))) || options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1)
      {
         char buffer[96];
         snprintf(buffer, sizeof(buffer), "DXR 1.1 not supported (raytracing tier %d)", (int)options5.RaytracingTier);
         return fail(buffer);
      }

      D3D12_COMMAND_QUEUE_DESC queue_desc = {};
      queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      hr = device12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
      if (FAILED(hr))
         return fail(HResultString("CreateCommandQueue", hr));
      queue->SetName(L"RTAO queue");
      for (auto& allocator : allocators)
      {
         hr = device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
         if (FAILED(hr))
            return fail(HResultString("CreateCommandAllocator", hr));
      }
      hr = device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0], nullptr, IID_PPV_ARGS(&command_list)); // ID3D12GraphicsCommandList4: DXR builds
      if (FAILED(hr))
         return fail(HResultString("CreateCommandList", hr));
      command_list->SetName(L"RTAO list");
      command_list->Close();

      hr = device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&shared_fence12));
      if (FAILED(hr))
         return fail(HResultString("CreateFence (shared)", hr));
      HANDLE fence_handle = nullptr;
      hr = device12->CreateSharedHandle(shared_fence12, nullptr, GENERIC_ALL, nullptr, &fence_handle);
      if (FAILED(hr))
         return fail(HResultString("CreateSharedHandle (fence)", hr));
      ID3D11Device5* device11_5 = nullptr;
      hr = device11->QueryInterface(IID_PPV_ARGS(&device11_5));
      if (SUCCEEDED(hr))
      {
         hr = device11_5->OpenSharedFence(fence_handle, IID_PPV_ARGS(&shared_fence11));
         device11_5->Release();
      }
      CloseHandle(fence_handle);
      if (FAILED(hr) || !shared_fence11)
         return fail(HResultString("ID3D11Device5::OpenSharedFence", hr));
      shared_fence_value = 0;

      hr = device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&own_fence));
      if (FAILED(hr))
         return fail(HResultString("CreateFence", hr));
      own_fence_value = 0;
      own_fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      if (!own_fence_event)
         return fail("CreateEvent failed");

      D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
      heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      heap_desc.NumDescriptors = 2;
      heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
      hr = device12->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptor_heap));
      if (FAILED(hr))
         return fail(HResultString("CreateDescriptorHeap", hr));
      descriptor_size = device12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

      std::string pipeline_error;
      if (!CreatePipeline(pipeline_error))
         return fail(pipeline_error);

      return true;
   }

   bool Interop::CreatePipeline(std::string& error)
   {
      HMODULE d3d12_module = GetModuleHandleW(L"d3d12.dll");
      auto serialize_root_signature = d3d12_module ? reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(GetProcAddress(d3d12_module, "D3D12SerializeRootSignature")) : nullptr;
      if (!serialize_root_signature)
      {
         error = "D3D12SerializeRootSignature not found";
         return false;
      }
      HMODULE compiler_module = LoadLibraryW(L"D3DCompiler_47.dll");
      auto compile = compiler_module ? reinterpret_cast<PFN_D3DCompile>(GetProcAddress(compiler_module, "D3DCompile")) : nullptr;
      if (!compile)
      {
         error = "D3DCompiler_47.dll not available";
         return false;
      }

      // Root signature: [0] root constants b0, [1] table { SRV t0, UAV u0 }
      D3D12_DESCRIPTOR_RANGE ranges[2] = {};
      ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
      ranges[0].NumDescriptors = 1;
      ranges[0].OffsetInDescriptorsFromTableStart = 0;
      ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
      ranges[1].NumDescriptors = 1;
      ranges[1].OffsetInDescriptorsFromTableStart = 1;
      D3D12_ROOT_PARAMETER parameters[2] = {};
      parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      parameters[0].Constants.Num32BitValues = sizeof(Constants) / 4;
      parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      parameters[1].DescriptorTable.NumDescriptorRanges = 2;
      parameters[1].DescriptorTable.pDescriptorRanges = ranges;
      parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      D3D12_ROOT_SIGNATURE_DESC root_desc = {};
      root_desc.NumParameters = 2;
      root_desc.pParameters = parameters;
      ID3DBlob* root_blob = nullptr;
      ID3DBlob* root_error = nullptr;
      HRESULT hr = serialize_root_signature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_error);
      if (FAILED(hr))
      {
         error = HResultString("D3D12SerializeRootSignature", hr);
         if (root_error)
            error += std::string(": ") + (const char*)root_error->GetBufferPointer();
         SafeRelease(root_error);
         return false;
      }
      SafeRelease(root_error);
      hr = device12->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&root_signature));
      root_blob->Release();
      if (FAILED(hr))
      {
         error = HResultString("CreateRootSignature", hr);
         return false;
      }

      ID3DBlob* code = nullptr;
      ID3DBlob* compile_errors = nullptr;
      hr = compile(kComputeShader, sizeof(kComputeShader) - 1, "RTAO_M0_CS", nullptr, nullptr, "main", "cs_5_1", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &compile_errors);
      if (FAILED(hr))
      {
         error = HResultString("D3DCompile (RTAO test CS)", hr);
         if (compile_errors)
            error += std::string(": ") + (const char*)compile_errors->GetBufferPointer();
         SafeRelease(compile_errors);
         return false;
      }
      SafeRelease(compile_errors);
      D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
      pso_desc.pRootSignature = root_signature;
      pso_desc.CS.pShaderBytecode = code->GetBufferPointer();
      pso_desc.CS.BytecodeLength = code->GetBufferSize();
      hr = device12->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline_state));
      code->Release();
      if (FAILED(hr))
      {
         error = HResultString("CreateComputePipelineState", hr);
         return false;
      }
      return true;
   }

   bool Interop::WaitForGPUIdle(uint32_t timeout_ms)
   {
      if (!queue || !own_fence)
         return true;
      const uint64_t value = ++own_fence_value;
      if (FAILED(queue->Signal(own_fence, value)))
         return false;
      if (own_fence->GetCompletedValue() >= value)
         return true;
      if (FAILED(own_fence->SetEventOnCompletion(value, own_fence_event)))
         return false;
      return WaitForSingleObject(own_fence_event, timeout_ms) == WAIT_OBJECT_0;
   }

   void Interop::ReleaseTargets()
   {
      SafeRelease(depth_rtv11);
      SafeRelease(depth_shared11);
      SafeRelease(depth_shared12);
      SafeRelease(ao_srv11);
      SafeRelease(ao_shared11);
      SafeRelease(ao_shared12);
      SafeRelease(ao_private12);
      depth_width = depth_height = ao_width = ao_height = 0;
      depth_format = ao_format = DXGI_FORMAT_UNKNOWN;
      depth_packed = false;
   }

   void Interop::Shutdown()
   {
      const std::lock_guard lock(mutex);
      ShutdownLocked();
   }

   void Interop::ShutdownLocked()
   {
      WaitForGPUIdle(2000);
      ReleaseTargets();
      SafeRelease(pipeline_state);
      SafeRelease(root_signature);
      SafeRelease(descriptor_heap);
      SafeRelease(command_list);
      for (auto& allocator : allocators)
         SafeRelease(allocator);
      for (auto& value : allocator_fence_values)
         value = 0;
      SafeRelease(own_fence);
      if (own_fence_event)
      {
         CloseHandle(own_fence_event);
         own_fence_event = nullptr;
      }
      SafeRelease(shared_fence11);
      SafeRelease(shared_fence12);
      SafeRelease(queue);
      SafeRelease(device12);
   }

   bool Interop::EnsureTargets(ID3D11Device* device11, uint32_t new_depth_width, uint32_t new_depth_height, uint32_t new_ao_width, uint32_t new_ao_height, std::string& error)
   {
      const std::lock_guard lock(mutex);
      if (!device12)
      {
         error = "not initialized";
         return false;
      }
      if (depth_rtv11 && ao_srv11 && new_depth_width == depth_width && new_depth_height == depth_height && new_ao_width == ao_width && new_ao_height == ao_height)
         return true;
      if (new_depth_width == 0 || new_depth_height == 0 || new_ao_width == 0 || new_ao_height == 0)
      {
         error = "zero sized target";
         return false;
      }

      // The old textures might still be read by in-flight D3D12 work
      WaitForGPUIdle(2000);
      ReleaseTargets();

      // Depth copy: R32_FLOAT if this driver can share it, otherwise the bits packed into R8G8B8A8_UNORM
      std::string depth_error;
      if (CreateSharedTexture(device12, device11, DXGI_FORMAT_R32_FLOAT, new_depth_width, new_depth_height, depth_shared12, depth_shared11, depth_error))
      {
         depth_format = DXGI_FORMAT_R32_FLOAT;
         depth_packed = false;
      }
      else if (CreateSharedTexture(device12, device11, DXGI_FORMAT_R8G8B8A8_UNORM, new_depth_width, new_depth_height, depth_shared12, depth_shared11, error))
      {
         depth_format = DXGI_FORMAT_R8G8B8A8_UNORM;
         depth_packed = true;
      }
      else
      {
         error = "shared depth: R32_FLOAT: " + depth_error + ", R8G8B8A8_UNORM: " + error;
         ReleaseTargets();
         return false;
      }

      // AO: R16_UNORM, R8_UNORM as fallback
      std::string ao_error;
      if (CreateSharedTexture(device12, device11, DXGI_FORMAT_R16_UNORM, new_ao_width, new_ao_height, ao_shared12, ao_shared11, ao_error))
      {
         ao_format = DXGI_FORMAT_R16_UNORM;
      }
      else if (CreateSharedTexture(device12, device11, DXGI_FORMAT_R8_UNORM, new_ao_width, new_ao_height, ao_shared12, ao_shared11, error))
      {
         ao_format = DXGI_FORMAT_R8_UNORM;
      }
      else
      {
         error = "shared AO: R16_UNORM: " + ao_error + ", R8_UNORM: " + error;
         ReleaseTargets();
         return false;
      }

      // Private UAV the compute shader writes (same format, so it can be copied into the shared one)
      D3D12_RESOURCE_DESC desc = {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = new_ao_width;
      desc.Height = new_ao_height;
      desc.DepthOrArraySize = 1;
      desc.MipLevels = 1;
      desc.Format = ao_format;
      desc.SampleDesc.Count = 1;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      D3D12_HEAP_PROPERTIES heap = {};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      HRESULT hr = device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&ao_private12));
      if (FAILED(hr))
      {
         error = HResultString("CreateCommittedResource (AO UAV)", hr);
         ReleaseTargets();
         return false;
      }

      // D3D11 views
      hr = device11->CreateRenderTargetView(depth_shared11, nullptr, &depth_rtv11);
      if (FAILED(hr))
      {
         error = HResultString("CreateRenderTargetView (shared depth)", hr);
         ReleaseTargets();
         return false;
      }
      hr = device11->CreateShaderResourceView(ao_shared11, nullptr, &ao_srv11);
      if (FAILED(hr))
      {
         error = HResultString("CreateShaderResourceView (shared AO)", hr);
         ReleaseTargets();
         return false;
      }

      // D3D12 descriptors
      D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
      D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
      srv_desc.Format = depth_format;
      srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      srv_desc.Texture2D.MipLevels = 1;
      device12->CreateShaderResourceView(depth_shared12, &srv_desc, cpu_handle);
      cpu_handle.ptr += descriptor_size;
      D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
      uav_desc.Format = ao_format;
      uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      device12->CreateUnorderedAccessView(ao_private12, nullptr, &uav_desc, cpu_handle);

      targets_generation++;
      depth_width = new_depth_width;
      depth_height = new_depth_height;
      ao_width = new_ao_width;
      ao_height = new_ao_height;
      return true;
   }

   ID3D11RenderTargetView* Interop::GetDepthRTV()
   {
      const std::lock_guard lock(mutex);
      if (depth_rtv11)
         depth_rtv11->AddRef();
      return depth_rtv11;
   }

   ID3D11ShaderResourceView* Interop::GetAOSRV()
   {
      const std::lock_guard lock(mutex);
      if (ao_srv11)
         ao_srv11->AddRef();
      return ao_srv11;
   }

   bool Interop::Run(ID3D11DeviceContext* immediate_context, Constants constants, std::string& error)
   {
      // The M0 test kernel, as a recorder (runs under the interop lock, so members are safe to read)
      return Run(immediate_context, [this, constants](ID3D12GraphicsCommandList4* list, const RecordInfo& info, std::string&) mutable {
         constants.ao_width = info.ao_width;
         constants.ao_height = info.ao_height;
         constants.depth_width = info.depth_width;
         constants.depth_height = info.depth_height;
         constants.depth_packed = info.depth_packed ? 1 : 0;
         list->SetPipelineState(pipeline_state);
         list->SetComputeRootSignature(root_signature);
         list->SetDescriptorHeaps(1, &descriptor_heap);
         list->SetComputeRoot32BitConstants(0, sizeof(Constants) / 4, &constants, 0);
         list->SetComputeRootDescriptorTable(1, descriptor_heap->GetGPUDescriptorHandleForHeapStart());
         list->Dispatch((info.ao_width + 7) / 8, (info.ao_height + 7) / 8, 1);
         return true;
      }, error);
   }

   bool Interop::Run(ID3D11DeviceContext* immediate_context, const Recorder& recorder, std::string& error)
   {
      const std::lock_guard lock(mutex);
      if (!device12 || !ao_private12 || !depth_shared12)
      {
         error = "not ready";
         return false;
      }
      if (immediate_context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
      {
         error = "Run() needs the immediate context";
         return false;
      }
      ID3D11DeviceContext4* context4 = nullptr;
      HRESULT hr = immediate_context->QueryInterface(IID_PPV_ARGS(&context4));
      if (FAILED(hr))
      {
         error = HResultString("QueryInterface(ID3D11DeviceContext4)", hr);
         return false;
      }

      // Recycle the oldest allocator (3 deep, so normally this never waits)
      const uint32_t index = allocator_index;
      if (own_fence->GetCompletedValue() < allocator_fence_values[index])
      {
         own_fence->SetEventOnCompletion(allocator_fence_values[index], own_fence_event);
         if (WaitForSingleObject(own_fence_event, 2000) != WAIT_OBJECT_0)
         {
            context4->Release();
            error = "timed out waiting for a D3D12 command allocator";
            return false;
         }
      }
      allocator_index = (allocator_index + 1) % frames_in_flight;
      ID3D12CommandAllocator* allocator = allocators[index];
      allocator->Reset();
      command_list->Reset(allocator, nullptr);

      D3D12_RESOURCE_BARRIER barriers[2];
      Transition(command_list, depth_shared12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, barriers[0]);
      command_list->ResourceBarrier(1, barriers);

      RecordInfo info = {};
      info.device = device12;
      info.depth = depth_shared12;
      info.depth_format = depth_format;
      info.depth_packed = depth_packed;
      info.depth_width = depth_width;
      info.depth_height = depth_height;
      info.ao = ao_private12;
      info.ao_format = ao_format;
      info.ao_width = ao_width;
      info.ao_height = ao_height;
      info.targets_generation = targets_generation;
      info.frame_slot = index;
      info.submit_fence_value = own_fence_value + 1;
      info.completed_fence_value = own_fence->GetCompletedValue();
      if (!recorder(command_list, info, error))
      {
         command_list->Close(); // Discarded, never executed
         context4->Release();
         if (error.empty())
            error = "recorder failed";
         return false;
      }

      Transition(command_list, ao_private12, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, barriers[0]);
      Transition(command_list, ao_shared12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, barriers[1]);
      command_list->ResourceBarrier(2, barriers);
      command_list->CopyResource(ao_shared12, ao_private12);
      // Shared resources go back to COMMON before the other API touches them
      Transition(command_list, ao_private12, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, barriers[0]);
      Transition(command_list, ao_shared12, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON, barriers[1]);
      command_list->ResourceBarrier(2, barriers);
      Transition(command_list, depth_shared12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON, barriers[0]);
      command_list->ResourceBarrier(1, barriers);
      hr = command_list->Close();
      if (FAILED(hr))
      {
         context4->Release();
         error = HResultString("ID3D12GraphicsCommandList::Close", hr);
         return false;
      }

      // D3D11 -> D3D12: the depth copy queued before this call must land first
      const uint64_t depth_ready_value = ++shared_fence_value;
      hr = context4->Signal(shared_fence11, depth_ready_value);
      if (FAILED(hr))
      {
         context4->Release();
         error = HResultString("ID3D11DeviceContext4::Signal", hr);
         return false;
      }
      immediate_context->Flush(); // Submit now, otherwise the D3D12 queue would sit waiting until the game's next flush
      queue->Wait(shared_fence12, depth_ready_value);

      ID3D12CommandList* lists[] = { command_list };
      queue->ExecuteCommandLists(1, lists);

      // D3D12 -> D3D11
      const uint64_t ao_ready_value = ++shared_fence_value;
      queue->Signal(shared_fence12, ao_ready_value);
      allocator_fence_values[index] = ++own_fence_value; // == info.submit_fence_value
      queue->Signal(own_fence, own_fence_value);
      hr = context4->Wait(shared_fence11, ao_ready_value);
      context4->Release();
      if (FAILED(hr))
      {
         error = HResultString("ID3D11DeviceContext4::Wait", hr);
         return false;
      }
      return true;
   }

   bool Interop::CreateSharedBuffer(ID3D11Device* device11, uint64_t size, ID3D12Resource*& out12, ID3D11Buffer*& out11, std::string& error)
   {
      // Measured on an RTX 4090 (Tests/rtao_interop_test.cpp history): a D3D12 buffer created with
      // D3D12_HEAP_FLAG_SHARED does not open in D3D11 (OpenSharedResource1: E_INVALIDARG, any resource flags), and
      // D3D11 buffers refuse D3D11_RESOURCE_MISC_SHARED_NTHANDLE (E_INVALIDARG). What works is the other direction
      // with a legacy (KMT) handle: a D3D11 buffer with D3D11_RESOURCE_MISC_SHARED, opened by
      // ID3D12Device::OpenSharedHandle.
      const std::lock_guard lock(mutex);
      out12 = nullptr;
      out11 = nullptr;
      if (!device12)
      {
         error = "not initialized";
         return false;
      }
      if (size == 0 || size > UINT32_MAX)
      {
         error = "invalid shared buffer size";
         return false;
      }
      D3D11_BUFFER_DESC desc = {};
      desc.ByteWidth = UINT((size + 15) & ~uint64_t(15));
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      HRESULT hr = device11->CreateBuffer(&desc, nullptr, &out11);
      if (FAILED(hr))
      {
         error = HResultString("ID3D11Device::CreateBuffer (shared)", hr);
         return false;
      }
      IDXGIResource* dxgi_resource = nullptr;
      HANDLE handle = nullptr;
      hr = out11->QueryInterface(IID_PPV_ARGS(&dxgi_resource));
      if (SUCCEEDED(hr))
      {
         hr = dxgi_resource->GetSharedHandle(&handle); // KMT handle: not closed
         dxgi_resource->Release();
      }
      if (FAILED(hr) || !handle)
      {
         error = HResultString("IDXGIResource::GetSharedHandle (buffer)", hr);
         SafeRelease(out11);
         return false;
      }
      hr = device12->OpenSharedHandle(handle, IID_PPV_ARGS(&out12));
      if (FAILED(hr))
      {
         error = HResultString("ID3D12Device::OpenSharedHandle (buffer)", hr);
         SafeRelease(out11);
         return false;
      }
      return true;
   }

   bool Interop::CreateSharedTexture2D(ID3D11Device* device11, DXGI_FORMAT format, uint32_t width, uint32_t height, ID3D12Resource*& out12, ID3D11Texture2D*& out11, std::string& error)
   {
      if (!device12)
      {
         error = "interop not initialized";
         return false;
      }
      return RTAO::CreateSharedTexture(device12, device11, format, width, height, out12, out11, error);
   }

   uint64_t Interop::GetTimestampFrequency()
   {
      const std::lock_guard lock(mutex);
      UINT64 frequency = 0;
      if (!queue || FAILED(queue->GetTimestampFrequency(&frequency)))
         return 0;
      return frequency;
   }

   bool Interop::WaitForIdle(uint32_t timeout_ms)
   {
      const std::lock_guard lock(mutex);
      return WaitForGPUIdle(timeout_ms);
   }

   std::string Interop::GetDescription()
   {
      const std::lock_guard lock(mutex);
      std::string text = "adapter \"" + adapter_name + "\"";
      if (depth_shared12)
      {
         text += depth_packed ? ", depth copy R8G8B8A8_UNORM (packed R32)" : ", depth copy R32_FLOAT";
         text += ao_format == DXGI_FORMAT_R16_UNORM ? ", AO R16_UNORM" : ", AO R8_UNORM";
         char buffer[96];
         snprintf(buffer, sizeof(buffer), ", depth %ux%u, AO %ux%u", depth_width, depth_height, ao_width, ao_height);
         text += buffer;
      }
      return text;
   }
}
