#pragma once

// RTAO interop (milestone M0): a side D3D12 device on the game's D3D11 adapter, plus the textures and the fence
// the two APIs share. Pure D3D11/D3D12, no Luma or ReShade dependencies, so it can be lifted elsewhere.
//
// Patterned on the user's Fallout 4 RT core (f4-mods/rtcore/rt_device.h), including its measured sharing rules
// (RTX 4090, Win11):
//  - a D3D12 texture created with D3D12_HEAP_FLAG_SHARED opens in D3D11 (OpenSharedResource1) only with
//    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET alone. Adding ALLOW_UNORDERED_ACCESS makes the open fail, so
//    compute writes a private UAV texture which is then copied into the shared one.
//  - only a few formats were found to open (R8, R16, R8G8B8A8, R10G10B10A2 UNORM). Float formats failed.
//    We still try R32_FLOAT for the depth copy first, and fall back to R8G8B8A8_UNORM with the 32 depth bits
//    packed into the 4 channels (bit-exact).
//
// Sync model: same frame, all on the D3D11 immediate context:
//   D3D11 Signal(v) + Flush -> D3D12 queue Wait(v) -> dispatch + copy -> D3D12 Signal(v+1) -> D3D11 Wait(v+1)
// The CPU never waits on the GPU, except to recycle a command allocator that is still in flight (3-deep ring).

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace RTAO
{
   // Root constants of the D3D12 test kernel (must match kComputeShader in rtao_interop.cpp).
   struct Constants
   {
      float inv_projection[16]; // Raw bytes of the game's "cb_inverseprojectionmatrix" (PerViewCB b1)
      uint32_t ao_width;
      uint32_t ao_height;
      uint32_t depth_width;
      uint32_t depth_height;
      uint32_t depth_packed; // Set by Interop::Run, callers can leave it at 0
      uint32_t debug_mode; // 0 depth based test AO, 1 test pattern
      float strength;
      uint32_t frame;
   };
   static_assert(sizeof(Constants) % 4 == 0);

   // What a custom D3D12 pass gets to record against (see Interop::Run with a recorder)
   struct RecordInfo
   {
      ID3D12Device5* device;
      // In NON_PIXEL_SHADER_RESOURCE state during the recorder. Packed: R8G8B8A8_UNORM holding the R32 bits.
      ID3D12Resource* depth;
      DXGI_FORMAT depth_format;
      bool depth_packed;
      uint32_t depth_width, depth_height;
      // Private AO texture, in UNORDERED_ACCESS state on entry and it must be left so. Copied to the shared
      // texture after the recorder returns.
      ID3D12Resource* ao;
      DXGI_FORMAT ao_format;
      uint32_t ao_width, ao_height;
      uint32_t targets_generation; // Changes whenever the targets above were re-created
      uint32_t frame_slot; // 0..frames_in_flight-1, the command allocator in use
      uint64_t submit_fence_value; // Own-fence value signalled when this submission completes (for deferred releases)
      uint64_t completed_fence_value; // Own-fence value already reached by the GPU
   };
   using Recorder = std::function<bool(ID3D12GraphicsCommandList4* list, const RecordInfo& info, std::string& error)>;

   class Interop
   {
   public:
      Interop() = default;
      ~Interop() { Shutdown(); }
      Interop(const Interop&) = delete;
      Interop& operator=(const Interop&) = delete;

      // Creates the D3D12 device on the adapter with the same LUID as "device11", checks DXR 1.1 (tier >= 1.1),
      // creates the direct queue, the shared fence (opened as ID3D11Fence) and the test compute pipeline.
      // Safe to call more than once: does nothing if already initialized.
      bool Init(ID3D11Device* device11, std::string& error);
      bool IsInitialized() const { return device12 != nullptr; }
      void Shutdown();

      // (Re)creates the shared textures if the sizes changed. Thread safe.
      bool EnsureTargets(ID3D11Device* device11, uint32_t depth_width, uint32_t depth_height, uint32_t ao_width, uint32_t ao_height, std::string& error);

      // D3D11 views of the shared textures. AddRef'd, the caller releases them. Null until EnsureTargets succeeded.
      ID3D11RenderTargetView* GetDepthRTV();
      ID3D11ShaderResourceView* GetAOSRV();
      bool IsDepthPacked() const { return depth_packed; }

      // Runs the D3D12 pass for the depth already written (or queued) into the depth RTV.
      // MUST be called on the IMMEDIATE context (ID3D11DeviceContext4::Signal/Wait don't exist on deferred ones).
      // On return the immediate context has a GPU wait queued, so anything recorded after can read GetAOSRV().
      bool Run(ID3D11DeviceContext* immediate_context, Constants constants, std::string& error);
      // Same, but "recorder" records the D3D12 work (it must write the whole AO texture). Runs under the interop lock.
      bool Run(ID3D11DeviceContext* immediate_context, const Recorder& recorder, std::string& error);

      // Creates a buffer both APIs can access (for D3D11 CopySubresourceRegion of game vertex/index data into memory
      // D3D12 can read): a D3D11 buffer with D3D11_RESOURCE_MISC_SHARED (also SRV/UAV raw views), opened in D3D12.
      // The D3D12 side starts in COMMON state. Size is rounded up to 16 bytes.
      bool CreateSharedBuffer(ID3D11Device* device11, uint64_t size, ID3D12Resource*& out12, ID3D11Buffer*& out11, std::string& error);
      // Creates a 2D texture D3D11 can render into and D3D12 can copy from (D3D12 created, ALLOW_RENDER_TARGET only, so one
      // of the formats listed above). The D3D12 side starts in COMMON state and must be in COMMON whenever D3D11 uses it.
      bool CreateSharedTexture2D(ID3D11Device* device11, DXGI_FORMAT format, uint32_t width, uint32_t height, ID3D12Resource*& out12, ID3D11Texture2D*& out11, std::string& error);

      // Valid after Init. Not reference counted: don't keep past Shutdown.
      ID3D12Device5* GetDevice() const { return device12; }
      static constexpr uint32_t GetFramesInFlight() { return frames_in_flight; }

      // Ticks per second of the direct queue's timestamps (0 if unavailable)
      uint64_t GetTimestampFrequency();

      // Blocks until the D3D12 queue has finished everything submitted so far.
      bool WaitForIdle(uint32_t timeout_ms);

      // Log friendly description (adapter, formats).
      std::string GetDescription();

   private:
      bool CreatePipeline(std::string& error);
      void ReleaseTargets();
      void ShutdownLocked();
      bool WaitForGPUIdle(uint32_t timeout_ms);

      std::mutex mutex;

      ID3D12Device5* device12 = nullptr;
      ID3D12CommandQueue* queue = nullptr;
      static constexpr uint32_t frames_in_flight = 3;
      ID3D12CommandAllocator* allocators[frames_in_flight] = {};
      uint64_t allocator_fence_values[frames_in_flight] = {};
      uint32_t allocator_index = 0;
      ID3D12GraphicsCommandList4* command_list = nullptr;

      // Shared fence: D3D11 signals odd steps, D3D12 the even ones. Monotonic.
      ID3D12Fence* shared_fence12 = nullptr;
      ID3D11Fence* shared_fence11 = nullptr;
      uint64_t shared_fence_value = 0;
      // Signalled by the D3D12 queue only, to know when an allocator is free again.
      ID3D12Fence* own_fence = nullptr;
      uint64_t own_fence_value = 0;
      HANDLE own_fence_event = nullptr;

      ID3D12RootSignature* root_signature = nullptr;
      ID3D12PipelineState* pipeline_state = nullptr;
      ID3D12DescriptorHeap* descriptor_heap = nullptr; // [0] depth SRV, [1] AO UAV
      uint32_t descriptor_size = 0;

      // Targets
      uint32_t depth_width = 0, depth_height = 0, ao_width = 0, ao_height = 0;
      bool depth_packed = false;
      DXGI_FORMAT depth_format = DXGI_FORMAT_UNKNOWN;
      DXGI_FORMAT ao_format = DXGI_FORMAT_UNKNOWN;
      ID3D12Resource* depth_shared12 = nullptr;
      ID3D11Texture2D* depth_shared11 = nullptr;
      ID3D11RenderTargetView* depth_rtv11 = nullptr;
      ID3D12Resource* ao_private12 = nullptr; // UAV, written by the compute shader
      ID3D12Resource* ao_shared12 = nullptr; // copy of ao_private12, read by D3D11
      ID3D11Texture2D* ao_shared11 = nullptr;
      ID3D11ShaderResourceView* ao_srv11 = nullptr;

      uint32_t targets_generation = 0;
      std::string adapter_name;
   };
}
