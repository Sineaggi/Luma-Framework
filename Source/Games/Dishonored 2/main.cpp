#define GAME_DISHONORED_2 1

// Hangs on boot
#define DISABLE_AUTO_DEBUGGER
// Previously disabled as it made boot extremely slow, it should now be fine as we optimized the code
#define ALLOW_SHADERS_DUMPING 1

// Not used by Dishonored 2?
#define ENABLE_SHADER_CLASS_INSTANCES 1

// RTAO creates a side D3D12 device, which ReShade wraps and reports to Luma like a game device. Without this,
// Luma's handlers treat it as D3D11, and release builds pop a "non DirectX 11 device" message box mid frame.
#define CHECK_GRAPHICS_API_COMPATIBILITY 1

#include "../../../Shaders/Dishonored 2/Includes/GameCBuffers.hlsl"
#include "..\..\Core\core.hpp"

#include "rtao_interop.h"
#include "rtao_accel.h"
#include "rtao_capture.h"
#include "rtao_pass.h"

#include <shared_mutex>

struct CBPerViewGlobals
{
   float4 cb_alwaystweak;
   float4 cb_viewrandom;
   Matrix44F cb_viewprojectionmatrix;
   Matrix44F cb_viewmatrix;
   // apparently zero? seemengly unrelated from jitters
   float4 cb_subpixeloffset;
   Matrix44F cb_projectionmatrix;
   Matrix44F cb_previousviewprojectionmatrix;
   Matrix44F cb_previousviewmatrix;
   Matrix44F cb_previousprojectionmatrix;
   float4 cb_mousecursorposition;
   float4 cb_mousebuttonsdown;
   // Jitters in UV space (not Halton, R2, or Sobol. Custom sequence?). xy current frame, zw previous frame.
   float4 cb_jittervectors;
   Matrix44F cb_inverseviewprojectionmatrix;
   Matrix44F cb_inverseviewmatrix;
   Matrix44F cb_inverseprojectionmatrix;
   float4 cb_globalviewinfos;
   float3 cb_wscamforwarddir;
   uint cb_alwaysone;
   float3 cb_wscamupdir;
   // This seems to be true at all times for TAA
   uint cb_usecompressedhdrbuffers;
   float3 cb_wscampos;
   float cb_time;
   float3 cb_wscamleftdir;
   float cb_systime;
   float2 cb_jitterrelativetopreviousframe;
   float2 cb_worldtime;
   float2 cb_shadowmapatlasslicedimensions;
   float2 cb_resolutionscale;
   float2 cb_parallelshadowmapslicedimensions;
   float cb_framenumber;
   uint cb_alwayszero;
};

namespace
{
   ShaderHashesList shader_hashes_TAA;
   ShaderHashesList shader_hashes_Tonemap;
   ShaderHashesList shader_hashes_UpscaleSharpen;
   ShaderHashesList shader_hashes_DownsampleDepth;
   ShaderHashesList shader_hashes_UnprojectDepth;
   ShaderHashesList shader_hashes_SSAO;
   ShaderHashesList shader_hashes_AODenoise; // The game's AO blur, after the SSAO (or XeGTAO/RTAO) draw

   // XeGTAO
   constexpr size_t XE_GTAO_DEPTH_MIP_LEVELS = 5;

   // Ambient occlusion mode. This is the single AO setting (config key "AOMode"): set it from anywhere
   // (UI, config, an exported setter) and the SSAO draw picks it up on the next frame.
   // XeGTAO and RTAO both replace the game's SSAO draw, so they are mutually exclusive by construction.
   enum AOMode : int
   {
      AO_MODE_OFF = 0, // No AO replacement and the game's HBAO+ off (the game's own SSAO/DWAO draws nothing on PC)
      AO_MODE_XEGTAO = 1,
      AO_MODE_RTAO = 2, // Ray traced AO on a side D3D12 device, from geometry captured off the game's draws
      AO_MODE_HBAO = 3, // The game's NVIDIA HBAO+ (r_hbao 1), no replacement
      AO_MODE_COUNT
   };
   std::atomic<int> g_ao_mode = AO_MODE_XEGTAO;
   // Set when g_ao_mode changed and the game's r_hbao has to follow (applied on present, on the render thread)
   std::atomic<bool> g_ao_mode_apply_pending = true;

   // The game's r_hbao cvar, driven from Luma (Steam 1.77.9.0). The addresses are the ones the user's AO menu ASI
   // uses (Dishonored2AOMenu.asi); nothing is hooked, the engine's own idCVar::SetBool is called.
   struct GameAOCvars
   {
      bool valid = false;
      uint8_t* r_hbao = nullptr;
      uint8_t* r_skip_ssao = nullptr; // With 1 the engine's AO pass only clears/blurs: the SSAO draw Luma replaces never runs
      using SetBoolFn = void(__fastcall*)(void* cvar, char value, char notify);
      SetBoolFn set_bool = nullptr;
      int applied = -1; // The r_hbao value Luma last set (-1: none yet)
      int applied_skip = -1; // The r_skipSSAO value last seen or set
      bool boot_applied = false;
   };
   GameAOCvars g_game_ao;

   void InitGameAOCvars()
   {
      uint8_t* base = reinterpret_cast<uint8_t*>(GetModuleHandleW(nullptr));
      if (!base)
         return;
      const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
      const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
      constexpr uint32_t kHbaoRegistration = 0x136b022; // "lea rdx, [name]; lea rcx, [r_hbao]" in the cvar registration
      constexpr uint32_t kHbaoCvar = 0x32ef770;
      constexpr uint32_t kSkipSsaoCvar = 0x397b4d0;
      constexpr uint32_t kSetBool = 0x102520;
      static const uint8_t kExpected[] = { 0x48, 0x8d, 0x15, 0x8f, 0x59, 0x7b, 0x00, 0x48, 0x8d, 0x0d, 0x40, 0x47, 0xf8, 0x01 };
      if (dos->e_magic != IMAGE_DOS_SIGNATURE || nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.SizeOfImage <= kSkipSsaoCvar + 0x30 ||
         std::memcmp(base + kHbaoRegistration, kExpected, sizeof(kExpected)) != 0)
      {
         reshade::log::message(reshade::log::level::warning, "AO: unknown Dishonored2.exe version, the HBAO+ option is disabled");
         return;
      }
      g_game_ao.r_hbao = base + kHbaoCvar;
      g_game_ao.r_skip_ssao = base + kSkipSsaoCvar;
      g_game_ao.set_bool = reinterpret_cast<GameAOCvars::SetBoolFn>(base + kSetBool);
      g_game_ao.valid = true;
      reshade::log::message(reshade::log::level::info, "AO: game r_hbao found, HBAO+ selectable from Luma");
   }

   int ReadGameHbao()
   {
      return *reinterpret_cast<const volatile int*>(g_game_ao.r_hbao + 0x28) != 0 ? 1 : 0;
   }

   int ReadGameSkipSsao()
   {
      return *reinterpret_cast<const volatile int*>(g_game_ao.r_skip_ssao + 0x28) != 0 ? 1 : 0;
   }
   // Set once if the D3D12/DXR 1.1 interop can't be created; RTAO then runs as XeGTAO
   std::atomic<bool> g_rtao_unavailable = false;
   std::mutex g_rtao_status_mutex;
   std::string g_rtao_status; // Last RTAO init result or error, shown in the UI
   // What RTAO mode shows (config key "RTAOTestOutput", its name kept from M0)
   enum RTAOOutput : int
   {
      RTAO_OUTPUT_AO = 0, // Ray traced AO, denoised
      RTAO_OUTPUT_TEST_PATTERN = 1, // M0 interop test kernel
      RTAO_OUTPUT_DEPTH_TEST = 2, // M0 depth crease test kernel
      RTAO_OUTPUT_COVERAGE = 3, // BLAS coverage: white = captured geometry matches the depth, grey = missing, black = misplaced
      RTAO_OUTPUT_RAW = 4, // Ray traced AO, no denoiser
      RTAO_OUTPUT_COUNT
   };
   int g_rtao_debug_mode = RTAO_OUTPUT_AO;
   // "View AO only" debug: 0 off, 1 the final AO (the game's blur output, any AO mode), 2 the RTAO output before the game's blur
   int g_ao_view_mode = 0;
   float g_rtao_strength = 1.f;
   float g_rtao_radius = 1.5f; // World units (config "RTAORadius")
   int g_rtao_rays = 4; // Per pixel per frame (config "RTAORays")
   int g_rtao_capture_pass = 1; // 0: depth only draws (no render target bound), 1: every depth writing draw (config "RTAOCapturePass")

   bool RTAOOutputIsRayTraced(int output)
   {
      return output == RTAO_OUTPUT_AO || output == RTAO_OUTPUT_COVERAGE || output == RTAO_OUTPUT_RAW;
   }

   // Geometry capture: what each game vertex shader / input layout means for capture, by native object pointer
   std::shared_mutex g_rtao_shader_mutex;
   std::unordered_map<uint64_t, RTAO::VertexShaderInfo> g_rtao_vertex_shaders;
   std::unordered_map<uint64_t, RTAO::InputLayoutInfo> g_rtao_input_layouts;
   std::atomic<bool> g_rtao_capture_active = false; // Set while RTAO traces, so the draw hook costs nothing otherwise
   std::atomic<uint32_t> g_rtao_main_depth_width = 0, g_rtao_main_depth_height = 0; // The main view's depth size
   std::atomic<uint64_t> g_rtao_vs_by_source[5] = {}; // Reflected vertex shaders, by RTAO::MatrixSource
   std::atomic<uint64_t> g_rtao_draws_by_source[5] = {}; // Main view draws, by their vertex shader's matrix source
   std::atomic<uint64_t> g_rtao_draws_unknown_shader = 0, g_rtao_draws_unusable_layout = 0, g_rtao_indirect_draws = 0, g_rtao_draws_not_triangle_list = 0;
   // Main view draws whose vertex shader can't be captured, by reason (the reasons are static strings)
   std::mutex g_rtao_skip_reason_mutex;
   std::unordered_map<const char*, uint64_t> g_rtao_skip_reasons;

   int GetEffectiveAOMode()
   {
      int mode = g_ao_mode.load();
      if (mode < AO_MODE_OFF || mode >= AO_MODE_COUNT)
         mode = AO_MODE_XEGTAO;
      if (mode == AO_MODE_RTAO && g_rtao_unavailable)
         mode = AO_MODE_XEGTAO;
      if (mode == AO_MODE_HBAO && !g_game_ao.valid)
         mode = AO_MODE_OFF;
      return mode;
   }

   void SetRTAOStatus(std::string status, bool error)
   {
      reshade::log::message(error ? reshade::log::level::error : reshade::log::level::info, ("RTAO: " + status).c_str());
      const std::lock_guard lock(g_rtao_status_mutex);
      g_rtao_status = std::move(status);
   }

#if DEVELOPMENT
   std::thread::id global_cbuffer_thread_id;
#endif

   // Directly from cbuffer (so these are transposed)
   Matrix44F projection_matrix;
   Matrix44F nearest_projection_matrix; // For first person weapons (view model)
   Matrix44F previous_projection_matrix;
   Matrix44F previous_nearest_projection_matrix;
   float2 previous_projection_jitters = { 0, 0 };
   float2 projection_jitters = { 0, 0 };
   CBPerViewGlobals cb_per_view_global = { };
   CBPerViewGlobals cb_per_view_global_previous = cb_per_view_global;

#if DEVELOPMENT
   std::vector<std::string> cb_per_view_globals_last_drawn_shader; // Not exactly thread safe but it's fine...
   std::vector<CBPerViewGlobals> cb_per_view_globals;
   std::vector<CBPerViewGlobals> cb_per_view_globals_previous;
#endif

   // Dev or User settings:
#if DEVELOPMENT
   float sr_custom_exposure = 0.f; // Ignored at 0
   float sr_custom_pre_exposure = 0.f; // Ignored at 0
   int dlss_test = 0; //TODOFT
   int force_taa_jitter_phases = 0; // Ignored if 0 (automatic mode), set to 1 to basically disable jitters
   bool disable_taa_jitters = false;
#endif
}

struct GameDeviceDataDishonored2 final : public GameDeviceData
{
   // TAA runs on a deferred context. Split its command list so the scene inputs execute before SR,
   // then inject SR before the game replays the post-TAA remainder.
   std::atomic<void*> sr_deferred_command_list = nullptr;
   std::atomic<void*> sr_remainder_command_list = nullptr;
   ComPtr<ID3D11CommandList> sr_partial_command_list;
   com_ptr<ID3D11Resource> sr_source_color;
   com_ptr<ID3D11Resource> sr_motion_vectors;
   //com_ptr<ID3D11Texture2D> sr_output_color_2; //TODOFT: delete this and related code

   // We are getting these from the game's TAA.
   ComPtr<ID3D11Buffer> cb_taa_b1;
   ComPtr<ID3D11Buffer> cb_taa_b2;
   ComPtr<ID3D11ShaderResourceView> srv_ro_postfx_luminance_buffautoexposure;

   // Game state
   com_ptr<ID3D11Resource> depth_buffer;

   std::atomic<bool> has_drawn_scene = false; // This is set early in the frame, as soon as we detect that the 3D scene is rendering (it won't be made true only when it finished rendering)
   std::atomic<void*> final_post_process_command_list = nullptr;
   bool has_drawn_scene_previous = false;

   bool found_per_view_globals = false;

   std::atomic<bool> prey_drs_active = false;
   std::atomic<bool> prey_drs_detected = false;
   std::atomic<bool> prey_taa_active = false; // Instant version of "taa_detected".
   // Index 0 is one frame ago, index 1 is two frames ago
   bool previous_prey_taa_active[2] = { false, false };

   // RTAO
   RTAO::Interop rtao;
   // When the SSAO draw is recorded on a deferred context, the D3D11<->D3D12 fence Signal/Wait can't be recorded
   // (ID3D11DeviceContext4::Signal/Wait are immediate context only). So the deferred context is split there:
   // everything up to (and including) our depth copy is finished into "prefix", and when the game executes the
   // remainder on the immediate context we execute the prefix, run the D3D12 pass, and let the remainder (which
   // starts with our AO composite draw) run after the GPU wait.
   struct RTAOPendingSplit
   {
      ID3D11DeviceContext* deferred_context = nullptr; // Identity only
      ComPtr<ID3D11CommandList> prefix;
      void* remainder_command_list = nullptr; // Identity only, set when the game calls FinishCommandList()
      RTAO::Constants constants = {};
      RTAO::PassSettings settings = {};
      int output = 0;
      std::shared_ptr<RTAO::GeometryCapture::PendingRecords> capture_records; // Captured before the split: in the prefix
      uint32_t frame = 0;
   };
   std::mutex rtao_pending_mutex;
   std::vector<RTAOPendingSplit> rtao_pending;
   uint32_t rtao_frame = 0;
   uint32_t rtao_init_frame = 0;
   std::atomic<bool> rtao_logged_context_type = false;
   std::atomic<bool> rtao_logged_run_error = false;
   // Diagnostics, logged a few times per session (OnPresent)
   std::atomic<uint32_t> rtao_stat_inline_ok = 0, rtao_stat_splits = 0, rtao_stat_mapped = 0, rtao_stat_deferred_ok = 0, rtao_stat_failed = 0, rtao_stat_dropped = 0;

   // RTAO ray tracing (used on the immediate context's thread, inside the interop's recorder)
   RTAO::GeometryCapture rtao_capture;
   RTAO::BlasCache rtao_blases;
   RTAO::TlasBuilder rtao_tlas;
   RTAO::AOPass rtao_pass;
   RTAO::GpuTimer rtao_timer;
   RTAO::Retirer rtao_retirer;
   bool rtao_rt_ready = false;
   bool rtao_rt_failed = false;
   int rtao_last_output = -1;
   // "View AO only": the last AO blur render target of the frame
   std::mutex ao_view_mutex;
   com_ptr<ID3D11Resource> ao_view_resource;
   ComPtr<ID3D11Buffer> ao_view_cb;
   uint32_t rtao_last_draw_frame = 0;
   // Snapshots for the UI (written by the recorder)
   std::atomic<uint64_t> rtao_blas_count = 0, rtao_blas_bytes = 0;
   std::atomic<uint32_t> rtao_tlas_instances = 0, rtao_gpu_us = 0;
};

//TODOFT: Building shaders menu (apply gamma?)
class Dishonored2 final : public Game
{
public:
   static const GameDeviceDataDishonored2& GetGameDeviceData(const DeviceData& device_data)
   {
      return *static_cast<const GameDeviceDataDishonored2*>(device_data.game);
   }
   static GameDeviceDataDishonored2& GetGameDeviceData(DeviceData& device_data)
   {
      return *static_cast<GameDeviceDataDishonored2*>(device_data.game);
   }

   void OnLoad(std::filesystem::path& file_path, bool failed) override
   {
      if (!failed)
      {
         reshade::register_event<reshade::addon_event::map_buffer_region>(Dishonored2::OnMapBufferRegion);
         reshade::register_event<reshade::addon_event::unmap_buffer_region>(Dishonored2::OnUnmapBufferRegion);
         reshade::register_event<reshade::addon_event::execute_secondary_command_list>(Dishonored2::OnExecuteSecondaryCommandList);
         InitGameAOCvars();
         reshade::register_event<reshade::addon_event::init_pipeline>(Dishonored2::OnRTAOInitPipeline);
         reshade::register_event<reshade::addon_event::destroy_pipeline>(Dishonored2::OnRTAODestroyPipeline);
         reshade::register_event<reshade::addon_event::draw_indexed>(Dishonored2::OnRTAODrawIndexed);
         reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(Dishonored2::OnRTAODrawIndirect);
         reshade::register_event<reshade::addon_event::destroy_resource>(Dishonored2::OnRTAODestroyResource);
      }
   }

   void OnInit(bool async) override
   {
      GetShaderDefineData(POST_PROCESS_SPACE_TYPE_HASH).SetDefaultValue('1');
      GetShaderDefineData(GAMMA_CORRECTION_TYPE_HASH).SetDefaultValue('1');
      GetShaderDefineData(UI_DRAW_TYPE_HASH).SetDefaultValue('3');

      std::vector<ShaderDefineData> game_shader_defines_data = {
         { "XE_GTAO_QUALITY", '2', true, false, "0 - Low\n1 - Medium\n2 - High\n3 - Very High\n4 - Ultra", 4 },
         { "DISABLE_LENS_DISTORTION", '0', true, false, "Disable lens distortion while running or ducked.", 1 }
      };

      shader_defines_data.append_range(game_shader_defines_data);

      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 XeGTAO Prefilter Depths CS"), ShaderDefinition{ "Luma_XeGTAO", reshade::api::pipeline_subobject_type::compute_shader, nullptr, "prefilter_depths16x16_cs" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 XeGTAO Main Pass PS"), ShaderDefinition{ "Luma_XeGTAO", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "main_pass_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Depth Copy PS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "depth_copy_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Depth Pack PS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "depth_pack_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 AO View PS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "ao_view_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Capture CS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::compute_shader, nullptr, "capture_instances_cs" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Composite PS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "composite_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 Pre DLSS CS"), ShaderDefinition{ "Luma_PreDLSS_CS", reshade::api::pipeline_subobject_type::compute_shader });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 Post DLSS CS"), ShaderDefinition{ "Luma_PostDLSS_CS", reshade::api::pipeline_subobject_type::compute_shader });
   }

   // This needs to be overridden with your own "GameDeviceData" sub-class (destruction is automatically handled)
   void OnCreateDevice(ID3D11Device* native_device, DeviceData& device_data) override
   {
      device_data.game = new GameDeviceDataDishonored2;
   }

   // "GameDeviceData" has no virtual destructor, so delete it as our type (releases its resources and the D3D12 device)
   void OnDestroyDeviceData(DeviceData& device_data) override
   {
      g_rtao_capture_active = false;
      if (device_data.game)
         static_cast<GameDeviceDataDishonored2*>(device_data.game)->rtao.WaitForIdle(2000); // Nothing in flight when the RT objects go
      delete static_cast<GameDeviceDataDishonored2*>(device_data.game);
      device_data.game = nullptr;
   }

   // Replaces the game's SSAO draw with the RTAO result. Returns false to let the draw run as usual (vanilla SSAO this frame).
   bool DrawRTAO(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, DeviceData& device_data, GameDeviceDataDishonored2& game_device_data, std::function<void()>* original_draw_dispatch_func)
   {
      auto& rtao = game_device_data.rtao;
      std::string error;
      if (!rtao.IsInitialized())
      {
         if (!rtao.Init(native_device, error))
         {
            g_rtao_unavailable = true;
            SetRTAOStatus("unavailable, using XeGTAO instead: " + error, true);
            return false;
         }
         SetRTAOStatus("D3D12 interop ready, " + rtao.GetDescription(), false);
         game_device_data.rtao_init_frame = game_device_data.rtao_frame;
      }

      const auto copy_vs = device_data.native_vertex_shaders.find(CompileTimeStringHash("Copy VS"));
      const auto depth_copy_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 RTAO Depth Copy PS"));
      const auto depth_pack_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 RTAO Depth Pack PS"));
      const auto composite_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 RTAO Composite PS"));
      if (copy_vs == device_data.native_vertex_shaders.end() || !copy_vs->second || depth_copy_ps == device_data.native_pixel_shaders.end() || !depth_copy_ps->second || depth_pack_ps == device_data.native_pixel_shaders.end() || !depth_pack_ps->second || composite_ps == device_data.native_pixel_shaders.end() || !composite_ps->second)
      {
         return false; // Shaders not compiled (yet)
      }

      com_ptr<ID3D11ShaderResourceView> depth_srv;
      native_device_context->PSGetShaderResources(0, 1, &depth_srv);
      com_ptr<ID3D11RenderTargetView> ao_rtv;
      native_device_context->OMGetRenderTargets(1, &ao_rtv, nullptr);
      if (!depth_srv || !ao_rtv)
      {
         return false;
      }
      com_ptr<ID3D11Resource> depth_resource;
      depth_srv->GetResource(&depth_resource);
      com_ptr<ID3D11Resource> ao_resource;
      ao_rtv->GetResource(&ao_resource);
      com_ptr<ID3D11Texture2D> depth_texture;
      com_ptr<ID3D11Texture2D> ao_texture;
      if (FAILED(depth_resource->QueryInterface(&depth_texture)) || FAILED(ao_resource->QueryInterface(&ao_texture)))
      {
         return false;
      }
      D3D11_TEXTURE2D_DESC depth_desc;
      depth_texture->GetDesc(&depth_desc);
      D3D11_TEXTURE2D_DESC ao_desc;
      ao_texture->GetDesc(&ao_desc);

      if (!rtao.EnsureTargets(native_device, depth_desc.Width, depth_desc.Height, ao_desc.Width, ao_desc.Height, error))
      {
         g_rtao_unavailable = true;
         SetRTAOStatus("shared textures failed, using XeGTAO instead: " + error, true);
         return false;
      }

      // Ray traced outputs need the capture, the TLAS and the pass; if they can't start, show the depth test instead
      int output = std::clamp(g_rtao_debug_mode, 0, int(RTAO_OUTPUT_COUNT) - 1);
      if (RTAOOutputIsRayTraced(output) && !EnsureRTAORayTracing(native_device, device_data, game_device_data))
         output = RTAO_OUTPUT_DEPTH_TEST;
      const bool ray_traced = RTAOOutputIsRayTraced(output);
      g_rtao_capture_active = ray_traced;
      game_device_data.rtao_last_draw_frame = game_device_data.rtao_frame;
      g_rtao_main_depth_width = depth_desc.Width;
      g_rtao_main_depth_height = depth_desc.Height;
      RTAO::PassSettings settings = MakeRTAOSettings(output);
      settings.reset_history = output != game_device_data.rtao_last_output;
      game_device_data.rtao_last_output = output;

      const bool immediate = native_device_context->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE;
      if (!game_device_data.rtao_logged_context_type.exchange(true))
      {
         SetRTAOStatus(std::string("SSAO draw is on the ") + (immediate ? "immediate context, D3D12 pass runs inline" : "deferred context, D3D12 pass runs when the game executes it on the immediate context") + ", " + rtao.GetDescription(), false);
      }
      if (!immediate)
      {
         // Only one split per deferred context can be pending at once
         const std::lock_guard lock(game_device_data.rtao_pending_mutex);
         for (const auto& pending : game_device_data.rtao_pending)
         {
            if (pending.deferred_context == native_device_context && pending.remainder_command_list == nullptr)
               return false;
         }
      }

      // 1) Copy the depth into the texture shared with D3D12
      {
         DrawStateStack<DrawStateStackType::FullGraphics> draw_state_stack;
         draw_state_stack.Cache(native_device_context, device_data.uav_max_count);
         ID3D11RenderTargetView* depth_rtv = rtao.GetDepthRTV();
         ID3D11PixelShader* ps = rtao.IsDepthPacked() ? depth_pack_ps->second.get() : depth_copy_ps->second.get();
         DrawCustomPixelShader(native_device_context, device_data.default_depth_stencil_state.get(), device_data.default_blend_state.get(), nullptr, copy_vs->second.get(), ps, depth_srv.get(), depth_rtv, depth_desc.Width, depth_desc.Height);
         if (depth_rtv)
            depth_rtv->Release();
         draw_state_stack.Restore(native_device_context);
      }

      RTAO::Constants constants = {};
      static_assert(sizeof(constants.inv_projection) == sizeof(cb_per_view_global.cb_inverseprojectionmatrix));
      std::memcpy(constants.inv_projection, &cb_per_view_global.cb_inverseprojectionmatrix, sizeof(constants.inv_projection));
      constants.debug_mode = output == RTAO_OUTPUT_TEST_PATTERN ? 1 : 0;
      constants.strength = g_rtao_strength;
      constants.frame = cb_luma_global_settings.FrameIndex;

      // 2) D3D11 -> D3D12 -> D3D11
      if (immediate)
      {
         const bool ok = ray_traced ? rtao.Run(native_device_context, MakeRTAORecorder(game_device_data, settings), error) : rtao.Run(native_device_context, constants, error);
         if (!ok)
         {
            game_device_data.rtao_stat_failed++;
            if (!game_device_data.rtao_logged_run_error.exchange(true))
            {
               SetRTAOStatus("D3D12 pass failed: " + error, true);
            }
            return false;
         }
         game_device_data.rtao_stat_inline_ok++;
         game_device_data.rtao_capture.MarkAOPassDone();
      }
      else
      {
         GameDeviceDataDishonored2::RTAOPendingSplit pending;
         pending.deferred_context = native_device_context;
         pending.constants = constants;
         pending.settings = settings;
         pending.output = output;
         pending.frame = game_device_data.rtao_frame;
         // Geometry recorded so far on this context lands in the prefix (executed before the D3D12 pass)
         if (ray_traced)
            pending.capture_records = game_device_data.rtao_capture.TakeContextRecords(native_device_context);
         if (FAILED(native_device_context->FinishCommandList(TRUE, pending.prefix.put())) || !pending.prefix)
         {
            return false;
         }
         const std::lock_guard lock(game_device_data.rtao_pending_mutex);
         game_device_data.rtao_pending.push_back(std::move(pending));
         game_device_data.rtao_stat_splits++;
      }

      // 3) Write the shared AO into the game's AO render target, reusing the game's own fullscreen draw state
      com_ptr<ID3D11PixelShader> original_ps;
      native_device_context->PSGetShader(&original_ps, nullptr, nullptr);
      ID3D11ShaderResourceView* ao_srv = rtao.GetAOSRV();
      native_device_context->PSSetShader(composite_ps->second.get(), nullptr, 0);
      native_device_context->PSSetShaderResources(0, 1, &ao_srv);
      if (original_draw_dispatch_func && *original_draw_dispatch_func)
         (*original_draw_dispatch_func)();
      else
         native_device_context->Draw(3, 0); // Same assumption as the XeGTAO path
      ID3D11ShaderResourceView* const original_srv = depth_srv.get();
      native_device_context->PSSetShaderResources(0, 1, &original_srv);
      native_device_context->PSSetShader(original_ps.get(), nullptr, 0);
      if (ao_srv)
         ao_srv->Release();
      return true;
   }

   static DXGI_FORMAT TypedViewFormat(DXGI_FORMAT format)
   {
      switch (format)
      {
      case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
      case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
      case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
      case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
      case DXGI_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_UNORM;
      case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
      case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_FLOAT;
      case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
      default: return format;
      }
   }

   // "View AO only" debug: draws the selected AO texture fullscreen as grey over the back buffer, at present (after
   // Luma's display composition, so it covers the scene and the UI)
   static void DrawAOView(ID3D11Device* native_device, DeviceData& device_data, GameDeviceDataDishonored2& game_device_data)
   {
      const auto copy_vs = device_data.native_vertex_shaders.find(CompileTimeStringHash("Copy VS"));
      const auto view_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 AO View PS"));
      if (copy_vs == device_data.native_vertex_shaders.end() || !copy_vs->second || view_ps == device_data.native_pixel_shaders.end() || !view_ps->second)
         return;

      // The AO texture
      com_ptr<ID3D11ShaderResourceView> srv;
      float2 uv_scale = { 1.f, 1.f };
      if (g_ao_view_mode == 2)
      {
         if (!game_device_data.rtao.IsInitialized())
            return;
         ID3D11ShaderResourceView* rtao_srv = game_device_data.rtao.GetAOSRV();
         if (!rtao_srv)
            return;
         srv = rtao_srv;
         rtao_srv->Release();
      }
      else
      {
         com_ptr<ID3D11Resource> resource;
         {
            const std::lock_guard lock(game_device_data.ao_view_mutex);
            resource = game_device_data.ao_view_resource;
         }
         com_ptr<ID3D11Texture2D> texture;
         if (!resource || FAILED(resource->QueryInterface(&texture)))
            return;
         D3D11_TEXTURE2D_DESC desc;
         texture->GetDesc(&desc);
         D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
         srv_desc.Format = TypedViewFormat(desc.Format);
         srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
         srv_desc.Texture2D.MipLevels = 1;
         if (FAILED(native_device->CreateShaderResourceView(texture.get(), &srv_desc, &srv)))
            return;
      }
      {
         // The game renders into the top left render resolution part of its textures
         com_ptr<ID3D11Resource> resource;
         srv->GetResource(&resource);
         com_ptr<ID3D11Texture2D> texture;
         if (!resource || FAILED(resource->QueryInterface(&texture)))
            return;
         D3D11_TEXTURE2D_DESC desc;
         texture->GetDesc(&desc);
         uv_scale.x = std::clamp(device_data.render_resolution.x / float(desc.Width), 0.f, 1.f);
         uv_scale.y = std::clamp(device_data.render_resolution.y / float(desc.Height), 0.f, 1.f);
      }

      // The back buffer
      com_ptr<ID3D11Texture2D> back_buffer;
      {
         const std::shared_lock lock(device_data.mutex);
         if (device_data.swapchains.empty())
            return;
         IDXGISwapChain* swapchain = (IDXGISwapChain*)((*device_data.swapchains.begin())->get_native());
         if (!swapchain || FAILED(swapchain->GetBuffer(0, IID_PPV_ARGS(&back_buffer))))
            return;
      }
      D3D11_TEXTURE2D_DESC back_buffer_desc;
      back_buffer->GetDesc(&back_buffer_desc);
      D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
      rtv_desc.Format = TypedViewFormat(back_buffer_desc.Format);
      rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
      com_ptr<ID3D11RenderTargetView> rtv;
      if (FAILED(native_device->CreateRenderTargetView(back_buffer.get(), &rtv_desc, &rtv)))
         return;
      const bool scrgb = rtv_desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;

      if (!game_device_data.ao_view_cb)
      {
         D3D11_BUFFER_DESC cb_desc = { 32, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
         if (FAILED(native_device->CreateBuffer(&cb_desc, nullptr, game_device_data.ao_view_cb.put())))
            return;
      }
      com_ptr<ID3D11DeviceContext> context;
      native_device->GetImmediateContext(&context);
      D3D11_MAPPED_SUBRESOURCE mapped;
      if (FAILED(context->Map(game_device_data.ao_view_cb.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
         return;
      const float params[8] = { 1.f / float(back_buffer_desc.Width), 1.f / float(back_buffer_desc.Height), uv_scale.x, uv_scale.y, scrgb ? 203.f / 80.f : 1.f, 0.f, 0.f, 0.f };
      std::memcpy(mapped.pData, params, sizeof(params));
      context->Unmap(game_device_data.ao_view_cb.get(), 0);

      DrawStateStack<DrawStateStackType::FullGraphics> draw_state_stack;
      draw_state_stack.Cache(context.get(), device_data.uav_max_count);
      ID3D11Buffer* const cb = game_device_data.ao_view_cb.get();
      context->PSSetConstantBuffers(0, 1, &cb);
      DrawCustomPixelShader(context.get(), device_data.default_depth_stencil_state.get(), device_data.default_blend_state.get(), device_data.sampler_state_linear.get(), copy_vs->second.get(), view_ps->second.get(), srv.get(), rtv.get(), back_buffer_desc.Width, back_buffer_desc.Height);
      draw_state_stack.Restore(context.get());
   }

   // Lazily starts the ray tracing part of RTAO (capture, acceleration structures, pass). Render thread.
   static bool EnsureRTAORayTracing(ID3D11Device* native_device, DeviceData& device_data, GameDeviceDataDishonored2& game_device_data)
   {
      if (game_device_data.rtao_rt_ready)
         return true;
      if (game_device_data.rtao_rt_failed)
         return false;
      const auto capture_cs = device_data.native_compute_shaders.find(CompileTimeStringHash("DS2 RTAO Capture CS"));
      if (capture_cs == device_data.native_compute_shaders.end() || !capture_cs->second)
         return false; // Not compiled (yet)
      std::string error;
      auto& rtao = game_device_data.rtao;
      if (!game_device_data.rtao_capture.Init(native_device, rtao, capture_cs->second.get(), error) || !game_device_data.rtao_pass.Init(rtao.GetDevice(), error) ||
         !game_device_data.rtao_timer.Init(rtao.GetDevice(), rtao.GetTimestampFrequency(), error))
      {
         game_device_data.rtao_rt_failed = true;
         SetRTAOStatus("ray tracing failed to start, showing the depth test instead: " + error, true);
         return false;
      }
      game_device_data.rtao_rt_ready = true;
      SetRTAOStatus("ray tracing ready (geometry capture, BLAS/TLAS, RayQuery AO + denoiser)", false);
      return true;
   }

   // Camera from the game's per view constants (the copy Luma keeps, see UpdateGlobalCB). The game's vertex shaders
   // compute clip = world.x * VP[0] + world.y * VP[1] + world.z * VP[2] + VP[3] on the matrix registers as stored,
   // which is the row vector convention RTAO::PassSettings uses, so the bytes are taken as they are.
   static RTAO::PassSettings MakeRTAOSettings(int output)
   {
      RTAO::PassSettings settings;
      static_assert(sizeof(settings.view_proj) == sizeof(cb_per_view_global.cb_viewprojectionmatrix));
      std::memcpy(settings.view_proj, &cb_per_view_global.cb_viewprojectionmatrix, sizeof(settings.view_proj));
      RTAO::InvertMatrix(settings.view_proj, settings.inv_view_proj);
      std::memcpy(settings.cam_pos, &cb_per_view_global.cb_wscampos, sizeof(settings.cam_pos));
      std::memcpy(settings.cam_fwd, &cb_per_view_global.cb_wscamforwarddir, sizeof(settings.cam_fwd));
      settings.reversed_z = true; // Dishonored 2 renders with inverted depth (see the DLSS settings)
      settings.radius = (std::max)(g_rtao_radius, 0.01f);
      settings.ray_count = uint32_t(std::clamp(g_rtao_rays, 1, 64));
      settings.filter_mode = (output == RTAO_OUTPUT_AO) ? RTAO::FilterMode::Temporal : RTAO::FilterMode::Off;
      settings.debug_mode = output == RTAO_OUTPUT_COVERAGE ? 3 : 0;
      settings.strength = g_rtao_strength;
      return settings;
   }

   static RTAO::Recorder MakeRTAORecorder(GameDeviceDataDishonored2& game_device_data, const RTAO::PassSettings& settings)
   {
      return [&game_device_data, settings](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& error) {
         auto& g = game_device_data;
         g.rtao_retirer.Collect(info.completed_fence_value);
         g.rtao_timer.Collect(info.frame_slot);
         g.rtao_timer.Begin(list, info.frame_slot);
         if (!g.rtao_capture.Record(list, info, g.rtao_blases, g.rtao_tlas, g.rtao_retirer, 64, error))
            return false;
         const RTAO::BuildContext ctx = { info.device, list, &g.rtao_retirer, info.submit_fence_value };
         D3D12_GPU_VIRTUAL_ADDRESS top = 0;
         uint32_t instances = 0;
         if (!g.rtao_tlas.Build(ctx, g.rtao_blases, info.frame_slot, top, instances, error))
            return false;
         if (!g.rtao_pass.Record(list, info, top, settings, g.rtao_retirer, error))
            return false;
         g.rtao_timer.End(list, info.frame_slot);
         g.rtao_blas_count = g.rtao_blases.Count();
         g.rtao_blas_bytes = g.rtao_blases.ResultBytes();
         g.rtao_tlas_instances = instances;
         g.rtao_gpu_us = uint32_t(g.rtao_timer.LastMs() * 1000.0);
         return true;
      };
   }

   // Deferred SSAO only: map the game's FinishCommandList() to our pending split, and run the D3D12 pass right before
   // the game executes that command list on the immediate context. Also does the geometry capture's command list
   // bookkeeping (records count as executed only when their command list runs).
   static void HandleRTAOCommandList(reshade::api::command_list* cmd_list, reshade::api::command_list* secondary_cmd_list)
   {
      auto& device_data = *cmd_list->get_device()->get_private_data<DeviceData>();
      if (!device_data.game)
         return;
      auto& game_device_data = GetGameDeviceData(device_data);
      bool any_pending;
      {
         const std::lock_guard lock(game_device_data.rtao_pending_mutex);
         any_pending = !game_device_data.rtao_pending.empty();
      }
      if (!any_pending && !game_device_data.rtao_rt_ready)
         return;

      ID3D11DeviceChild* first = (ID3D11DeviceChild*)(cmd_list->get_native());
      ID3D11DeviceChild* second = (ID3D11DeviceChild*)(secondary_cmd_list->get_native());
      com_ptr<ID3D11DeviceContext> context;
      com_ptr<ID3D11CommandList> command_list;

      // ExecuteCommandList(): (immediate context, command list)
      if (SUCCEEDED(first->QueryInterface(&context)) && SUCCEEDED(second->QueryInterface(&command_list)))
      {
         if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
            return;
         GameDeviceDataDishonored2::RTAOPendingSplit pending;
         bool found = false;
         if (any_pending)
         {
            const std::lock_guard lock(game_device_data.rtao_pending_mutex);
            auto it = std::find_if(game_device_data.rtao_pending.begin(), game_device_data.rtao_pending.end(), [&](const auto& p) { return p.remainder_command_list == command_list.get(); });
            if (it != game_device_data.rtao_pending.end())
            {
               pending = std::move(*it);
               game_device_data.rtao_pending.erase(it);
               found = true;
            }
         }
         if (found)
         {
            context->ExecuteCommandList(pending.prefix.get(), TRUE);
            game_device_data.rtao_capture.Commit(pending.capture_records); // The prefix's draws have now executed
            std::string error;
            const bool ok = RTAOOutputIsRayTraced(pending.output) ? game_device_data.rtao.Run(context.get(), MakeRTAORecorder(game_device_data, pending.settings), error) : game_device_data.rtao.Run(context.get(), pending.constants, error);
            (ok ? game_device_data.rtao_stat_deferred_ok : game_device_data.rtao_stat_failed)++;
            if (!ok && !game_device_data.rtao_logged_run_error.exchange(true))
            {
               SetRTAOStatus("D3D12 pass failed (deferred): " + error, true); // The composite then reads the previous AO
            }
            game_device_data.rtao_capture.MarkAOPassDone();
         }
         // After the D3D12 pass: this list's own draws run after it (for the split list, the post SSAO remainder)
         game_device_data.rtao_capture.OnExecuteCommandList(command_list.get());
         return;
      }

      // FinishCommandList(): (command list, deferred context)
      context.reset();
      command_list.reset();
      if (SUCCEEDED(first->QueryInterface(&command_list)) && SUCCEEDED(second->QueryInterface(&context)))
      {
         game_device_data.rtao_capture.OnFinishCommandList(context.get(), command_list.get());
         const std::lock_guard lock(game_device_data.rtao_pending_mutex);
         for (auto& pending : game_device_data.rtao_pending)
         {
            if (pending.deferred_context == context.get() && pending.remainder_command_list == nullptr)
            {
               pending.remainder_command_list = command_list.get();
               game_device_data.rtao_stat_mapped++;
               break;
            }
         }
      }
   }

   //TODOFT: delete?
   void UpdateLumaInstanceDataCB(CB::LumaInstanceDataPadded& data, CommandListData& cmd_list_data, DeviceData& device_data) override
   {
   }

   bool UpdateGlobalCB(const void* global_buffer_data_ptr, reshade::api::device* device) override
   {
      const CBPerViewGlobals& global_buffer_data = *((const CBPerViewGlobals*)global_buffer_data_ptr);

      bool is_valid_cbuffer = true;

#if DEVELOPMENT && 0
      cb_per_view_globals.emplace_back(global_buffer_data);
      cb_per_view_globals_last_drawn_shader.emplace_back(last_drawn_shader != SHADER_HASH_NONE ? Shader::Hash_NumToStr(last_drawn_shader) : ""); // The shader hash could we unspecified if we didn't replace the shader
#endif // DEVELOPMENT

      if (!is_valid_cbuffer)
      {
         return false;
      }

      // Shadow maps and other things temporarily change the values in the global cbuffer,
      // like not use inverse depth (which affects the projection matrix, and thus many other matrices?),
      // use different render and output resolutions, etc etc.
      // We could also base our check on "CV_ProjRatio" (x and y) and "CV_FrustumPlaneEquation" and "CV_DecalZFightingRemedy" as these are also different for alternative views.
      // "CV_PrevViewProjMatr" is not a raw projection matrix when rendering shadow maps, so we can easily detect that.
      // Note: we can check if the matrix is identity to detect whether we are currently in a menu (the main menu?)
      bool is_custom_draw_version = !MatrixIsProjection(global_buffer_data.cb_projectionmatrix);

      if (is_custom_draw_version)
      {
         return false;
      }

      DeviceData& device_data = *device->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);

#if DEVELOPMENT
      std::thread::id new_global_cbuffer_thread_id = std::this_thread::get_id();
      // Make sure this cbuffer is always updated in the same thread (forever)
      if (global_cbuffer_thread_id != std::thread::id())
      {
         ASSERT_ONCE(global_cbuffer_thread_id == new_global_cbuffer_thread_id);
      }
      global_cbuffer_thread_id = new_global_cbuffer_thread_id;
#endif

      // Copy the temporary buffer ptr into our persistent data
      cb_per_view_global = global_buffer_data;

      auto current_projection_matrix = cb_per_view_global.cb_projectionmatrix;
      if (!game_device_data.has_drawn_scene_previous)
      {
         previous_projection_matrix = current_projection_matrix;
      }

      // Update our cached data with information from the cbuffer.
      // After vanilla tonemapping (as soon as AA starts),
      // camera jitters are removed from the cbuffer projection matrices, and the render resolution is also set to 100% (after the upscaling pass),
      // so we want to ignore these cases. We stop at the gbuffer compositions draw, because that's the last know cbuffer 13 to have the perfect values we are looking for (that shader is always run, so it's reliable)!
      // A lot of passes are drawn on scaled down render targets and the cbuffer values would have been updated to reflect that (e.g. "CV_ScreenSize"), so ignore these cases.
      if (!game_device_data.found_per_view_globals)
      {
#if DEVELOPMENT
         static float2 local_previous_render_resolution;
         if (!game_device_data.found_per_view_globals)
         {
            local_previous_render_resolution.x = device_data.output_resolution.x * cb_per_view_global.cb_resolutionscale.x;
            local_previous_render_resolution.y = device_data.output_resolution.y * cb_per_view_global.cb_resolutionscale.y;
         }
#endif // DEVELOPMENT

         //TODOFT: these have read/writes that are possibly not thread safe but they should never cause issues in actual usages of Prey
         device_data.render_resolution.x = device_data.output_resolution.x * cb_per_view_global.cb_resolutionscale.x; // TODO: round or floor?
         device_data.render_resolution.y = device_data.output_resolution.y * cb_per_view_global.cb_resolutionscale.y;

         auto previous_prey_drs_active = game_device_data.prey_drs_active.load();
         game_device_data.prey_drs_active = std::abs(device_data.render_resolution.x - device_data.output_resolution.x) >= 0.5f || std::abs(device_data.render_resolution.y - device_data.output_resolution.y) >= 0.5f;
         // Make sure this doesn't change within a frame (once we found DRS in a frame, we should never "lose" it again for that frame.
         // Ignore this when we have no shaders loaded as it would always break due to the "has_drawn_tonemapping" check failing.
         ASSERT_ONCE(device_data.cloned_pipeline_count == 0 || !game_device_data.found_per_view_globals || !previous_prey_drs_active || (previous_prey_drs_active == game_device_data.prey_drs_active));

#if DEVELOPMENT
         // Make sure that our rendering resolution doesn't change randomly within the pipeline (it probably will, it seems to trigger during quick save loads, maybe for the very first draw call to clear buffers)
         const float2 previous_render_resolution = local_previous_render_resolution;
         ASSERT_ONCE(!game_device_data.has_drawn_scene_previous || !game_device_data.found_per_view_globals || !game_device_data.prey_drs_detected || (AlmostEqual(device_data.render_resolution.x, previous_render_resolution.x, 0.25f) && AlmostEqual(device_data.render_resolution.y, previous_render_resolution.y, 0.25f)));
#endif // DEVELOPMENT

         // Once we detect the user enabled DRS, we can't ever know it's been disabled because the game only occasionally drops to lower rendering resolutions, so we couldn't know if it was ever disabled
         if (game_device_data.prey_drs_active)
         {
            game_device_data.prey_drs_detected = true;

            float resolution_scale = device_data.render_resolution.y / device_data.output_resolution.y;
            // Lower the DLSS quality mode (which might introduce a stutter, or a slight blurring of the image as it resets the history),
            // but this will make DLSS not use DLAA and instead fall back on a quality mode that allows for a dynamic range of resolutions.
            // This isn't the exact rend resolution DLSS will be forced to use, but the center of a range it's gonna expect.
            // Unfortunately DLSS has a limited range of accepted resolutions per quality mode, and if you go beyond it, it fails to render (until in range again),
            // thus, we need to make sure the automatic DRS range of Prey is within the same range!
            // We couldn't change this resolution scale every frame as it's make DLSS stutter massively.
            // See CryEngine "osm_fbMinScale" cvar (config), that drives the min rend res scale, the DLSS rend scale should ideally be set to the same value, but it's fine if it's above it, given it's the target "average" dynamic resolution.
            // If CryEngine ever went below 50% render scale, we force DLSS into ultra performance mode (33%), as the range allowed by quality mode (67%) can't go below 50%. There will be a stutter (and history reset?) every time we swap back and forth, but at least it works...
            if (resolution_scale < 0.5f - FLT_EPSILON)
            {
#if 1 // Unfortunately no quality mode with a res scale below 0.5 supports dynamic resolution scaling, so we are forced to change the quality mode every frame or so (or at least, every time Prey changes DRS value, which might further slow down the DRS detection mechanism...)
               device_data.sr_render_resolution_scale = resolution_scale;
#else // If we do this, DLSS would fail if any resolution that didn't exactly match 33% render scale was used by the game
               device_data.sr_render_resolution_scale = 1.f / 3.f;
#endif
            }
            else
            {
               // This should pick quality or balanced mode, with a range from 100% to 50% resolution scale
               device_data.sr_render_resolution_scale = 1.f / 1.5f;
            }
         }
         // Reset to DLAA and try again (once), given that we can't go from a 1/3 to a 1 rend scale (e.g. in case DRS was disabled in the menu)
         else if (false && device_data.sr_suppressed && device_data.sr_render_resolution_scale != 1.f)
         {
            device_data.sr_render_resolution_scale = 1.f;
            device_data.sr_suppressed = false;
         }

         // NOTE: we could just save the first one we found, it should always be jittered and "correct".
         projection_matrix = current_projection_matrix;

         const auto projection_jitters_copy = projection_jitters;

         // The matrix is transposed so we flip the matrix x and y indices.
         projection_jitters.x = current_projection_matrix(2, 0);
         projection_jitters.y = current_projection_matrix(2, 1);
         // TODO: cb_per_view_global.cb_jittervectors?

#if DEVELOPMENT
         ASSERT_ONCE(disable_taa_jitters || (projection_jitters_copy.x == 0 && projection_jitters_copy.y == 0) || (projection_jitters.x != 0 || projection_jitters.y != 0)); // Once we found jitters, we should never cache matrices that don't have jitters anymore
#endif

         bool prey_taa_active_copy = game_device_data.prey_taa_active;
         // This is a reliable check to tell whether TAA is enabled. Jitters are "never" zero if they are enabled:
         // they can be if we use the "srand" method, but it would happen one in a billion years;
         // they could also be zero with Halton if the frame index was reset to zero (it is every x frames), but that happens very rarely, and for one frame only (we have two frames as tolerance).
         game_device_data.prey_taa_active = (std::abs(projection_jitters.x * device_data.render_resolution.x) >= 0.00075) || (std::abs(projection_jitters.y * device_data.render_resolution.y) >= 0.00075); //TODOFT: make calculations more accurate (the threshold), especially with higher Halton phases!
#if DEVELOPMENT
         game_device_data.prey_taa_active = game_device_data.prey_taa_active || disable_taa_jitters;
#endif // DEVELOPMENT
         // Make sure that once we detect that TAA was active within a frame, then it should never be detected as off in the same frame (it would mean we are reading a bad cbuffer 13 that we should have discarded).
         // Ignore this when we have no shaders loaded as it would always break due to the "has_drawn_tonemapping" check failing.
         ASSERT_ONCE(device_data.cloned_pipeline_count == 0 || !game_device_data.found_per_view_globals || !prey_taa_active_copy || (prey_taa_active_copy == game_device_data.prey_taa_active));
         if (prey_taa_active_copy != game_device_data.prey_taa_active && game_device_data.has_drawn_scene_previous) // TAA changed
         {
            // Detect if TAA was ever detected as on/off/on or off/on/off over 3 frames, because if that was so, our jitter "length" detection method isn't solid enough and we should do more (or add more tolernace to it),
            // this might even happen every x hours once the randomization triggers specific enough values, though all TAA modes have a pretty short cycle with fixed jitters,
            // so it should either happen quickly or never.
            bool middle_value_different = (game_device_data.prey_taa_active == game_device_data.previous_prey_taa_active[0]) != (game_device_data.prey_taa_active == game_device_data.previous_prey_taa_active[1]);
            ASSERT_ONCE(!middle_value_different);
         }
         bool drew_sr = cb_luma_global_settings.SRType > 0; // If this was true, SR would have been enabled and probably drew
         device_data.taa_detected = game_device_data.prey_taa_active || game_device_data.previous_prey_taa_active[0]; // This one has a two frames tolerance. We let it persist even if the game stopped drawing the 3D scene.
         cb_luma_global_settings.SRType = (device_data.sr_type != SR::Type::None && !device_data.sr_suppressed && device_data.taa_detected) ? (uint(device_data.sr_type) + 1) : 0; // No need for "s_mutex_reshade" here, given that they are generally only also changed by the user manually changing the settings in ImGUI, which runs at the very end of the frame
         if (cb_luma_global_settings.SRType > 0 && !drew_sr)
         {
            device_data.cb_luma_global_settings_dirty = true;
            // Reset DLSS history when we toggle DLSS on and off manually, or when the user in the game changes the AA mode,
            // otherwise the history from the last time DLSS was active will be kept (DLSS doesn't know time passes since it was last used).
            // We could also clear DLSS resources here when we know it's unused for a while, but it would possibly lead to stutters.
            device_data.force_reset_sr = true;
         }

         if (!custom_texture_mip_lod_bias_offset)
         {
            std::shared_lock shared_lock_samplers(s_mutex_samplers);

            const auto prev_texture_mip_lod_bias_offset = device_data.texture_mip_lod_bias_offset;
            if (device_data.sr_type != SR::Type::None && !device_data.sr_suppressed && device_data.taa_detected && device_data.cloned_pipeline_count != 0)
            {
               device_data.texture_mip_lod_bias_offset = SR::GetMipLODBias(device_data.render_resolution.y, device_data.output_resolution.y); // This results in -1 at output res
            }
            else
            {
               // Reset to best fallback value.
               // This bias offset replaces the value from the game (see "samplers_upgrade_mode" 5), which was based on the "r_AntialiasingTSAAMipBias" cvar for most textures (it doesn't apply to all the ones that would benefit from it, and still applies to ones that exhibit moire patterns),
               // but only if TAA was engaged (not SMAA or SMAA+TAA) (it might persist on SMAA after once using TAA, due to a bug).
               // Prey defaults that to 0 but Luma's configs set it to -1.
               device_data.texture_mip_lod_bias_offset = device_data.taa_detected ? -1.f : 0.f;
            }
            const auto new_texture_mip_lod_bias_offset = device_data.texture_mip_lod_bias_offset;

            bool texture_mip_lod_bias_offset_changed = prev_texture_mip_lod_bias_offset != new_texture_mip_lod_bias_offset;
            // Re-create all samplers immediately here instead of doing it at the end of the frame.
            // This allows us to avoid possible (but very unlikely) hitches that could happen if we re-created a new sampler for a new resolution later on when samplers descriptors are set.
            // It also allows us to use the right samplers for this frame's resolution.
            if (texture_mip_lod_bias_offset_changed)
            {
               ID3D11Device* native_device = (ID3D11Device*)(device->get_native());
               for (auto& samplers_handle : device_data.custom_sampler_by_original_sampler)
               {
                  if (samplers_handle.second.contains(new_texture_mip_lod_bias_offset)) continue; // Skip "resolutions" that already got their custom samplers created
                  ID3D11SamplerState* native_sampler = reinterpret_cast<ID3D11SamplerState*>(samplers_handle.first);
                  shared_lock_samplers.unlock(); // This is fine!
                  {
                     D3D11_SAMPLER_DESC native_desc;
                     native_sampler->GetDesc(&native_desc);
                     com_ptr<ID3D11SamplerState> custom_sampler = CreateCustomSampler(device_data, native_device, native_desc);
                     const std::unique_lock unique_lock_samplers(s_mutex_samplers);
                     samplers_handle.second[new_texture_mip_lod_bias_offset] = custom_sampler;
                  }
                  shared_lock_samplers.lock();
               }
            }
         }

         if (!game_device_data.has_drawn_scene_previous)
         {
            device_data.previous_render_resolution = device_data.render_resolution;
            previous_projection_jitters = projection_jitters;

            // Set it to the latest value (ignoring the actual history)
            game_device_data.previous_prey_taa_active[0] = game_device_data.prey_taa_active;
            game_device_data.previous_prey_taa_active[1] = game_device_data.prey_taa_active;
         }

         game_device_data.found_per_view_globals = true;
      }

      return true;
   }

   void OnPresent(ID3D11Device* native_device, DeviceData& device_data) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      game_device_data.has_drawn_scene_previous = game_device_data.has_drawn_scene;
      game_device_data.has_drawn_scene = false;
      game_device_data.final_post_process_command_list = nullptr;

      //TODOFT: do this in the super?
      device_data.has_drawn_main_post_processing = false;

      device_data.taa_detected = true;
      device_data.has_drawn_sr = false;
      game_device_data.found_per_view_globals = false;

      // Drop deferred RTAO splits the game never executed (e.g. discarded command lists)
      game_device_data.rtao_frame++;
      if (game_device_data.rtao_capture.IsInitialized())
         game_device_data.rtao_capture.OnPresent();
      if (g_ao_view_mode != 0)
         DrawAOView(native_device, device_data, game_device_data);

      // The game's HBAO+ follows Luma's AO mode. Applied when the mode changes, and once when the first 3D scene
      // draws (after the game has set its own cvars from its settings). A change made elsewhere (e.g. the in game
      // AO menu) is adopted into Luma's mode instead of being fought.
      if (g_game_ao.valid)
      {
         const int observed = ReadGameHbao();
         const int observed_skip = ReadGameSkipSsao();
         const int effective_mode = GetEffectiveAOMode();
         const int desired = effective_mode == AO_MODE_HBAO ? 1 : 0;
         const bool boot_apply = !g_game_ao.boot_applied && game_device_data.has_drawn_scene_previous;
         if (g_ao_mode_apply_pending.exchange(false) || boot_apply)
         {
            if (observed != desired)
               g_game_ao.set_bool(g_game_ao.r_hbao, char(desired), 1);
            // Any AO but Off needs the engine's AO pass to run (Off leaves r_skipSSAO as it is)
            if (effective_mode != AO_MODE_OFF && observed_skip != 0)
               g_game_ao.set_bool(g_game_ao.r_skip_ssao, 0, 1);
            g_game_ao.applied = desired;
            g_game_ao.applied_skip = effective_mode != AO_MODE_OFF ? 0 : observed_skip;
            if (game_device_data.has_drawn_scene_previous)
               g_game_ao.boot_applied = true;
         }
         else if (g_game_ao.applied_skip != -1 && observed_skip != g_game_ao.applied_skip)
         {
            // The in game AO menu's "Off" sets r_skipSSAO 1 (and picking an AO there sets it back to 0)
            g_game_ao.applied_skip = observed_skip;
            if (observed_skip == 1 && g_ao_mode != AO_MODE_OFF)
            {
               g_ao_mode = AO_MODE_OFF;
               reshade::log::message(reshade::log::level::info, "AO: r_skipSSAO turned on in game, Luma AO mode -> Off");
            }
         }
         else if (g_game_ao.applied != -1 && observed != g_game_ao.applied)
         {
            g_game_ao.applied = observed;
            if (observed == 1)
               g_ao_mode = AO_MODE_HBAO;
            else if (g_ao_mode == AO_MODE_HBAO)
               g_ao_mode = AO_MODE_OFF;
            reshade::log::message(reshade::log::level::info, observed ? "AO: r_hbao turned on in game, Luma AO mode -> HBAO+" : "AO: r_hbao turned off in game, Luma AO mode -> Off");
         }
      }

      if (game_device_data.rtao_frame - game_device_data.rtao_last_draw_frame > 2)
         g_rtao_capture_active = false; // RTAO not drawing (other AO mode, menus): no capture cost
      {
         const std::lock_guard lock(game_device_data.rtao_pending_mutex);
         game_device_data.rtao_stat_dropped += uint32_t(std::erase_if(game_device_data.rtao_pending, [&](const auto& p) { return game_device_data.rtao_frame - p.frame > 3; }));
      }
      const uint32_t rtao_frames = game_device_data.rtao_frame - game_device_data.rtao_init_frame;
      if (game_device_data.rtao.IsInitialized() && (rtao_frames == 300 || rtao_frames == 3000 || rtao_frames == 30000))
      {
         char buffer[256];
         snprintf(buffer, sizeof(buffer), "RTAO: %u frames after init: inline passes %u, deferred splits %u, mapped %u, deferred passes %u, failed %u, dropped %u",
            rtao_frames, game_device_data.rtao_stat_inline_ok.load(), game_device_data.rtao_stat_splits.load(), game_device_data.rtao_stat_mapped.load(),
            game_device_data.rtao_stat_deferred_ok.load(), game_device_data.rtao_stat_failed.load(), game_device_data.rtao_stat_dropped.load());
         reshade::log::message(reshade::log::level::info, buffer);
         if (game_device_data.rtao_capture.IsInitialized())
         {
            const RTAO::CaptureStats stats = game_device_data.rtao_capture.GetStats();
            snprintf(buffer, sizeof(buffer), "RTAO: capture: draws %llu/%llu captured, meshes %llu (%llu with BLAS, %llu MB), arena %llu MB, TLAS %u, indirect draws %llu, unknown VS %llu, GPU %.2f ms",
               (unsigned long long)stats.draws_captured, (unsigned long long)stats.draws_seen, (unsigned long long)stats.meshes, (unsigned long long)stats.meshes_ready,
               (unsigned long long)(game_device_data.rtao_blas_bytes.load() >> 20), (unsigned long long)(stats.arena_vertex_bytes >> 20), game_device_data.rtao_tlas_instances.load(),
               (unsigned long long)g_rtao_indirect_draws.load(), (unsigned long long)g_rtao_draws_unknown_shader.load(), game_device_data.rtao_gpu_us.load() / 1000.0);
            reshade::log::message(reshade::log::level::info, buffer);
            std::string reasons = "RTAO: main view draws not captured:";
            {
               const std::lock_guard lock(g_rtao_skip_reason_mutex);
               for (const auto& pair : g_rtao_skip_reasons)
                  reasons += std::string(" ") + pair.first + " " + std::to_string(pair.second) + ",";
            }
            reasons += " not a triangle list " + std::to_string(g_rtao_draws_not_triangle_list.load()) + ", unusable input layout " + std::to_string(g_rtao_draws_unusable_layout.load()) +
               ", capture skips: format " + std::to_string(stats.skipped[RTAO::kSkipFormat]) + ", dynamic buffer " + std::to_string(stats.skipped[RTAO::kSkipBuffer]) + ", binding " + std::to_string(stats.skipped[RTAO::kSkipBinding]) +
               ", other " + std::to_string(stats.skipped[RTAO::kSkipOther]) + ", drawn after the AO pass " + std::to_string(stats.after_ao_records);
            reshade::log::message(reshade::log::level::info, reasons.c_str());
         }
      }
   }
   
   DrawOrDispatchOverrideType OnDrawOrDispatch(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      D3D11_TEXTURE2D_DESC depth_desc; // TODO
#if 1
      if (!game_device_data.has_drawn_scene && original_shader_hashes.Contains(shader_hashes_DownsampleDepth))
      {
         game_device_data.has_drawn_scene = true;
         //game_device_data.has_drawn_post_process = true; //TODOFT: already done later?

         com_ptr<ID3D11ShaderResourceView> depth_srv;
         native_device_context->CSGetShaderResources(0, 1, &depth_srv);

         com_ptr<ID3D11UnorderedAccessView> depth_uav;
         native_device_context->CSGetUnorderedAccessViews(3, 1, &depth_uav);

         game_device_data.depth_buffer = nullptr;
         if (depth_srv)
         {
            depth_srv->GetResource(&game_device_data.depth_buffer);

            com_ptr<ID3D11Texture2D> depth_2d;
            HRESULT hr = game_device_data.depth_buffer->QueryInterface(&depth_2d);
            ASSERT_ONCE(SUCCEEDED(hr));
            depth_2d->GetDesc(&depth_desc);
         }
      }
#else
      if (!game_device_data.has_drawn_scene && original_shader_hashes.Contains(shader_hashes_UnprojectDepth) && false)
      {
         game_device_data.has_drawn_scene = true;

         com_ptr<ID3D11ShaderResourceView> depth_opaque_srv;
         native_device_context->CSGetShaderResources(1, 1, &depth_opaque_srv);
         com_ptr<ID3D11ShaderResourceView> depth_alpha_srv;
         native_device_context->CSGetShaderResources(0, 1, &depth_alpha_srv);

         com_ptr<ID3D11UnorderedAccessView> depth_uav;
         native_device_context->CSGetUnorderedAccessViews(0, 1, &depth_uav); // float4

         game_device_data.depth_buffer = nullptr;
         if (depth_opaque_srv)
         {
            depth_opaque_srv->GetResource(&game_device_data.depth_buffer);

            com_ptr<ID3D11Texture2D> depth_2d;
            HRESULT hr = game_device_data.depth_buffer->QueryInterface(&depth_2d);
            ASSERT_ONCE(SUCCEEDED(hr));
            depth_2d->GetDesc(&depth_desc);
         }
      }
#endif

      if (g_ao_view_mode == 1 && original_shader_hashes.Contains(shader_hashes_AODenoise))
      {
         com_ptr<ID3D11RenderTargetView> rtv;
         native_device_context->OMGetRenderTargets(1, &rtv, nullptr);
         if (rtv)
         {
            com_ptr<ID3D11Resource> resource;
            rtv->GetResource(&resource);
            const std::lock_guard lock(game_device_data.ao_view_mutex);
            game_device_data.ao_view_resource = resource; // The last one of the frame is the final AO
         }
      }

      if (original_shader_hashes.Contains(shader_hashes_SSAO))
      {
         // Depth SRV is PS t0, the per view cbuffer PS b1, the AO render target RT0.
         int ao_mode = GetEffectiveAOMode();
         if (ao_mode == AO_MODE_RTAO)
         {
            if (DrawRTAO(native_device, native_device_context, device_data, game_device_data, original_draw_dispatch_func))
            {
               return DrawOrDispatchOverrideType::Replaced;
            }
            // RTAO became unavailable: fall back to XeGTAO this same frame. Transient failures draw vanilla SSAO.
            if (g_rtao_unavailable)
               ao_mode = GetEffectiveAOMode();
         }

         if (ao_mode == AO_MODE_XEGTAO)
         {
            ComPtr<ID3D11ShaderResourceView> srv_original;
            native_device_context->PSGetShaderResources(0, 1, srv_original.put());
            ComPtr<ID3D11Buffer> cb_original;
            native_device_context->PSGetConstantBuffers(1, 1, cb_original.put());
            ComPtr<ID3D11SamplerState> smp_original;
            native_device_context->PSGetSamplers(0, 1, smp_original.put());

            // We have to manually set the LumaSettings CB even it's suposed to be already set.
            SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel | reshade::api::shader_stage::compute, LumaConstantBufferType::LumaSettings);

            // XeGTAOPrefilterDepths16x16 pass
            //

            D3D11_TEXTURE2D_DESC tex_desc = {};
            tex_desc.Width = device_data.render_resolution.x;
            tex_desc.Height = device_data.render_resolution.y;
            tex_desc.MipLevels = XE_GTAO_DEPTH_MIP_LEVELS;
            tex_desc.ArraySize = 1;
            tex_desc.Format = DXGI_FORMAT_R32_FLOAT;
            tex_desc.SampleDesc.Count = 1;
            tex_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

            // Create prefilter depths views.
            ComPtr<ID3D11Texture2D> tex;
            ensure(native_device->CreateTexture2D(&tex_desc, nullptr, tex.put()), >= 0);
            std::array<ID3D11UnorderedAccessView*, XE_GTAO_DEPTH_MIP_LEVELS> uav_prefilter_depths = {};
            D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
            uav_desc.Format = tex_desc.Format;
            uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            for (int i = 0; i < uav_prefilter_depths.size(); ++i)
            {
               uav_desc.Texture2D.MipSlice = i;
               ensure(native_device->CreateUnorderedAccessView(tex.get(), &uav_desc, &uav_prefilter_depths[i]), >= 0);
            }
            ComPtr<ID3D11ShaderResourceView> srv_prefilter_depths;
            ensure(native_device->CreateShaderResourceView(tex.get(), nullptr, srv_prefilter_depths.put()), >= 0);

            // Bindings.
            native_device_context->CSSetUnorderedAccessViews(0, uav_prefilter_depths.size(), uav_prefilter_depths.data(), nullptr);
            native_device_context->CSSetShader(device_data.native_compute_shaders.at(CompileTimeStringHash("DS2 XeGTAO Prefilter Depths CS")).get(), nullptr, 0);
            native_device_context->CSSetConstantBuffers(1, 1, &cb_original);
            const std::array smps = { device_data.sampler_state_point.get() };
            native_device_context->CSSetSamplers(0, smps.size(), smps.data());
            native_device_context->CSSetShaderResources(0, 1, &srv_original);

            native_device_context->Dispatch((tex_desc.Width + 16 - 1) / 16, (tex_desc.Height + 16 - 1) / 16, 1);

            // Unbind and release uav_prefilter_depths.
            static constexpr std::array<ID3D11UnorderedAccessView*, uav_prefilter_depths.size()> uav_nulls = {};
            native_device_context->CSSetUnorderedAccessViews(0, uav_nulls.size(), uav_nulls.data(), nullptr);
            auto release_com_array = [](auto& array){ for (auto* p : array) if (p) p->Release(); };
            release_com_array(uav_prefilter_depths);

            // XeGTAOMainPass pass
            //
            // We will render to the original RT.
            //

            // Bindings.
            native_device_context->PSSetShader(device_data.native_pixel_shaders.at(CompileTimeStringHash("DS2 XeGTAO Main Pass PS")).get(), nullptr, 0);
            native_device_context->PSSetSamplers(0, smps.size(), smps.data());
            native_device_context->PSSetShaderResources(0, 1, &srv_prefilter_depths);

            // Should match the original draw call. Just call the original instead?
            native_device_context->Draw(3, 0);

            //

            // We won't do denoise here, will rely on game's denoiser.

            return DrawOrDispatchOverrideType::Replaced;            
         }

         return DrawOrDispatchOverrideType::None;
      }

#if ENABLE_SR
      // Native TAA only fills the render-resolution region, but SR has written a full-resolution
      // result to the same texture. Tonemap is the first confirmed consumer of that result, so it
      // must rasterize over the full target for its SV_POSITION Load to address all SR pixels.
      if (cb_luma_global_settings.SRType > 0 && original_shader_hashes.Contains(shader_hashes_Tonemap))
      {
         SetViewportFullscreen(native_device_context);
      }
#endif

      if (!device_data.has_drawn_main_post_processing && original_shader_hashes.Contains(shader_hashes_UpscaleSharpen))
      {
#if ENABLE_SR
         // All preceding post-processing still uses render-resolution resources. The game's native
         // upscale/copy pass is the resolution-transition point, so make only this output full size.
         if (cb_luma_global_settings.SRType > 0)
         {
            SetViewportFullscreen(native_device_context);
         }
#endif

         if (native_device_context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
         {
            game_device_data.final_post_process_command_list = native_device_context;
         }
         else
         {
            device_data.has_drawn_main_post_processing = true;
         }
      }

      if (original_shader_hashes.Contains(shader_hashes_TAA))
      {
         // Not thread safe?
         device_data.cb_per_view_global_buffer = nullptr;
         native_device_context->CSGetConstantBuffers(1, 1, &device_data.cb_per_view_global_buffer);
         device_data.taa_detected = true;
      }

   #if ENABLE_SR
      // SR/TAA
      if (device_data.sr_type != SR::Type::None && !device_data.sr_suppressed && original_shader_hashes.Contains(shader_hashes_TAA))
      {
         native_device_context->CSGetConstantBuffers(1, 1, game_device_data.cb_taa_b1.put());
         native_device_context->CSGetConstantBuffers(2, 1, game_device_data.cb_taa_b2.put());
         native_device_context->CSGetShaderResources(3, 1, game_device_data.srv_ro_postfx_luminance_buffautoexposure.put());

         // TODO: Clean up all this, I think game will always use deferred rendering, so most of this is not needed.
         assert(native_device_context->GetType() == D3D11_DEVICE_CONTEXT_DEFERRED);

         com_ptr<ID3D11ShaderResourceView> srvs[2]; // TODO: rename
         // 1 motion vectors
         // 2 color source (pre TAA, jittered)
         native_device_context->CSGetShaderResources(1, ARRAYSIZE(srvs), reinterpret_cast<ID3D11ShaderResourceView**>(srvs));

         com_ptr<ID3D11UnorderedAccessView> uav;
         native_device_context->CSGetUnorderedAccessViews(1, 1, &uav);

         const bool dlss_inputs_valid = srvs[0].get() != nullptr && srvs[1].get() != nullptr && uav.get() != nullptr; // We don't check for "device_data.depth_buffer" here yet as this can happen before in a deferred context
         ASSERT_ONCE(dlss_inputs_valid);
         if (dlss_inputs_valid)
         {
            auto* sr_instance_data = device_data.GetSRInstanceData();
            ASSERT_ONCE(sr_instance_data);

            com_ptr<ID3D11Resource> output_color_resource;
            uav->GetResource(&output_color_resource);
            com_ptr<ID3D11Texture2D> output_color;
            HRESULT hr = output_color_resource->QueryInterface(&output_color);
            ASSERT_ONCE(SUCCEEDED(hr));

            D3D11_TEXTURE2D_DESC taa_output_texture_desc;
            output_color->GetDesc(&taa_output_texture_desc);

   #if FORCE_SMAA_MIPS // Define from the native plugin (not ever defined here!)
            ASSERT_ONCE(taa_output_texture_desc.MipLevels > 1); // To improve "Perfect Perspective" lens distortion
   #endif

            ASSERT_ONCE(std::lrintf(device_data.output_resolution.x) == taa_output_texture_desc.Width && std::lrintf(device_data.output_resolution.y) == taa_output_texture_desc.Height);
            std::array<uint32_t, 2> dlss_render_resolution = FindClosestIntegerResolutionForAspectRatio((double)taa_output_texture_desc.Width * (double)device_data.sr_render_resolution_scale, (double)taa_output_texture_desc.Height * (double)device_data.sr_render_resolution_scale, (double)taa_output_texture_desc.Width / (double)taa_output_texture_desc.Height);
            

            bool delay_dlss = native_device_context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE;
            //native_device->GetImmediateContext(&native_device_context);

            // TODO: we could do this async from the beginning of rendering (when we can detect res changes), to here, with a mutex, to avoid potential stutters when DRS first engages (same with creating DLSS textures?) or changes resolution? (we could allow for creating more than one DLSS feature???)
            // 
            // Our DLSS implementation picks a quality mode based on a fixed rendering resolution, but we scale it back in case we detected the game is running DRS, otherwise we run DLAA.
            // At lower quality modes (non DLAA), DLSS actually seems to allow for a wider input resolution range that it actually claims when queried for it, but if we declare a resolution scale below 50% here, we can get an hitch,
            // still, DLSS will keep working at any input resolution (or at least with a pretty big tolerance range).
            // This function doesn't alter the pipeline state (e.g. shaders, cbuffers, RTs, ...), if not, we need to move it to the "Present()" function, it doesn't seem like we can do it async though (DLSS rendering crashes in deferred context, possibly this would too)
            if (!delay_dlss)
            {
               SR::SettingsData settings_data;
               settings_data.output_width = taa_output_texture_desc.Width;
               settings_data.output_height = taa_output_texture_desc.Height;
               settings_data.render_width = dlss_render_resolution[0];
               settings_data.render_height = dlss_render_resolution[1];
               settings_data.dynamic_resolution = game_device_data.prey_drs_detected;
               settings_data.hdr = true; // The "HDR" flag in DLSS SR actually means whether the color is in linear space or "sRGB gamma" (apparently not 2.2) (SDR) space, colors beyond 0-1 don't seem to be clipped either way
               settings_data.inverted_depth = true;
               settings_data.mvs_jittered = false;
               settings_data.render_preset = dlss_render_preset;
               sr_implementations[device_data.sr_type]->UpdateSettings(sr_instance_data, native_device_context, settings_data);
            }

            bool skip_dlss = taa_output_texture_desc.Width < sr_instance_data->min_resolution || taa_output_texture_desc.Height < sr_instance_data->min_resolution;
            bool dlss_output_changed = false;
            constexpr bool dlss_use_native_uav = true;
            bool dlss_output_supports_uav = dlss_use_native_uav && (taa_output_texture_desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) != 0;
            if (!dlss_output_supports_uav)
            {
               ASSERT_ONCE(!dlss_use_native_uav); // Should never happen anymore

               taa_output_texture_desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;

               if (device_data.sr_output_color.get())
               {
                  D3D11_TEXTURE2D_DESC dlss_taa_output_texture_desc;
                  device_data.sr_output_color->GetDesc(&dlss_taa_output_texture_desc);
                  dlss_output_changed = dlss_taa_output_texture_desc.Width != taa_output_texture_desc.Width || dlss_taa_output_texture_desc.Height != taa_output_texture_desc.Height || dlss_taa_output_texture_desc.Format != taa_output_texture_desc.Format;
               }
               if (!device_data.sr_output_color.get() || dlss_output_changed)
               {
                  device_data.sr_output_color = nullptr; // Make sure we discard the previous one
                  hr = native_device->CreateTexture2D(&taa_output_texture_desc, nullptr, &device_data.sr_output_color);
                  ASSERT_ONCE(SUCCEEDED(hr));
               }
               if (!device_data.sr_output_color.get())
               {
                  skip_dlss = true;
               }
            }
            else
            {
               ASSERT_ONCE(device_data.sr_output_color == nullptr);
               device_data.sr_output_color = output_color;
            }

            if (!skip_dlss)
            {
               game_device_data.sr_source_color = nullptr;
               game_device_data.sr_motion_vectors = nullptr;
               srvs[1]->GetResource(&game_device_data.sr_source_color);
               srvs[0]->GetResource(&game_device_data.sr_motion_vectors);

               // TODO
               com_ptr<ID3D11Texture2D> source_color_2d;
               HRESULT hr = game_device_data.sr_source_color->QueryInterface(&source_color_2d);
               ASSERT_ONCE(SUCCEEDED(hr));
               D3D11_TEXTURE2D_DESC source_color_2d_desc;
               source_color_2d->GetDesc(&source_color_2d_desc);
               com_ptr<ID3D11Texture2D> motion_vectors_2d;
               hr = game_device_data.sr_motion_vectors->QueryInterface(&motion_vectors_2d);
               ASSERT_ONCE(SUCCEEDED(hr));
               D3D11_TEXTURE2D_DESC motion_vectors_2d_desc;
               motion_vectors_2d->GetDesc(&motion_vectors_2d_desc);

               // Generate "fake" exposure texture
               bool exposure_changed = false;
               float sr_scene_exposure = device_data.sr_scene_exposure;
   #if DEVELOPMENT
               if (sr_custom_exposure > 0.f)
               {
                  sr_scene_exposure = sr_custom_exposure;
               }
   #endif // DEVELOPMENT
               exposure_changed = sr_scene_exposure != device_data.sr_exposure_texture_value;
               device_data.sr_exposure_texture_value = sr_scene_exposure;
               // TODO: optimize this for the "SR_RELATIVE_PRE_EXPOSURE" false case! Avoid re-creating the texture every frame the exposure changes and instead make it dynamic and re-write it from the CPU? Or simply make our exposure calculation shader write to a texture directly
               // (though in that case it wouldn't have the same delay as the CPU side pre-exposure buffer readback)
               if (!device_data.sr_exposure.get() || exposure_changed)
               {
                  D3D11_TEXTURE2D_DESC exposure_texture_desc; // DLSS fails if we pass in a 1D texture so we have to make a 2D one
                  exposure_texture_desc.Width = 1;
                  exposure_texture_desc.Height = 1;
                  exposure_texture_desc.MipLevels = 1;
                  exposure_texture_desc.ArraySize = 1;
                  exposure_texture_desc.Format = DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT; // FP32 just so it's easier to initialize data for it
                  exposure_texture_desc.SampleDesc.Count = 1;
                  exposure_texture_desc.SampleDesc.Quality = 0;
                  exposure_texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
                  exposure_texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                  exposure_texture_desc.CPUAccessFlags = 0;
                  exposure_texture_desc.MiscFlags = 0;

                  // It's best to force an exposure of 1 given that DLSS runs after the auto exposure is applied (in tonemapping).
                  // Theoretically knowing the average exposure of the frame would still be beneficial to it (somehow) so maybe we could simply let the auto exposure in,
                  D3D11_SUBRESOURCE_DATA exposure_texture_data;
                  exposure_texture_data.pSysMem = &sr_scene_exposure; // This needs to be "static" data in case the texture initialization was somehow delayed and read the data after the stack destroyed it
                  exposure_texture_data.SysMemPitch = sizeof(float);
                  exposure_texture_data.SysMemSlicePitch = sizeof(float);

                  device_data.sr_exposure = nullptr; // Make sure we discard the previous one
                  hr = native_device->CreateTexture2D(&exposure_texture_desc, &exposure_texture_data, &device_data.sr_exposure);
                  assert(SUCCEEDED(hr));
               }

               // Reset DLSS history if we did not draw motion blur (and we previously did). Based on CryEngine source code, mb is skipped on the first frame after scene cuts, so we want to re-use that information (this works even if MB was disabled).
               // Reset DLSS history if for one frame we had stopped tonemapping. This might include some scene cuts, but also triggers when entering full screen UI menus or videos and then leaving them (it shouldn't be a problem).
               // Reset DLSS history if the output resolution or format changed (just an extra safety mechanism, it might not actually be needed).
               bool reset_dlss = device_data.force_reset_sr || dlss_output_changed;
               if (!delay_dlss)
               {
                  device_data.force_reset_sr = false;
               }

               uint32_t render_width_dlss = std::lrintf(device_data.render_resolution.x);
               uint32_t render_height_dlss = std::lrintf(device_data.render_resolution.y);

               // These configurations store the image already multiplied by paper white from the beginning of tonemapping, including at the time DLSS runs.
               // The other configurations run DLSS in "SDR" Gamma Space so we couldn't safely change the exposure.
               const bool dlss_use_paper_white_pre_exposure = false; // Not needed for DH2, DLSS is before TM
               //const bool dlss_use_paper_white_pre_exposure = GetShaderDefineCompiledNumericalValue(POST_PROCESS_SPACE_TYPE_HASH) >= 1;

               float dlss_pre_exposure = 0.f; // 0 means it's ignored
               if (dlss_use_paper_white_pre_exposure)
               {
   #if 1 // Alternative that considers a value of 1 in the DLSS color textures to match the SDR output nits range (whatever that is)
                  dlss_pre_exposure = cb_luma_global_settings.ScenePaperWhite / default_paper_white;
   #else // Alternative that considers a value of 1 in the DLSS color textures to match 203 nits
                  dlss_pre_exposure = cb_luma_global_settings.ScenePaperWhite / srgb_white_level;
   #endif
                  dlss_pre_exposure *= device_data.sr_scene_pre_exposure;
               }
   #if DEVELOPMENT
               if (sr_custom_pre_exposure > 0.f)
                  dlss_pre_exposure = sr_custom_pre_exposure;
   #endif

               // Clean up random stuff to default values and copy resources to make sure it's all clear
               //dlss_pre_exposure = 1.0;
               //device_data.sr_exposure = nullptr;
               ////game_device_data.depth_buffer = nullptr;
               //reset_dlss = false;
               //projection_jitters.x = 0;
               //projection_jitters.y = 0;
               //device_data.sr_output_color_2 = CloneTexture<ID3D11Texture2D>(cmd_list, device_data.sr_output_color.get(), D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS, 0, false);
               //device_data.source_color_2 = CloneTexture<ID3D11Texture2D>(cmd_list, sr_source_color.get(), D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS, 0, false);
               //device_data.motion_vectors_2 = CloneTexture<ID3D11Texture2D>(cmd_list, sr_motion_vectors.get(), D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS, 0, false);
               //game_device_data.sr_source_color = (ID3D11Texture2D*)sr_source_color.get();
               //game_device_data.sr_motion_vectors = (ID3D11Texture2D*)sr_motion_vectors.get();

               // There doesn't seem to be a need to restore the DX state to whatever we had before (e.g. render targets, cbuffers, samplers, UAVs, texture shader resources, viewport, scissor rect, ...), CryEngine always sets everything it needs again for every pass.
               // DLSS internally keeps its own frames history, we don't need to do that ourselves (by feeding in an output buffer that was the previous frame's output, though we do have that if needed, it should be in srvs[1]).
               SR::SuperResolutionImpl::DrawData draw_data;
               draw_data.source_color = game_device_data.sr_source_color.get();
               draw_data.output_color = device_data.sr_output_color.get();
               draw_data.motion_vectors = game_device_data.sr_motion_vectors.get();
               draw_data.depth_buffer = game_device_data.depth_buffer.get();
               draw_data.exposure = device_data.sr_exposure.get();
               draw_data.pre_exposure = dlss_pre_exposure;
               draw_data.jitter_x = projection_jitters.x;
               draw_data.jitter_y = projection_jitters.y;
               draw_data.reset = reset_dlss;
               draw_data.render_width = render_width_dlss;
               draw_data.render_height = render_height_dlss;

               if (!delay_dlss && sr_implementations[device_data.sr_type]->Draw(sr_instance_data, native_device_context, draw_data))
               {
                  device_data.has_drawn_sr = true;
                  game_device_data.sr_deferred_command_list = native_device_context;
               }

               if (!delay_dlss)
               {
                  game_device_data.sr_source_color = nullptr;
                  game_device_data.sr_motion_vectors = nullptr;
               }

               if (device_data.has_drawn_sr)
               {
                  if (!dlss_output_supports_uav)
                  {
                     native_device_context->CopyResource(output_color.get(), device_data.sr_output_color.get()); // DX11 doesn't need barriers
                  }
                  // In this case it's not our buisness to keep alive this "external" texture
                  else
                  {
                     device_data.sr_output_color = nullptr;
                  }

                  return DrawOrDispatchOverrideType::Replaced; // "Cancel" the previously set draw call, DLSS has taken care of it
               }
               // DLSS Failed, suppress it for this frame and fall back on SMAA/TAA, hoping that anything before would have been rendered correctly for it already (otherwise it will start being correct in the next frame, given we suppress it (until manually toggled again, given that it'd likely keep failing))
               else if (!delay_dlss)
               {
                  ASSERT_ONCE(false);
                  cb_luma_global_settings.SRType = 0;
                  device_data.cb_luma_global_settings_dirty = true;
                  device_data.sr_suppressed = true;
                  device_data.force_reset_sr = true; // We missed frames so it's good to do this, it might also help prevent further errors
               }
               else // "delay_dlss"
               {
                  // Seal and replay all commands preceding TAA before SR. Otherwise the immediate
                  // context sees stale or only partially initialized SR inputs when DLSS runs.
                  HRESULT finish_hr = native_device_context->FinishCommandList(TRUE, game_device_data.sr_partial_command_list.put());
                  ASSERT_ONCE(SUCCEEDED(finish_hr));
                  game_device_data.sr_deferred_command_list = native_device_context;
                  return DrawOrDispatchOverrideType::Skip;
               }
            }
            if (dlss_output_supports_uav)
            {
               device_data.sr_output_color = nullptr;
            }
         }
      }
   #endif // ENABLE_SR
      return DrawOrDispatchOverrideType::None; // Return true to cancel this draw call
   }

   // ---- RTAO geometry capture hooks

   static void OnRTAOInitPipeline(reshade::api::device* device, reshade::api::pipeline_layout, uint32_t subobject_count, const reshade::api::pipeline_subobject* subobjects, reshade::api::pipeline pipeline)
   {
      if (device->get_api() != reshade::api::device_api::d3d11)
         return;
      for (uint32_t i = 0; i < subobject_count; i++)
      {
         const auto& subobject = subobjects[i];
         if (subobject.type == reshade::api::pipeline_subobject_type::vertex_shader && subobject.data)
         {
            const auto* desc = static_cast<const reshade::api::shader_desc*>(subobject.data);
            if (!desc->code || desc->code_size == 0)
               continue;
            RTAO::VertexShaderInfo info;
            RTAO::ReflectVertexShader(desc->code, desc->code_size, info);
            g_rtao_vs_by_source[(std::min)(uint32_t(info.source), 4u)]++;
            const std::unique_lock lock(g_rtao_shader_mutex);
            g_rtao_vertex_shaders[pipeline.handle] = info;
         }
         else if (subobject.type == reshade::api::pipeline_subobject_type::input_layout && subobject.data)
         {
            const auto* elements = static_cast<const reshade::api::input_element*>(subobject.data);
            std::vector<D3D11_INPUT_ELEMENT_DESC> descs(subobject.count);
            for (uint32_t j = 0; j < subobject.count; j++)
            {
               descs[j].SemanticName = elements[j].semantic;
               descs[j].SemanticIndex = elements[j].semantic_index;
               descs[j].Format = DXGI_FORMAT(elements[j].format);
               descs[j].InputSlot = elements[j].buffer_binding;
               descs[j].AlignedByteOffset = elements[j].offset;
               descs[j].InputSlotClass = elements[j].instance_step_rate > 0 ? D3D11_INPUT_PER_INSTANCE_DATA : D3D11_INPUT_PER_VERTEX_DATA;
               descs[j].InstanceDataStepRate = elements[j].instance_step_rate;
            }
            const RTAO::InputLayoutInfo info = RTAO::ParseInputLayout(descs.data(), uint32_t(descs.size()));
            const std::unique_lock lock(g_rtao_shader_mutex);
            g_rtao_input_layouts[pipeline.handle] = info;
         }
      }
   }

   static void OnRTAODestroyPipeline(reshade::api::device* device, reshade::api::pipeline pipeline)
   {
      if (device->get_api() != reshade::api::device_api::d3d11)
         return;
      const std::unique_lock lock(g_rtao_shader_mutex);
      g_rtao_vertex_shaders.erase(pipeline.handle);
      g_rtao_input_layouts.erase(pipeline.handle);
   }

   // Is the bound depth target the main view's, with depth writes on? (Shadow maps and other views use other targets.)
   static bool IsMainViewDepthWrite(ID3D11DeviceContext* context, bool depth_only)
   {
      com_ptr<ID3D11RenderTargetView> rtv;
      com_ptr<ID3D11DepthStencilView> dsv;
      context->OMGetRenderTargets(1, &rtv, &dsv);
      if (!dsv || (depth_only && rtv))
         return false;
      com_ptr<ID3D11Resource> resource;
      dsv->GetResource(&resource);
      com_ptr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource->QueryInterface(&texture)))
         return false;
      D3D11_TEXTURE2D_DESC desc;
      texture->GetDesc(&desc);
      if (desc.Width != g_rtao_main_depth_width.load(std::memory_order_relaxed) || desc.Height != g_rtao_main_depth_height.load(std::memory_order_relaxed))
         return false;
      com_ptr<ID3D11DepthStencilState> state;
      UINT stencil_ref = 0;
      context->OMGetDepthStencilState(&state, &stencil_ref);
      if (state) // Null is the default state: depth test and writes on
      {
         D3D11_DEPTH_STENCIL_DESC state_desc;
         state->GetDesc(&state_desc);
         if (!state_desc.DepthEnable || state_desc.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ALL)
            return false;
      }
      return true;
   }

   static bool OnRTAODrawIndexed(reshade::api::command_list* cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
   {
      if (!g_rtao_capture_active.load(std::memory_order_relaxed))
         return false;
      auto* device = cmd_list->get_device();
      if (device->get_api() != reshade::api::device_api::d3d11)
         return false;
      auto* context = reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native());
      if (!IsMainViewDepthWrite(context, g_rtao_capture_pass == 0))
         return false;
      D3D11_PRIMITIVE_TOPOLOGY topology;
      context->IAGetPrimitiveTopology(&topology);
      if (topology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
      {
         g_rtao_draws_not_triangle_list++;
         return false;
      }

      com_ptr<ID3D11VertexShader> vs;
      context->VSGetShader(&vs, nullptr, nullptr);
      com_ptr<ID3D11InputLayout> layout;
      context->IAGetInputLayout(&layout);
      RTAO::VertexShaderInfo vs_info;
      RTAO::InputLayoutInfo layout_info;
      {
         const std::shared_lock lock(g_rtao_shader_mutex);
         auto vs_it = g_rtao_vertex_shaders.find(uint64_t(uintptr_t(vs.get())));
         if (vs_it == g_rtao_vertex_shaders.end())
         {
            g_rtao_draws_unknown_shader++;
            return false;
         }
         vs_info = vs_it->second;
         auto layout_it = g_rtao_input_layouts.find(uint64_t(uintptr_t(layout.get())));
         if (layout_it != g_rtao_input_layouts.end())
            layout_info = layout_it->second;
      }
      g_rtao_draws_by_source[(std::min)(uint32_t(vs_info.source), 4u)]++;
      if (vs_info.source == RTAO::MatrixSource::None)
      {
         const std::lock_guard lock(g_rtao_skip_reason_mutex);
         g_rtao_skip_reasons[vs_info.skip_reason ? vs_info.skip_reason : "unknown"]++;
         return false;
      }
      if (!layout_info.has_position || layout_info.position_per_instance)
      {
         g_rtao_draws_unusable_layout++;
         return false;
      }

      RTAO::DrawInfo draw;
      draw.vs = &vs_info;
      draw.position_format = layout_info.position_format;
      draw.position_offset = layout_info.position_offset;
      draw.index_count = index_count;
      draw.first_index = first_index;
      draw.base_vertex = vertex_offset;
      draw.instance_count = instance_count;
      com_ptr<ID3D11Buffer> vertex_buffer, index_buffer, instance_cb;
      com_ptr<ID3D11ShaderResourceView> instance_buffer, instance_index;
      UINT stride = 0, offset = 0;
      context->IAGetVertexBuffers(layout_info.position_slot, 1, &vertex_buffer, &stride, &offset);
      DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
      UINT index_offset = 0;
      context->IAGetIndexBuffer(&index_buffer, &index_format, &index_offset);
      if (vs_info.instance_cb_slot != RTAO::kNoOffset)
      {
         com_ptr<ID3D11DeviceContext1> context1;
         UINT first_constant = 0, num_constants = 0;
         if (SUCCEEDED(context->QueryInterface(&context1)))
            context1->VSGetConstantBuffers1(vs_info.instance_cb_slot, 1, &instance_cb, &first_constant, &num_constants);
         else
            context->VSGetConstantBuffers(vs_info.instance_cb_slot, 1, &instance_cb);
         draw.instance_cb_first_constant = first_constant;
         draw.instance_cb_num_constants = num_constants;
      }
      com_ptr<ID3D11Buffer> per_view_cb;
      if (vs_info.source == RTAO::MatrixSource::MvpConstantBuffer && vs_info.per_view_cb_slot != RTAO::kNoOffset)
      {
         com_ptr<ID3D11DeviceContext1> context1;
         UINT first_constant = 0, num_constants = 0;
         if (SUCCEEDED(context->QueryInterface(&context1)))
            context1->VSGetConstantBuffers1(vs_info.per_view_cb_slot, 1, &per_view_cb, &first_constant, &num_constants);
         else
            context->VSGetConstantBuffers(vs_info.per_view_cb_slot, 1, &per_view_cb);
         draw.per_view_cb_first_constant = first_constant;
         draw.per_view_cb_num_constants = num_constants;
      }
      if (vs_info.instance_buffer_slot != RTAO::kNoOffset)
         context->VSGetShaderResources(vs_info.instance_buffer_slot, 1, &instance_buffer);
      if (vs_info.instance_index_slot != RTAO::kNoOffset)
         context->VSGetShaderResources(vs_info.instance_index_slot, 1, &instance_index);
      draw.vertex_buffer = vertex_buffer.get();
      draw.vertex_buffer_offset = offset;
      draw.vertex_stride = stride;
      draw.index_buffer = index_buffer.get();
      draw.index_format = index_format;
      draw.index_buffer_offset = index_offset;
      draw.instance_cb = instance_cb.get();
      draw.instance_buffer = instance_buffer.get();
      draw.instance_index = instance_index.get();
      draw.per_view_cb = per_view_cb.get();

      auto& device_data = *device->get_private_data<DeviceData>();
      if (device_data.game)
         GetGameDeviceData(device_data).rtao_capture.CaptureDraw(context, draw);
      return false;
   }

   // GPU culled geometry (arkcullgeometry) is drawn indirectly: counted only for now (see RTAO_NOTES.md)
   static bool OnRTAODrawIndirect(reshade::api::command_list* cmd_list, reshade::api::indirect_command type, reshade::api::resource buffer, uint64_t offset, uint32_t draw_count, uint32_t stride)
   {
      if (!g_rtao_capture_active.load(std::memory_order_relaxed) || type != reshade::api::indirect_command::draw_indexed)
         return false;
      if (cmd_list->get_device()->get_api() != reshade::api::device_api::d3d11)
         return false;
      if (IsMainViewDepthWrite(reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native()), false))
         g_rtao_indirect_draws += draw_count;
      return false;
   }

   static void OnRTAODestroyResource(reshade::api::device* device, reshade::api::resource resource)
   {
      if (device->get_api() != reshade::api::device_api::d3d11 || resource.handle == 0)
         return;
      auto* native = reinterpret_cast<ID3D11Resource*>(resource.handle);
      D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      native->GetType(&dimension);
      if (dimension != D3D11_RESOURCE_DIMENSION_BUFFER)
         return;
      auto* device_data = device->get_private_data<DeviceData>();
      if (device_data && device_data->game)
      {
         auto& game_device_data = GetGameDeviceData(*device_data);
         if (game_device_data.rtao_capture.IsInitialized())
            game_device_data.rtao_capture.OnDestroyBuffer(static_cast<ID3D11Buffer*>(native));
      }
   }

   static void OnMapBufferRegion(reshade::api::device* device, reshade::api::resource resource, uint64_t offset, uint64_t size, reshade::api::map_access access, void** data)
   {
      if (device->get_api() != reshade::api::device_api::d3d11) // e.g. RTAO's D3D12 device
         return;
      ID3D11Buffer* buffer = reinterpret_cast<ID3D11Buffer*>(resource.handle);
      // No need to convert to native DX11 flags
      if (access == reshade::api::map_access::write_only || access == reshade::api::map_access::write_discard || access == reshade::api::map_access::read_write)
      {
         auto& device_data = *device->get_private_data<DeviceData>();
         // There seems to only ever be one buffer type of this size, but it's not guaranteed (we might have found more, but it doesn't matter, they are discarded later)...
         // They seemingly all happen on the same thread.
         // Some how these are not marked as "D3D11_BIND_CONSTANT_BUFFER", probably because it copies them over to some other buffer later?
         if (device_data.cb_per_view_global_buffer == buffer)
         {
            D3D11_BUFFER_DESC buffer_desc;
            buffer->GetDesc(&buffer_desc);
#if DEVELOPMENT
            // These are the classic "features" of cbuffer 13 (the one we are looking for), in case any of these were different, it could possibly mean we are looking at the wrong buffer here.
            ASSERT_ONCE(buffer_desc.Usage == D3D11_USAGE_DYNAMIC && buffer_desc.BindFlags == D3D11_BIND_CONSTANT_BUFFER && buffer_desc.CPUAccessFlags == D3D11_CPU_ACCESS_WRITE && buffer_desc.MiscFlags == 0 && buffer_desc.StructureByteStride == 0);
#endif // DEVELOPMENT
            ASSERT_ONCE(!device_data.cb_per_view_global_buffer_map_data);
            device_data.cb_per_view_global_buffer_map_data = *data;
         }
      }
   }

   static void OnUnmapBufferRegion(reshade::api::device* device, reshade::api::resource resource)
   {
      if (device->get_api() != reshade::api::device_api::d3d11) // e.g. RTAO's D3D12 device
         return;
      ID3D11Buffer* buffer = reinterpret_cast<ID3D11Buffer*>(resource.handle);
      DeviceData& device_data = *device->get_private_data<DeviceData>();
      // We assume this buffer is always unmapped before destruction.
      bool is_global_cbuffer = device_data.cb_per_view_global_buffer != nullptr && device_data.cb_per_view_global_buffer == buffer;
      ASSERT_ONCE(!device_data.cb_per_view_global_buffer_map_data || is_global_cbuffer);
      if (is_global_cbuffer && device_data.cb_per_view_global_buffer_map_data != nullptr)
      {
         // The whole buffer size is theoretically "CBPerViewGlobal_buffer_size" but we actually don't have the data for the excessive (padding) bytes,
         // they are never read by shaders on the GPU anyway.
         char global_buffer_data[sizeof(CBPerViewGlobals)];
         std::memcpy(&global_buffer_data[0], device_data.cb_per_view_global_buffer_map_data, sizeof(global_buffer_data));
         if (game->UpdateGlobalCB(&global_buffer_data[0], device))
         {
            // Write back the cbuffer data after we have fixed it up (we always do!)
            std::memcpy(device_data.cb_per_view_global_buffer_map_data, &cb_per_view_global, sizeof(CBPerViewGlobals));
#if DEVELOPMENT
            device_data.cb_per_view_global_buffers.emplace(buffer);
#endif // DEVELOPMENT
         }
         device_data.cb_per_view_global_buffer_map_data = nullptr;
      }
   }

   static void OnExecuteSecondaryCommandList(reshade::api::command_list* cmd_list, reshade::api::command_list* secondary_cmd_list)
   {
      if (cmd_list->get_device()->get_api() != reshade::api::device_api::d3d11) // e.g. RTAO's D3D12 device (bundles)
         return;
      // First, so a deferred RTAO prefix executes before the SR prefix that was split later from the same context
      HandleRTAOCommandList(cmd_list, secondary_cmd_list);

#if ENABLE_SR
      com_ptr<ID3D11DeviceContext> native_device_context;
      com_ptr<ID3D11CommandList> native_command_list;
      ID3D11DeviceChild* device_child = (ID3D11DeviceChild*)(cmd_list->get_native());
      HRESULT hr = device_child->QueryInterface(&native_device_context);
      // ReShade calls this for both DX11's "ExecuteCommandList()" and "FinishCommandList()",
      // we are only interested in the first case, as the second one is irrelevant.
      // "reshade::api::command_list" is a proxy of both "ID3D11DeviceContext" and "ID3D11CommandList",
      // though only for the "ExecuteCommandList()" call the first one passes here will be a "ID3D11DeviceContext" and the second a "ID3D11CommandList",
      // so we branch to make sure we are in that case.
      if (SUCCEEDED(hr))
      {
         device_child = (ID3D11DeviceChild*)(secondary_cmd_list->get_native());
         hr = device_child->QueryInterface(&native_command_list);
         if (SUCCEEDED(hr))
         {
            auto& device_data = *cmd_list->get_device()->get_private_data<DeviceData>();
            auto& game_device_data = GetGameDeviceData(device_data);
            if (game_device_data.final_post_process_command_list == native_command_list.get())
            {
               game_device_data.final_post_process_command_list = nullptr;
               device_data.has_drawn_main_post_processing = true;
            }

            if (game_device_data.sr_remainder_command_list != native_command_list.get())
            {
               return;
            }
            game_device_data.sr_remainder_command_list = nullptr;

            // This command list only contains commands after TAA. Execute the preserved prefix
            // first, so the current frame's color, motion vectors, and depth are available to SR.
            if (game_device_data.sr_partial_command_list)
            {
               native_device_context->ExecuteCommandList(game_device_data.sr_partial_command_list.get(), FALSE);
               game_device_data.sr_partial_command_list.reset();
            }

            ASSERT_ONCE(game_device_data.depth_buffer.get());

            if (!device_data.sr_output_color.get())
            {
               ASSERT_ONCE(false); // This shouldn't happen so we don't really handle it properly
               game_device_data.sr_source_color = nullptr;
               game_device_data.sr_motion_vectors = nullptr;
               device_data.sr_output_color = nullptr;
               return;
            }

            // Do "delayed" DLSS:

            DrawStateStack<DrawStateStackType::FullGraphics> draw_state_stack;
            DrawStateStack<DrawStateStackType::Compute> compute_state_stack;
            draw_state_stack.Cache(native_device_context.get(), device_data.uav_max_count);
            compute_state_stack.Cache(native_device_context.get(), device_data.uav_max_count);

            auto* sr_instance_data = device_data.GetSRInstanceData();
            ASSERT_ONCE(sr_instance_data);

            std::array<uint32_t, 2> dlss_render_resolution = FindClosestIntegerResolutionForAspectRatio((double)device_data.output_resolution.x * (double)device_data.sr_render_resolution_scale, (double)device_data.output_resolution.y * (double)device_data.sr_render_resolution_scale, (double)device_data.output_resolution.x / (double)device_data.output_resolution.y);
            uint32_t render_width_dlss = std::lrintf(device_data.render_resolution.x);
            uint32_t render_height_dlss = std::lrintf(device_data.render_resolution.y);

            // PreDLSS pass
		    //
            // The game does some HDR compression/decompression before and after the TAA,
            // so we replicate that here in PreDLSS pass and PostDLSS pass (immediately after DLSS draw).
            //

		    // Create UAV.
            ComPtr<ID3D11Device> native_device;
            native_device_context->GetDevice(native_device.put());
            ComPtr<ID3D11UnorderedAccessView> uav;
		    ensure(native_device->CreateUnorderedAccessView(game_device_data.sr_source_color.get(), nullptr, uav.put()), >= 0);

		    // Bindings.
		    native_device_context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
		    native_device_context->CSSetShader(device_data.native_compute_shaders.at(CompileTimeStringHash("DS2 Pre DLSS CS")).get(), nullptr, 0);
		    native_device_context->CSSetConstantBuffers(1, 1, &game_device_data.cb_taa_b1);
		    native_device_context->CSSetConstantBuffers(2, 1, &game_device_data.cb_taa_b2);
		    native_device_context->CSSetShaderResources(0, 1, &game_device_data.srv_ro_postfx_luminance_buffautoexposure);

		    native_device_context->Dispatch((render_width_dlss + 8 - 1) / 8, (render_height_dlss + 8 - 1) / 8, 1);

            // SR reads this texture while the native TAA output remains bound as a UAV from the
            // replayed command list. Clear the conflicting compute views before the SR backend.
            std::array<ID3D11UnorderedAccessView*, 2> null_uavs = {};
            native_device_context->CSSetUnorderedAccessViews(0, null_uavs.size(), null_uavs.data(), nullptr);
		    
		    //

            SR::SettingsData settings_data;
            settings_data.output_width = device_data.output_resolution.x;
            settings_data.output_height = device_data.output_resolution.y;
            settings_data.render_width = dlss_render_resolution[0];
            settings_data.render_height = dlss_render_resolution[1];
            settings_data.dynamic_resolution = game_device_data.prey_drs_detected;
            settings_data.hdr = true; // The "HDR" flag in DLSS SR actually means whether the color is in linear space or "sRGB gamma" (apparently not 2.2) (SDR) space, colors beyond 0-1 don't seem to be clipped either way
            settings_data.inverted_depth = true;
            settings_data.mvs_jittered = false;
            settings_data.render_preset = dlss_render_preset;
            settings_data.auto_exposure = true;

            // MVs are in UV space so we need to scale them to screen space for DLSS,
            // aslo we need to flip sign on both for DLSS.
            settings_data.mvs_x_scale = -(float)render_width_dlss;
            settings_data.mvs_y_scale = -(float)render_height_dlss;

            sr_implementations[device_data.sr_type]->UpdateSettings(sr_instance_data, native_device_context.get(), settings_data);

            bool reset_dlss = device_data.force_reset_sr;
            device_data.force_reset_sr = false;

            float dlss_pre_exposure = 1.0;
            SR::SuperResolutionImpl::DrawData draw_data;
            draw_data.source_color = game_device_data.sr_source_color.get();
            draw_data.output_color = device_data.sr_output_color.get();
            draw_data.motion_vectors = game_device_data.sr_motion_vectors.get();
            draw_data.depth_buffer = game_device_data.depth_buffer.get();
            draw_data.exposure = nullptr;
            draw_data.pre_exposure = dlss_pre_exposure;

            // Jitters are in UV offsets so we need to scale them to pixel offsets for DLSS.
            draw_data.jitter_x = cb_per_view_global.cb_projectionmatrix.m20 * (float)render_width_dlss * -0.5;
            draw_data.jitter_y = cb_per_view_global.cb_projectionmatrix.m21 * (float)render_height_dlss * 0.5;

            draw_data.reset = reset_dlss;
            draw_data.render_width = render_width_dlss;
            draw_data.render_height = render_height_dlss;
            draw_data.vert_fov = std::atan(1.0f / projection_matrix.m11) * 2.0;
            draw_data.near_plane = cb_per_view_global.cb_globalviewinfos.z;
            draw_data.far_plane = cb_per_view_global.cb_globalviewinfos.w;
            draw_data.frame_index = cb_luma_global_settings.FrameIndex;

            bool dlss_succeeded = sr_implementations[device_data.sr_type]->Draw(sr_instance_data, native_device_context.get(), draw_data);
            ASSERT_ONCE(dlss_succeeded); // We can't restore the original TAA pass at this point (well, we could, but it's pointless, we'll just skip one frame) // TODO: copy the resource instead?

            // PostDLSS pass
		    //

		    // Create UAV.
		    ensure(native_device->CreateUnorderedAccessView(device_data.sr_output_color.get(), nullptr, uav.put()), >= 0);

		    // Bindings.
		    native_device_context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
		    native_device_context->CSSetShader(device_data.native_compute_shaders.at(CompileTimeStringHash("DS2 Post DLSS CS")).get(), nullptr, 0);
		    native_device_context->CSSetConstantBuffers(1, 1, &game_device_data.cb_taa_b1);
		    native_device_context->CSSetConstantBuffers(2, 1, &game_device_data.cb_taa_b2);
		    native_device_context->CSSetShaderResources(0, 1, &game_device_data.srv_ro_postfx_luminance_buffautoexposure);

		    native_device_context->Dispatch((device_data.output_resolution.x + 8 - 1) / 8, (device_data.output_resolution.y + 8 - 1) / 8, 1);
		    
		    //

            game_device_data.sr_source_color = nullptr;
            game_device_data.sr_motion_vectors = nullptr;
            device_data.sr_output_color = nullptr;

            draw_state_stack.Restore(native_device_context.get());
            compute_state_stack.Restore(native_device_context.get());

            if (dlss_succeeded)
            {
               device_data.has_drawn_sr = true;
            }
            // DLSS Failed, suppress it for this frame and fall back on SMAA/TAA, hoping that anything before would have been rendered correctly for it already (otherwise it will start being correct in the next frame, given we suppress it (until manually toggled again, given that it'd likely keep failing))
            else
            {
               ASSERT_ONCE(false);
               cb_luma_global_settings.SRType = 0;
               device_data.cb_luma_global_settings_dirty = true;
               device_data.sr_suppressed = true;
               device_data.force_reset_sr = true; // We missed frames so it's good to do this, it might also help prevent further errors
            }
         }
         return;
      }
      hr = device_child->QueryInterface(&native_command_list);
      {
         device_child = (ID3D11DeviceChild*)(secondary_cmd_list->get_native());
         hr = device_child->QueryInterface(&native_device_context);
         if (SUCCEEDED(hr))
         {
            auto& device_data = *cmd_list->get_device()->get_private_data<DeviceData>();
            auto& game_device_data = GetGameDeviceData(device_data);
            if (game_device_data.final_post_process_command_list == native_device_context.get())
            {
               game_device_data.final_post_process_command_list = native_command_list.get();
            }
            if (game_device_data.sr_deferred_command_list == native_device_context.get())
            {
               game_device_data.sr_deferred_command_list = nullptr;
               game_device_data.sr_remainder_command_list = native_command_list.get();
            }
            return;
         }
      }
      ASSERT_ONCE(false); // Invalid case?
#endif // ENABLE_SR
   }

   void LoadConfigs() override
   {
      reshade::api::effect_runtime* runtime = nullptr;

      int ao_mode = AO_MODE_XEGTAO;
      if (!reshade::get_config_value(runtime, NAME, "AOMode", ao_mode))
      {
         // Migrate the old boolean setting
         bool xegtao_enable = true;
         if (reshade::get_config_value(runtime, NAME, "XeGTAOEnable", xegtao_enable))
            ao_mode = xegtao_enable ? AO_MODE_XEGTAO : AO_MODE_OFF;
      }
      g_ao_mode = (ao_mode >= AO_MODE_OFF && ao_mode < AO_MODE_COUNT) ? ao_mode : AO_MODE_XEGTAO;
      g_ao_mode_apply_pending = true;
      reshade::get_config_value(runtime, NAME, "RTAOTestOutput", g_rtao_debug_mode);
      g_rtao_debug_mode = std::clamp(g_rtao_debug_mode, 0, int(RTAO_OUTPUT_COUNT) - 1);
      reshade::get_config_value(runtime, NAME, "RTAORadius", g_rtao_radius);
      reshade::get_config_value(runtime, NAME, "RTAORays", g_rtao_rays);
      reshade::get_config_value(runtime, NAME, "RTAOCapturePass", g_rtao_capture_pass);
      reshade::get_config_value(runtime, NAME, "RTAOStrength", g_rtao_strength);
      g_rtao_strength = std::clamp(g_rtao_strength, 0.25f, 4.f);
      g_rtao_radius = std::clamp(g_rtao_radius, 0.05f, 50.f);
      g_rtao_rays = std::clamp(g_rtao_rays, 1, 64);
      g_rtao_capture_pass = std::clamp(g_rtao_capture_pass, 0, 1);

      auto& game_settings = cb_luma_global_settings.GameSettings;
      reshade::get_config_value(runtime, NAME, "BloomStrength", game_settings.BloomStrength);
      reshade::get_config_value(runtime, NAME, "LensDirtStrength", game_settings.LensDirtStrength);
      reshade::get_config_value(runtime, NAME, "LensFlareStrength", game_settings.LensFlareStrength);
      reshade::get_config_value(runtime, NAME, "VignetteStrength", game_settings.VignetteStrength);
   }

   void DrawImGuiSettings(DeviceData& device_data) override
   {
      reshade::api::effect_runtime* runtime = nullptr;

      ImGui::SeparatorText("Effects");
      auto& game_settings = cb_luma_global_settings.GameSettings;

      if (ImGui::SliderFloat("Bloom Strength", &game_settings.BloomStrength, 0.f, 2.f))
         device_data.cb_luma_global_settings_dirty = true;
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "BloomStrength", game_settings.BloomStrength);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Bloom strength (1 = vanilla, 0 = disabled).");
      if (DrawResetButton(game_settings.BloomStrength, default_luma_global_game_settings.BloomStrength, "BloomStrength"))
         device_data.cb_luma_global_settings_dirty = true;

      if (ImGui::SliderFloat("Lens Dirt Strength", &game_settings.LensDirtStrength, 0.f, 2.f))
         device_data.cb_luma_global_settings_dirty = true;
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "LensDirtStrength", game_settings.LensDirtStrength);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Lens dirt strength (1 = vanilla, 0 = disabled).");
      if (DrawResetButton(game_settings.LensDirtStrength, default_luma_global_game_settings.LensDirtStrength, "LensDirtStrength"))
         device_data.cb_luma_global_settings_dirty = true;

      if (ImGui::SliderFloat("Lens Flare Strength", &game_settings.LensFlareStrength, 0.f, 2.f))
         device_data.cb_luma_global_settings_dirty = true;
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "LensFlareStrength", game_settings.LensFlareStrength);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Lens flare and streak strength (1 = vanilla, 0 = disabled).");
      if (DrawResetButton(game_settings.LensFlareStrength, default_luma_global_game_settings.LensFlareStrength, "LensFlareStrength"))
         device_data.cb_luma_global_settings_dirty = true;

      if (ImGui::SliderFloat("Vignette Strength", &game_settings.VignetteStrength, 0.f, 2.f))
         device_data.cb_luma_global_settings_dirty = true;
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "VignetteStrength", game_settings.VignetteStrength);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Screen-overlay vignette strength (1 = vanilla, 0 = disabled).");
      if (DrawResetButton(game_settings.VignetteStrength, default_luma_global_game_settings.VignetteStrength, "VignetteStrength"))
         device_data.cb_luma_global_settings_dirty = true;

      ImGui::SeparatorText("Ambient Occlusion");

      // Listed in this order; the stored mode values stay 0 off, 1 XeGTAO, 2 RTAO, 3 HBAO+
      static const int ao_mode_order[AO_MODE_COUNT] = { AO_MODE_OFF, AO_MODE_HBAO, AO_MODE_XEGTAO, AO_MODE_RTAO };
      const char* ao_mode_names[AO_MODE_COUNT] = { "Off", g_game_ao.valid ? "HBAO+ (game)" : "HBAO+ (unavailable: unknown game version)", "XeGTAO", "RTAO (ray traced, experimental)" };
      int ao_mode = g_ao_mode.load();
      if (ao_mode < AO_MODE_OFF || ao_mode >= AO_MODE_COUNT)
         ao_mode = AO_MODE_XEGTAO;
      int ao_index = 0;
      for (int i = 0; i < AO_MODE_COUNT; i++)
         if (ao_mode_order[i] == ao_mode)
            ao_index = i;
      if (ImGui::Combo("Ambient Occlusion", &ao_index, ao_mode_names, AO_MODE_COUNT))
      {
         ao_mode = ao_mode_order[ao_index];
         if (ao_mode == AO_MODE_HBAO && !g_game_ao.valid)
            ao_mode = AO_MODE_OFF;
         g_ao_mode = ao_mode;
         g_ao_mode_apply_pending = true;
         reshade::set_config_value(runtime, NAME, "AOMode", ao_mode);
      }
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      {
         ImGui::SetTooltip("Off: no AO (the game's own SSAO/DWAO draws nothing on PC).\nHBAO+: the game's NVIDIA HBAO+ (sets the game's r_hbao; the in game AO menu stays in sync).\nXeGTAO: Intel's ground truth AO, replacing the game's SSAO pass.\nRTAO: ray traced AO on a side D3D12 device (needs DXR 1.1, falls back to XeGTAO if unavailable).");
      }
      static const char* ao_view_names[] = { "Off", "Final AO (after the game's blur)", "RTAO output (before the game's blur)" };
      ImGui::Combo("View AO only (debug)", &g_ao_view_mode, ao_view_names, IM_ARRAYSIZE(ao_view_names));
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Shows the AO buffer fullscreen in grey (white = no occlusion) instead of the game.\n\"Final AO\" works with every AO mode (the game's SSAO, XeGTAO, RTAO), so you can compare them directly.\nWith RTAO, the RTAO Output choice (e.g. BLAS coverage, no denoiser) is what shows here.");
      if (ao_mode == AO_MODE_RTAO)
      {
         std::string status;
         {
            const std::lock_guard lock(g_rtao_status_mutex);
            status = g_rtao_status;
         }
         ImGui::TextWrapped("RTAO status: %s", status.empty() ? "waiting for the SSAO pass" : status.c_str());
         static const char* rtao_output_names[RTAO_OUTPUT_COUNT] = { "Ray traced AO", "Test pattern (interop check)", "Depth test AO (no ray tracing)", "BLAS coverage (debug)", "Ray traced AO, no denoiser" };
         if (ImGui::Combo("RTAO Output", &g_rtao_debug_mode, rtao_output_names, RTAO_OUTPUT_COUNT))
            reshade::set_config_value(runtime, NAME, "RTAOTestOutput", g_rtao_debug_mode);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("BLAS coverage: white = captured geometry matches the depth buffer, grey = geometry missing from the ray tracing scene, black = geometry in front of what the depth shows (misplaced or extra).");
         if (ImGui::SliderFloat("RTAO Radius", &g_rtao_radius, 0.05f, 10.f, "%.2f", ImGuiSliderFlags_Logarithmic))
            g_rtao_radius = std::clamp(g_rtao_radius, 0.05f, 50.f);
         if (ImGui::IsItemDeactivatedAfterEdit())
            reshade::set_config_value(runtime, NAME, "RTAORadius", g_rtao_radius);
         if (ImGui::SliderInt("RTAO Rays", &g_rtao_rays, 1, 16))
            g_rtao_rays = std::clamp(g_rtao_rays, 1, 64);
         if (ImGui::IsItemDeactivatedAfterEdit())
            reshade::set_config_value(runtime, NAME, "RTAORays", g_rtao_rays);
         static const char* capture_pass_names[] = { "Depth only draws", "All depth writing draws" };
         if (ImGui::Combo("RTAO Capture", &g_rtao_capture_pass, capture_pass_names, IM_ARRAYSIZE(capture_pass_names)))
            reshade::set_config_value(runtime, NAME, "RTAOCapturePass", g_rtao_capture_pass);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Which main view draws feed the ray tracing scene. Use \"All\" if the coverage view shows missing (grey) geometry.");
         if (ImGui::SliderFloat("RTAO Strength", &g_rtao_strength, 0.25f, 4.f, "%.2f"))
            g_rtao_strength = std::clamp(g_rtao_strength, 0.25f, 4.f);
         if (ImGui::IsItemDeactivatedAfterEdit())
            reshade::set_config_value(runtime, NAME, "RTAOStrength", g_rtao_strength);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Deepens (> 1) or lightens (< 1) the ray traced AO: a power curve on the result. 1 = as traced.");
         DrawResetButton(g_rtao_strength, 1.f, "RTAOStrength");

         auto& game_device_data = GetGameDeviceData(device_data);
         if (game_device_data.rtao_capture.IsInitialized() && ImGui::TreeNode("RTAO stats"))
         {
            const RTAO::CaptureStats stats = game_device_data.rtao_capture.GetStats();
            ImGui::Text("GPU time (capture readback, BLAS/TLAS builds, trace, denoise): %.2f ms", game_device_data.rtao_gpu_us.load() / 1000.0);
            ImGui::Text("Draws seen %llu, captured %llu (%llu instances)", (unsigned long long)stats.draws_seen, (unsigned long long)stats.draws_captured, (unsigned long long)stats.instances_captured);
            ImGui::Text("Main view draws by matrix source: none %llu, direct %llu, indexed %llu, cbuffer %llu, MVP %llu",
               (unsigned long long)g_rtao_draws_by_source[0].load(), (unsigned long long)g_rtao_draws_by_source[1].load(), (unsigned long long)g_rtao_draws_by_source[2].load(), (unsigned long long)g_rtao_draws_by_source[3].load(), (unsigned long long)g_rtao_draws_by_source[4].load());
            ImGui::Text("Unknown vertex shader %llu, unusable input layout %llu, GPU culled (indirect, not captured) %llu",
               (unsigned long long)g_rtao_draws_unknown_shader.load(), (unsigned long long)g_rtao_draws_unusable_layout.load(), (unsigned long long)g_rtao_indirect_draws.load());
            ImGui::Text("Skipped: shader %llu, format %llu, buffer %llu, arena full %llu, ring %llu, binding %llu, other %llu",
               (unsigned long long)stats.skipped[RTAO::kSkipShader], (unsigned long long)stats.skipped[RTAO::kSkipFormat], (unsigned long long)stats.skipped[RTAO::kSkipBuffer], (unsigned long long)stats.skipped[RTAO::kSkipArenaFull],
               (unsigned long long)stats.skipped[RTAO::kSkipRingFull], (unsigned long long)stats.skipped[RTAO::kSkipBinding], (unsigned long long)stats.skipped[RTAO::kSkipOther]);
            ImGui::Text("Meshes %llu (%llu with a BLAS, %llu failed), BLAS %llu (%.1f MB), buffers copied %llu (arena %.1f MB)", (unsigned long long)stats.meshes, (unsigned long long)stats.meshes_ready,
               (unsigned long long)stats.blas_failed, (unsigned long long)game_device_data.rtao_blas_count.load(), game_device_data.rtao_blas_bytes.load() / 1048576.0, (unsigned long long)stats.buffers_copied, stats.arena_vertex_bytes / 1048576.0);
            ImGui::Text("TLAS instances %u, instances read back last frame %llu, view model instances %llu, draws after the AO pass (left out) %llu", game_device_data.rtao_tlas_instances.load(), (unsigned long long)stats.frame_instances, (unsigned long long)stats.view_model_instances, (unsigned long long)stats.after_ao_records);
            ImGui::Text("Vertex shaders reflected: none %llu, direct %llu, indexed %llu, cbuffer %llu, MVP %llu", (unsigned long long)g_rtao_vs_by_source[0].load(), (unsigned long long)g_rtao_vs_by_source[1].load(),
               (unsigned long long)g_rtao_vs_by_source[2].load(), (unsigned long long)g_rtao_vs_by_source[3].load(), (unsigned long long)g_rtao_vs_by_source[4].load());
            ImGui::TreePop();
         }
      }
   }

   void PrintImGuiAbout() override
   {
      ImGui::Text("Luma for \"Dishonored 2\" is developed by Pumbo and Musa and is open source and free.\nIf you enjoy it, consider donating.", "");

      const auto button_color = ImGui::GetStyleColorVec4(ImGuiCol_Button);
      const auto button_hovered_color = ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered);
      const auto button_active_color = ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive);
      ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(70, 134, 0, 255));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(70 + 9, 134 + 9, 0, 255));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(70 + 18, 134 + 18, 0, 255));
      static const std::string donation_link_pumbo = std::string("Buy Pumbo a Coffee on buymeacoffee ") + std::string(ICON_FK_OK);
      if (ImGui::Button(donation_link_pumbo.c_str()))
      {
         system("start https://buymeacoffee.com/realfiloppi");
      }
      static const std::string donation_link_pumbo_2 = std::string("Buy Pumbo a Coffee on ko-fi ") + std::string(ICON_FK_OK);
      if (ImGui::Button(donation_link_pumbo_2.c_str()))
      {
         system("start https://ko-fi.com/realpumbo");
      }
      ImGui::PopStyleColor(3);

      ImGui::NewLine();
      // Restore the previous color, otherwise the state we set would persist even if we popped it
      ImGui::PushStyleColor(ImGuiCol_Button, button_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, button_hovered_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, button_active_color);
#if 0
      static const std::string mod_link = std::string("Nexus Mods Page ") + std::string(ICON_FK_SEARCH);
      if (ImGui::Button(mod_link.c_str()))
      {
         system("start https://www.nexusmods.com/prey2017/mods/149");
      }
#endif
      static const std::string social_link = std::string("Join our \"HDR Den\" Discord ") + std::string(ICON_FK_SEARCH);
      if (ImGui::Button(social_link.c_str()))
      {
         // Unique link for Luma by Pumbo (to track the origin of people joining), do not share for other purposes
         static const std::string obfuscated_link = std::string("start https://discord.gg/J9fM") + std::string("3EVuEZ");
         system(obfuscated_link.c_str());
      }
      static const std::string contributing_link = std::string("Contribute on Github ") + std::string(ICON_FK_FILE_CODE);
      if (ImGui::Button(contributing_link.c_str()))
      {
         system("start https://github.com/Filoppi/Luma-Framework");
      }
      ImGui::PopStyleColor(3);

      ImGui::NewLine();
      ImGui::Text("Credits:"
         "\n\nMain:"
         "\nPumbo"
         "\nMusa"

         "\n\nThird Party:"
         "\nReShade"
         "\nImGui"
         "\nRenoDX"
         "\n3Dmigoto"
         "\nOklab"
         "\nDICE (HDR tonemapper)"
         , "");
   }

#if DEVELOPMENT || TEST
   void PrintImGuiInfo(const DeviceData& device_data) override
   {
      std::string text;

      ImGui::NewLine();
      ImGui::Text("Camera Jitters: ", "");
      // In NCD space
      // Add padding to make it draw consistently even with a "-" in front of the numbers.
      text = (projection_jitters.x >= 0 ? " " : "") + std::to_string(projection_jitters.x) + " " + (projection_jitters.y >= 0 ? " " : "") + std::to_string(projection_jitters.y);
      ImGui::Text(text.c_str(), "");
      // In absolute space
      // These values should be between -1 and 1 (note that X might be flipped)
      text = (projection_jitters.x >= 0 ? " " : "") + std::to_string(projection_jitters.x * device_data.render_resolution.x) + " " + (projection_jitters.y >= 0 ? " " : "") + std::to_string(projection_jitters.y * device_data.render_resolution.y);
      ImGui::Text(text.c_str(), "");

      ImGui::NewLine();
      ImGui::Text("Texture Mip LOD Bias: ", "");
      text = std::to_string(device_data.texture_mip_lod_bias_offset);
      ImGui::Text(text.c_str(), "");

      ImGui::NewLine();
      ImGui::Text("Camera: ", "");
      //TODOFT3: figure out if this is meters or what
      text = "Scene Near: " + std::to_string(cb_per_view_global.cb_globalviewinfos.z) + " Scene Far: " + std::to_string(cb_per_view_global.cb_globalviewinfos.w);
      ImGui::Text(text.c_str(), "");
      float tanHalfFOVX = 1.f / projection_matrix.m00;
      float tanHalfFOVY = 1.f / projection_matrix.m11;
      float FOVX = atan(tanHalfFOVX) * 2.0 * 180 / M_PI;
      float FOVY = atan(tanHalfFOVY) * 2.0 * 180 / M_PI;
      text = "Scene: Hor FOV: " + std::to_string(FOVX) + " Vert FOV: " + std::to_string(FOVY);
      ImGui::Text(text.c_str(), "");
      tanHalfFOVX = 1.f / nearest_projection_matrix.m00;
      tanHalfFOVY = 1.f / nearest_projection_matrix.m11;
      FOVX = atan(tanHalfFOVX) * 2.0 * 180 / M_PI;
      FOVY = atan(tanHalfFOVY) * 2.0 * 180 / M_PI;
   }
#endif
};

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
   if (ul_reason_for_call == DLL_PROCESS_ATTACH)
   {
      Globals::SetGlobals(PROJECT_NAME, "Dishonored 2 + Death of the Outsider Luma mod");
      Globals::VERSION = 1;
      Globals::DEVELOPMENT_STATE = Globals::ModDevelopmentState::WorkInProgress;

      default_luma_global_game_settings.BloomStrength = 1.f;
      default_luma_global_game_settings.LensDirtStrength = 1.f;
      default_luma_global_game_settings.LensFlareStrength = 1.f;
      default_luma_global_game_settings.VignetteStrength = 1.f;
      cb_luma_global_settings.GameSettings = default_luma_global_game_settings;

      shader_hashes_TAA.compute_shaders.emplace(std::stoul("06BBC941", nullptr, 16)); // DH2
      shader_hashes_TAA.compute_shaders.emplace(std::stoul("8EDF67D9", nullptr, 16)); // DH2 Low quality TAA? // TODO: add an assert on this!
      shader_hashes_TAA.compute_shaders.emplace(std::stoul("9F77B624", nullptr, 16)); // DH DOTO
      shader_hashes_Tonemap.pixel_shaders.emplace(0xA6F33860); // DH2
      shader_hashes_Tonemap.pixel_shaders.emplace(0xD2F50617); // DH DOTO
      shader_hashes_UpscaleSharpen.pixel_shaders.emplace(std::stoul("1A0CD2AE", nullptr, 16)); // DH2 + DH DOTO
      shader_hashes_DownsampleDepth.compute_shaders.emplace(std::stoul("27BD5265", nullptr, 16)); // DH2 + DH DOTO
      shader_hashes_UnprojectDepth.compute_shaders.emplace(std::stoul("223FB9DA", nullptr, 16)); // DH2
      shader_hashes_UnprojectDepth.compute_shaders.emplace(std::stoul("74E15FB8", nullptr, 16)); // DH DOTO
      shader_hashes_SSAO.pixel_shaders.emplace(0x94445D2D); // DH2 + DH DOTO
      shader_hashes_AODenoise.pixel_shaders.emplace(0x4CD2BE39); // DH2
      // All UI pixel shaders (these are all Shader Model 4.0, as opposed to the rest of the rendering using SM5.0)
      shader_hashes_UI.pixel_shaders = {
         std::stoul("6FE8114D", nullptr, 16),
         std::stoul("08F8ECFE", nullptr, 16),
         std::stoul("28E5E21A", nullptr, 16),
         std::stoul("38E853C8", nullptr, 16),
         std::stoul("B9E43380", nullptr, 16),
         std::stoul("BC1D41CE", nullptr, 16),
         std::stoul("CC4FB5BF", nullptr, 16),
         std::stoul("D34DA30E", nullptr, 16),
         std::stoul("E3BB1976", nullptr, 16),
         std::stoul("EE4A38D2", nullptr, 16),
      };

      // 6-13 are seemingly totally unused by Dishonored 2
      luma_settings_cbuffer_index = 13;
      luma_data_cbuffer_index = 12;

      swapchain_format_upgrade_type = TextureFormatUpgradesType::AllowedEnabled;
      swapchain_upgrade_type = SwapchainUpgradeType::scRGB;
      texture_format_upgrades_type = TextureFormatUpgradesType::AllowedEnabled;
      texture_upgrade_formats = {
            reshade::api::format::r8g8b8a8_unorm,
            reshade::api::format::r8g8b8a8_unorm_srgb,
            reshade::api::format::r8g8b8a8_typeless,
            reshade::api::format::r8g8b8x8_unorm,
            reshade::api::format::r8g8b8x8_unorm_srgb,
            reshade::api::format::b8g8r8a8_unorm,
            reshade::api::format::b8g8r8a8_unorm_srgb,
            reshade::api::format::b8g8r8a8_typeless,
            reshade::api::format::b8g8r8x8_unorm,
            reshade::api::format::b8g8r8x8_unorm_srgb,
            reshade::api::format::b8g8r8x8_typeless,

            reshade::api::format::r11g11b10_float,
      };

      // FIXME: These are asserting.
      //texture_format_upgrades_lut_size = 32;
      //texture_format_upgrades_lut_dimensions = LUTDimensions::_3D;

      enable_samplers_upgrade = true;

      enable_ui_separation = true;
      ui_separation_use_ui_hashes_only = true;
      ui_separation_format = DXGI_FORMAT_R16G16B16A16_FLOAT; // TODO: pick the best format, it's probably DXGI_FORMAT_R16G16B16A16_UNORM or DXGI_FORMAT_R8G8B8A8_UNORM.

      game = new Dishonored2();
   }
   else if (ul_reason_for_call == DLL_PROCESS_DETACH)
   {
      reshade::unregister_event<reshade::addon_event::map_buffer_region>(Dishonored2::OnMapBufferRegion);
      reshade::unregister_event<reshade::addon_event::unmap_buffer_region>(Dishonored2::OnUnmapBufferRegion);
      reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(Dishonored2::OnExecuteSecondaryCommandList);
      reshade::unregister_event<reshade::addon_event::init_pipeline>(Dishonored2::OnRTAOInitPipeline);
      reshade::unregister_event<reshade::addon_event::destroy_pipeline>(Dishonored2::OnRTAODestroyPipeline);
      reshade::unregister_event<reshade::addon_event::draw_indexed>(Dishonored2::OnRTAODrawIndexed);
      reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(Dishonored2::OnRTAODrawIndirect);
      reshade::unregister_event<reshade::addon_event::destroy_resource>(Dishonored2::OnRTAODestroyResource);
   }

   CoreMain(hModule, ul_reason_for_call, lpReserved);

   return TRUE;
}