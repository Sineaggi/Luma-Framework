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

// RT sun shadows replays the game's screen shadows draw itself (to bind the traced visibility around it), and the
// RTAO composite replays the SSAO draw: both need the original draw call, which release builds only provide with this.
#define ENABLE_POST_DRAW_DISPATCH_CALLBACK 1

#include "../../../Shaders/Dishonored 2/Includes/GameCBuffers.hlsl"
#include "..\..\Core\core.hpp"

#include "rtao_interop.h"
#include "rtao_accel.h"
#include "rtao_capture.h"
#include "rtao_pass.h"
#include "rtsun_pass.h"
#include "rt_debug_capture.h"

#include <chrono>
#include <map>
#include <tuple>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <shared_mutex>
#include <thread>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../External/reshade/deps/fpng/src/stb_image_write.h"

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

   // The game's r_hbao cvar, driven from Luma (Steam 1.77.9.0). The addresses are the ones Sineaggi's AO menu ASI
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
   float g_rtao_strength = 2.f; // RTAO power curve (config "RTAOStrength"); 2 matched XeGTAO's depth in testing
   float g_rtao_radius = 1.5f; // World units (config "RTAORadius")
   int g_rtao_rays = 4; // Per pixel per frame (config "RTAORays")
   int g_rtao_capture_pass = 1; // 0: depth only draws (no render target bound), 1: every depth writing draw (config "RTAOCapturePass")
   bool g_rtao_skinned = true; // Skinned characters in the ray tracing scene (config "RTAOSkinned")
   bool g_rtao_dynamic = true; // Meshes with a DYNAMIC vertex buffer (movable objects) in the ray tracing scene (config "RTAODynamic")
   // Also capture the game's shadow map draws (config "RTAOShadowCasters"): the casters behind and beside the camera,
   // which the main view never draws but the sun needs (RTAO::DrawInfo::shadow_view)
   bool g_rtao_shadow_casters = true;
   std::atomic<uint64_t> g_rtao_shadow_view_draws = 0; // Indexed draws into shadow maps seen (captured or not)
   std::atomic<uint64_t> g_rtao_shadow_indirect_draws = 0; // Indirect (GPU culled) draws into shadow maps: not captured
   // Shadow map draws by array slice (the DSV's first slice: one cascade per slice), and those left out as far slices
   std::atomic<uint64_t> g_rtao_shadow_slice_draws[8] = {}, g_rtao_shadow_far_skipped = 0;
   // Shadow map casters only from the first slices (config "RTAOShadowNearSlices"): the game draws simplified copies of
   // the world (other LODs, no window openings) into its far cascades, which must not enter the ray tracing scene (the
   // near cascade, used for the view, has the detailed geometry). "RTAOShadowNearOnly" off takes every slice (test).
   int g_rtao_shadow_near_slices = 2;
   // Shadow map layout diagnostics: draws per (slice, viewport rectangle), so the periodic diag line shows how the game
   // lays out its cascades (one per array slice, or several viewports in one slice)
   std::mutex g_rtao_shadow_layout_mutex;
   std::map<std::tuple<uint32_t, int, int, int, int>, uint64_t> g_rtao_shadow_layout;
   bool g_rtao_shadow_near_only = true;
   // Shadow map draws per frame (at present): frames with none, the most in one frame, over the current log window
   uint64_t g_rtao_shadow_draws_at_present = 0;
   uint32_t g_rtao_shadow_zero_frames = 0, g_rtao_shadow_frames = 0;
   uint64_t g_rtao_shadow_max_per_frame = 0;
   // Wobble diagnostics (GetRTAOPerView): SSAO draws, ones whose thread had no per view copy, ones where the thread's
   // view projection differed from Luma's shared copy (and the largest element difference, x 1e6)
   std::atomic<uint64_t> g_rtao_view_checks = 0, g_rtao_view_no_thread_copy = 0, g_rtao_view_mismatch = 0, g_rtao_view_other_frame = 0, g_rtao_view_other_camera = 0;
   std::atomic<uint32_t> g_rtao_view_mismatch_max = 0;
   thread_local CBPerViewGlobals t_rtao_per_view;
   thread_local bool t_rtao_per_view_valid = false;
   std::atomic<uint32_t> g_rtao_shadow_view_width = 0, g_rtao_shadow_view_height = 0, g_rtao_shadow_view_slices = 0; // The last shadow map seen

   bool RTAOOutputIsRayTraced(int output)
   {
      return output == RTAO_OUTPUT_AO || output == RTAO_OUTPUT_COVERAGE || output == RTAO_OUTPUT_RAW;
   }

   // Geometry capture: what each game vertex shader / input layout means for capture, by native object pointer
   std::shared_mutex g_rtao_shader_mutex;
   std::unordered_map<uint64_t, RTAO::VertexShaderInfo> g_rtao_vertex_shaders;
   std::unordered_map<uint64_t, uint32_t> g_rtao_vertex_shader_hashes; // Luma's shader hash (as in the Dump folder names), diagnostics
   std::unordered_map<uint64_t, uint32_t> g_rtao_pixel_shader_hashes; // Same, pixel shaders
   // Diagnostics: each uncaptured main view (shader, reason) logged once, up to a limit
   std::mutex g_rtao_diag_mutex;
   std::unordered_set<uint64_t> g_rtao_diag_logged;
   std::unordered_map<uint64_t, RTAO::InputLayoutInfo> g_rtao_input_layouts;
   std::atomic<bool> g_rtao_capture_active = false; // Set while RTAO traces, so the draw hook costs nothing otherwise
   std::atomic<uint32_t> g_rtao_main_depth_width = 0, g_rtao_main_depth_height = 0; // The main view's depth size
   std::atomic<uint64_t> g_rtao_vs_by_source[5] = {}; // Reflected vertex shaders, by RTAO::MatrixSource
   std::atomic<uint64_t> g_rtao_vs_skinned = 0; // Reflected skinned vertex shaders (also counted by source)
   std::atomic<uint64_t> g_rtao_draws_skinned = 0; // Main view skinned draws
   std::atomic<uint64_t> g_rtao_draws_by_source[5] = {}; // Main view draws, by their vertex shader's matrix source
   std::atomic<uint64_t> g_rtao_draws_unknown_shader = 0, g_rtao_draws_unusable_layout = 0, g_rtao_indirect_draws = 0, g_rtao_draws_not_triangle_list = 0;
   // Main view draws whose vertex shader can't be captured, by reason (the reasons are static strings)
   std::mutex g_rtao_skip_reason_mutex;
   std::unordered_map<const char*, uint64_t> g_rtao_skip_reasons;

   // RTAO foliage: wind (vertex animated vegetation, captured in rest pose) and alpha tested (cutout) geometry
   bool g_rtao_wind_meshes = true; // Capture wind meshes (config "RTAOWindMeshes")
   bool g_rtao_cutouts = true; // Capture alpha tested draws, as cutouts (config "RTAOCutouts")
   float g_rtao_cutout_opacity = 0.5f; // Chance a ray crossing a cutout triangle stops there (config "RTAOCutoutOpacity")
   bool g_rtao_cutouts_exact = true; // Texture exact cutouts: the material's alpha mask decides each hit (config "RTAOCutoutsExact")
   std::unordered_map<uint64_t, RTAO::PixelShaderInfo> g_rtao_alpha_tested_ps; // Alpha tested pixel shaders, by native object pointer (under g_rtao_shader_mutex)
   std::atomic<uint64_t> g_rtao_ps_reflected = 0, g_rtao_ps_discard = 0, g_rtao_ps_alpha_tested = 0, g_rtao_ps_texture_exact = 0;
   // Trace cost (config "RTAOQuality"): Low = half resolution trace (a quarter of the rays, upsampled by the denoiser),
   // Medium = full resolution with half of "RTAO Rays", High = full resolution with all of them
   enum RTAOQuality : int
   {
      RTAO_QUALITY_LOW = 0,
      RTAO_QUALITY_MEDIUM = 1,
      RTAO_QUALITY_HIGH = 2,
      RTAO_QUALITY_COUNT
   };
   int g_rtao_quality = RTAO_QUALITY_LOW;
   // Cutouts decided once per ray (solid or absent) instead of per crossed triangle: no candidate loop (config "RTAOFastCutouts")
   bool g_rtao_fast_cutouts = false;
   // Skip draws that repeat one captured earlier in the frame (depth prepass + G-buffer pass) (config "RTAODedup")
   bool g_rtao_dedup = true;
   // Frames a skinned mesh (character, swinging window flap) keeps its last pose when a frame misses its draw (config "RTAOSkinnedHoldFrames")
   int g_rtao_skinned_hold = 30;
   bool g_rtao_recopy_written = false; // Re-copy captured buffers the game rewrites (config "RTAORecopyWrittenBuffers", H3 test)
   bool g_rtao_freeze_scene = false;
   // Popping tests (not saved): openables taken from shadow map draws; shadow map objects not confirmed yet kept 64 frames, not 4
   bool g_rtdbg_shadow_openables = true;
   bool g_rtdbg_shadow_young_long = false; // "old shadow map retention" (TlasBuilder::shadow_mode 1) // Debug, not saved: trace the last TLAS again every frame (no capture, no rebuild)
   int g_rtao_blas_budget = 64; // New static BLASes built per frame (config "RTAOBlasBudget")
   // Map(WRITE_DISCARD) of game buffers, per recording thread: buffer -> unique id of its last discard. Part of a draw's
   // identity for the dedup, so a per instance buffer rewritten between two draws never makes them look the same.
   std::atomic<uint64_t> g_rtao_discard_counter = 0;
   thread_local std::unordered_map<ID3D11Resource*, uint64_t> t_rtao_discards;
   void RTAONoteDiscard(ID3D11Resource* resource)
   {
      if (t_rtao_discards.size() > 8192)
         t_rtao_discards.clear(); // Forgetting only makes later draws look different (no dedup), never the same
      t_rtao_discards[resource] = ++g_rtao_discard_counter;
   }
   uint64_t RTAODiscardVersion(ID3D11Resource* resource)
   {
      if (!resource)
         return 0;
      auto it = t_rtao_discards.find(resource);
      return it != t_rtao_discards.end() ? it->second : 0;
   }
   uint64_t RTAODiscardVersion(ID3D11ShaderResourceView* view)
   {
      if (!view)
         return 0;
      com_ptr<ID3D11Resource> resource;
      view->GetResource(&resource);
      return RTAODiscardVersion(resource.get());
   }
   std::atomic<uint64_t> g_rtao_draws_alpha_tested = 0; // Main view draws with an alpha tested pixel shader

   // RT sun shadows (RTAO_NOTES.md "RT sun shadows"): the directional light's visibility, traced in the RTAO D3D12
   // submission (at the SSAO draw, from the RTAO TLAS), used by the game's screen shadows pass (replaced shaders
   // ScreenShadows_*.hlsl) in place of, or on top of, its cascaded shadow maps. Needs AO mode RTAO.
   ShaderHashesList shader_hashes_ScreenShadows;
   enum RTSunMode : int
   {
      RTSUN_MODE_OFF = 0, // The game's shadow maps
      RTSUN_MODE_RT = 1, // Ray traced (times the cloud shadow)
      RTSUN_MODE_RT_X_GAME = 2, // Ray traced x the game's shadow maps
      RTSUN_MODE_COUNT
   };
   int g_rtsun_mode = RTSUN_MODE_OFF; // Config "RTSunMode" (mirrored in LumaGameSettings.RTSunMode for the shaders)
   float g_rtsun_angle = 0.75f; // The light's angular diameter, degrees (config "RTSunAngle"); the real sun is 0.53
   int g_rtsun_rays = 1; // Per pixel per frame (config "RTSunRays")
   // Only objects the game draws into its shadow maps block the RT sun (config "RTSunShadowCastersOnly"): the game keeps
   // some objects out of its shadow maps on purpose (window glass, light shaft meshes); camera only ones still occlude the AO
   bool g_rtsun_casters_only = true;
   // Shadow map check (config "RTSunShadowMapCheck"): a sun ray hitting an object the game never drew into its shadow map
   // (camera only: glass, light shafts, or a caster drawn only in passes we don't capture as casters) is blocked only
   // where the game's own shadow map has a surface there (or is in shadow), within "RTSunShadowMapTolerance" world units
   bool g_rtsun_shadow_consult = false;
   float g_rtsun_shadow_tolerance = 0.1f;
   int g_rtsun_output = int(RTSun::Output::Denoised); // Config "RTSunOutput"
   // ---- RT diagnostic capture (RTAO_NOTES.md "Diagnostic capture"): Luma panel button or Ctrl+Shift+Pause (not F9: the game's quickload). About 10 s:
   // 5 s as is, then 5 s with the ray tracing scene frozen, ~5 captures per second. Each capture copies, from the SAME
   // frame, the game's own screen shadow (its draw run again into a side target, without the RT input), the RT raw and
   // denoised visibility, the camera scene view (traced into a side target), the depth and the back buffer, and writes
   // them small (rt_debug_capture.h) with the frame's camera, light and scene counts. Every frame of the run also gets a
   // timeline line (did the lighting use the RT result, scene counts). Tests/rt_diag_analyze.cpp compares the captures.
   struct RTDiagStage
   {
      const char* name;
      RTDiag::Kind kind;
      int channel;
      com_ptr<ID3D11Texture2D> staging[2];
      DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
      uint32_t width = 0, height = 0;
      bool copied[2] = {};
   };
   struct RTDiagCapture
   {
      std::mutex mutex;
      std::atomic<bool> active = false;
      bool start_requested = false;
      std::string dir;
      std::chrono::steady_clock::time_point start, next;
      uint32_t index = 0; // Captures taken
      uint32_t frame = 0;
      int phase = 0; // 0 as is, 1 scene frozen
      bool saved_freeze = false, saved_openables = true, saved_young_long = false, saved_near_only = true;
      // Test matrix: 0 near slices + static retention, 1 near slices + old retention, 2 every slice + old retention
      int combo = 0;
      std::chrono::steady_clock::time_point switch_time;
      std::atomic<bool> reset_history = false; // The sun denoiser's history, at a switch (consumed by PrepareRTSun)
      // The pending capture per staging set: its number and metadata, read back at a later present
      int pending_set = -1;
      uint32_t pending_index = 0, pending_frame = 0;
      std::string pending_meta;
      std::vector<std::string> timeline;
      uint32_t last_bound = 0, last_stale = 0;
      uint64_t last_slices[8] = {};
      RTDiagStage stages[9] = {
         { "game_shadow", RTDiag::Kind::Gray, 1 }, // The game's own screen shadow (y of its target)
         { "rt_raw", RTDiag::Kind::Gray, 0 },
         { "rt_denoised", RTDiag::Kind::Gray, 0 }, // What the lighting reads
         { "scene", RTDiag::Kind::Rgb565, 0 },
         { "depth", RTDiag::Kind::DepthPacked, 0 },
         { "final", RTDiag::Kind::Color, 0 },
         { "scene_sun", RTDiag::Kind::Rgb565, 0 }, // The ray tracing scene seen from the sun (60 m around the camera)
         { "blocker", RTDiag::Kind::Id16, 0 }, // TLAS index + 1 of the first thing a hard sun ray hits (cNNNN_tlas.csv)
         { "blocker_all", RTDiag::Kind::Id16, 0 }, // The same ray with every instance (casters only mask off)
      };
      std::string status = "idle";
   };
   RTDiagCapture g_rtdiag;
   // Unattended capture (config "RTDiagAutoStartSeconds", 0 off): starts one diagnostic capture this many seconds after the
   // RT sun first traced with a copy of the game's shadow map (e.g. on the main menu's scene)
   int g_rtdiag_auto_seconds = 0;
   // Opt in (config "RTDiagAutoLevelSeconds", 0 off): one capture this many seconds into the first level of the session.
   // Tracing runs in stretches separated by loading screens (no RT pass for over kRTDiagLoadGapSeconds): the first
   // stretch is the main menu, a later one a loaded level. The timer restarts if another load comes first.
   int g_rtdiag_auto_level_seconds = 0;
   constexpr double kRTDiagLoadGapSeconds = 2.0;
   uint32_t g_rtdiag_trace_stretches = 0, g_rtdiag_last_trace_count = 0;
   std::chrono::steady_clock::time_point g_rtdiag_last_trace_time, g_rtdiag_stretch_start;
   bool g_rtdiag_auto_level_done = false;
   // Periodic shadow map reproduction score in the log (every kReproLogFrames frames, outside captures)
   constexpr uint32_t kReproLogFrames = 600;
   uint32_t g_repro_logged_trace = 0;
   bool g_rtdiag_auto_armed = false, g_rtdiag_auto_done = false;
   std::chrono::steady_clock::time_point g_rtdiag_auto_at;

   // Shadow pass audit: every shadow map draw, by slice, outcome (captured, or why not), shaders and index count, since
   // the last diagnostic capture wrote it (cNNNN_shadow_audit.csv)
   struct ShadowAuditKey
   {
      uint32_t slice;
      std::string outcome;
      uint32_t vs, ps, indices;
      bool operator<(const ShadowAuditKey& o) const { return std::tie(slice, outcome, vs, ps, indices) < std::tie(o.slice, o.outcome, o.vs, o.ps, o.indices); }
   };
   std::mutex g_shadow_audit_mutex;
   // Draw probe (diagnostic capture): every indexed draw of the capture, by mesh key (as the TLAS csv) and render state,
   // written as draws.csv at the end: which passes draw a mesh, blended or not, into which targets
   std::mutex g_draw_probe_mutex;
   std::map<std::string, uint64_t> g_draw_probe;
   std::atomic<bool> g_draw_probe_active = false;
   std::mutex g_copy_probe_mutex;
   std::map<std::string, uint64_t> g_copy_probe; // Copies between depth textures during a capture (copies.csv)
   // Shadow maps the game renders into a plain (non array) square depth texture >= 1024 (config "RTAOShadowFlatTargets"):
   // part of the sun's shadow content (window trim) is drawn there. Their draws get slice kFlatShadowSlice.
   bool g_rtao_shadow_flat = true;
   constexpr uint32_t kFlatShadowSlice = 8;
   // Skinned shadow map draws (config "RTAOShadowSkinned"): many sun casters are drawn with skinned shaders
   bool g_rtao_shadow_skinned = true;
   // The game's shadow map rasterizer bias (last shadow draw): the reproduction allows for it
   std::atomic<int> g_shadow_depth_bias = 0;
   std::atomic<float> g_shadow_slope_bias = 0.f;
   std::atomic<uint32_t> g_shadow_depth_format = 0; // DXGI format of the shadow map's depth view
   std::map<ShadowAuditKey, uint64_t> g_shadow_audit;
   // ~20 s cycling the 4 test combos in 1 s blocks (the order rotates every 4 blocks, so each combo sees similar camera
   // motion), then 4 s with the ray tracing scene frozen. Captures within kRTDiagSettle of a switch are tagged settle=1.
   constexpr double kRTDiagMatrixSeconds = 20.0, kRTDiagFrozenSeconds = 4.0, kRTDiagBlock = 1.0, kRTDiagSettle = 0.3, kRTDiagInterval = 0.2;

   std::mutex g_rtsun_status_mutex;
   // The RT sun output is one of the debug scene views (a color image of the ray tracing scene, shown fullscreen)
   bool RTSunSceneView()
   {
      return g_rtsun_mode != RTSUN_MODE_OFF && (g_rtsun_output == int(RTSun::Output::SceneView) || g_rtsun_output == int(RTSun::Output::SceneSun));
   }
   std::string g_rtsun_status;

   void SetRTSunStatus(std::string status, bool error)
   {
      reshade::log::message(error ? reshade::log::level::error : reshade::log::level::info, ("RT sun: " + status).c_str());
      const std::lock_guard lock(g_rtsun_status_mutex);
      g_rtsun_status = std::move(status);
   }

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
      bool sun = false; // Also trace the RT sun shadows
      RTSun::SunSettings sun_settings = {};
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
   D3D12_GPU_VIRTUAL_ADDRESS rtao_last_top = 0; // The last TLAS built (recording thread only), for "Freeze ray tracing scene"
   // GPU time split (us): capture (readback, BLAS builds, per frame meshes), TLAS build, trace, denoise
   std::atomic<uint32_t> rtao_gpu_capture_us = 0, rtao_gpu_tlas_us = 0, rtao_gpu_trace_us = 0, rtao_gpu_filter_us = 0;
   int rtao_last_quality = -1;
   std::atomic<uint32_t> rtao_tlas_cutouts = 0;
   std::atomic<uint32_t> rtao_tlas_missing_blas = 0; // Retained instances left out of the last TLAS (BLAS not built yet)

   // RT sun shadows (after the RTAO objects, so they are destroyed first)
   RTSun::SunPass rtsun_pass;
   RTSun::LightCapture rtsun_light;
   bool rtsun_failed = false; // Init failed (the pass stays off)
   int rtsun_last_output = -1;
   // rtao_frame of the last SSAO draw that queued a sun trace (the screen shadows draw only uses a fresh one)
   std::atomic<uint32_t> rtsun_scheduled_frame = UINT32_MAX / 2;
   std::atomic<uint32_t> rtsun_gpu_us = 0, rtsun_record_failures = 0, rtsun_bound_draws = 0, rtsun_stale_draws = 0;
   // Shadow map check: the frame the game's cascades were last copied (UINT32_MAX: never), its compare direction
   std::atomic<uint32_t> rtsun_shadow_copied_frame = UINT32_MAX;
   std::atomic<bool> rtsun_shadow_reversed = false;
   std::atomic<uint64_t> rtsun_shadow_copies = 0;
   std::atomic<bool> rtsun_shadow_logged_error = false;
   std::atomic<bool> rtsun_logged_error = false;
   float rtsun_logged_dir[3] = {};
   std::atomic<uint64_t> rtao_cutout_tests = 0, rtao_cutout_hits = 0, rtao_cutout_fallback_tests = 0; // Trace candidates last frame (read back late)
   // RT diagnostic capture (see g_rtdiag): the game's own screen shadow of the frame (side target), the camera of the SSAO draw
   com_ptr<ID3D11Texture2D> rtdiag_game_shadow;
   com_ptr<ID3D11RenderTargetView> rtdiag_game_shadow_rtv;
   std::atomic<uint32_t> rtdiag_game_shadow_frame = UINT32_MAX;
   float rtdiag_view_proj[16] = {}, rtdiag_cam_pos[3] = {}, rtdiag_cam_fwd[3] = {};
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
         reshade::register_event<reshade::addon_event::update_buffer_region>(Dishonored2::OnRTAOUpdateBuffer);
         reshade::register_event<reshade::addon_event::update_buffer_region_command>(Dishonored2::OnRTAOUpdateBufferCommand);
         reshade::register_event<reshade::addon_event::copy_resource>(Dishonored2::OnRTAOCopyResource);
         reshade::register_event<reshade::addon_event::copy_texture_region>(Dishonored2::OnRTAOCopyTextureRegion);
         reshade::register_event<reshade::addon_event::copy_buffer_region>(Dishonored2::OnRTAOCopyBufferRegion);
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
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTSun Shadow Copy PS"), ShaderDefinition{ "Luma_RTSunShadowCopy", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "shadow_copy_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTSun Shadow Pack PS"), ShaderDefinition{ "Luma_RTSunShadowCopy", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "shadow_pack_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 AO View PS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, "ao_view_ps" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Capture CS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::compute_shader, nullptr, "capture_instances_cs" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Skin CS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::compute_shader, nullptr, "skin_vertices_cs" });
      native_shaders_definitions.emplace(CompileTimeStringHash("DS2 RTAO Alpha Strip CS"), ShaderDefinition{ "Luma_RTAO", reshade::api::pipeline_subobject_type::compute_shader, nullptr, "alpha_strip_cs" });
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
      game_device_data.rtao_capture.skinning_enabled = g_rtao_skinned;
      game_device_data.rtao_capture.dynamic_enabled = g_rtao_dynamic;
      game_device_data.rtao_capture.cutouts_exact = g_rtao_cutouts_exact;
      game_device_data.rtao_capture.dedup_enabled = g_rtao_dedup;
      game_device_data.rtao_capture.shadow_skinned_enabled = g_rtao_shadow_skinned;
      game_device_data.rtao_capture.skinned_hold_frames = uint32_t(std::clamp(g_rtao_skinned_hold, 0, 600));
      game_device_data.rtao_capture.shadow_frame_meshes_enabled = g_rtdbg_shadow_openables;
      game_device_data.rtao_tlas.shadow_mode = g_rtdbg_shadow_young_long ? 1u : 0u;
      game_device_data.rtao_tlas.camera_only_non_opaque = g_rtsun_shadow_consult;
      game_device_data.rtao_last_draw_frame = game_device_data.rtao_frame;
      g_rtao_main_depth_width = depth_desc.Width;
      g_rtao_main_depth_height = depth_desc.Height;
      RTAO::PassSettings settings = MakeRTAOSettings(output);
      std::memcpy(game_device_data.rtdiag_view_proj, settings.view_proj, sizeof(settings.view_proj)); // RT diagnostic capture
      std::memcpy(game_device_data.rtdiag_cam_pos, settings.cam_pos, sizeof(settings.cam_pos));
      std::memcpy(game_device_data.rtdiag_cam_fwd, settings.cam_fwd, sizeof(settings.cam_fwd));
      settings.reset_history = output != game_device_data.rtao_last_output || g_rtao_quality != game_device_data.rtao_last_quality;
      game_device_data.rtao_last_output = output;
      game_device_data.rtao_last_quality = g_rtao_quality;

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

      RTSun::SunSettings sun_settings;
      const bool sun = ray_traced && PrepareRTSun(native_device, game_device_data, settings, ao_desc.Width, ao_desc.Height, sun_settings);

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
         const bool ok = ray_traced ? rtao.Run(native_device_context, MakeRTAORecorder(game_device_data, settings, sun ? &sun_settings : nullptr), error) : rtao.Run(native_device_context, constants, error);
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
         if (sun)
            game_device_data.rtsun_scheduled_frame = game_device_data.rtao_frame;
      }
      else
      {
         GameDeviceDataDishonored2::RTAOPendingSplit pending;
         pending.deferred_context = native_device_context;
         pending.constants = constants;
         pending.settings = settings;
         pending.output = output;
         pending.frame = game_device_data.rtao_frame;
         pending.sun = sun;
         pending.sun_settings = sun_settings;
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
         if (sun)
            game_device_data.rtsun_scheduled_frame = game_device_data.rtao_frame;
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

   // The game's screen shadows draw again, without the RT input (the replaced shader then computes the game's own shadow),
   // into a side copy of its render target: stage "game_shadow" of the diagnostic capture. Render thread of that draw.
   static void RTDiagDrawGameShadow(ID3D11Device* native_device, ID3D11DeviceContext* context, GameDeviceDataDishonored2& g, std::function<void()>* original_draw_dispatch_func)
   {
      ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
      ID3D11DepthStencilView* dsv = nullptr;
      context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, &dsv);
      if (rtvs[0])
      {
         com_ptr<ID3D11Resource> resource;
         rtvs[0]->GetResource(&resource);
         com_ptr<ID3D11Texture2D> texture;
         D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
         rtvs[0]->GetDesc(&rtv_desc);
         if (resource && SUCCEEDED(resource->QueryInterface(&texture)))
         {
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            ID3D11RenderTargetView* side_rtv = nullptr;
            {
               const std::lock_guard lock(g_rtdiag.mutex);
               D3D11_TEXTURE2D_DESC have = {};
               if (g.rtdiag_game_shadow)
                  g.rtdiag_game_shadow->GetDesc(&have);
               if (!g.rtdiag_game_shadow || have.Width != desc.Width || have.Height != desc.Height || have.Format != desc.Format)
               {
                  g.rtdiag_game_shadow.reset();
                  g.rtdiag_game_shadow_rtv.reset();
                  D3D11_TEXTURE2D_DESC side = desc;
                  side.MipLevels = 1;
                  side.ArraySize = 1;
                  side.SampleDesc = { 1, 0 };
                  side.Usage = D3D11_USAGE_DEFAULT;
                  side.BindFlags = D3D11_BIND_RENDER_TARGET;
                  side.CPUAccessFlags = 0;
                  side.MiscFlags = 0;
                  if (SUCCEEDED(native_device->CreateTexture2D(&side, nullptr, &g.rtdiag_game_shadow)))
                     native_device->CreateRenderTargetView(g.rtdiag_game_shadow.get(), &rtv_desc, &g.rtdiag_game_shadow_rtv);
               }
               side_rtv = g.rtdiag_game_shadow_rtv.get();
               if (side_rtv)
                  side_rtv->AddRef();
            }
            if (side_rtv)
            {
               ID3D11RenderTargetView* bound[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
               std::memcpy(bound, rtvs, sizeof(bound));
               bound[0] = side_rtv;
               context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, bound, dsv);
               ID3D11ShaderResourceView* no_rt = nullptr;
               context->PSSetShaderResources(3, 1, &no_rt);
               (*original_draw_dispatch_func)();
               context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, dsv);
               side_rtv->Release();
               g.rtdiag_game_shadow_frame = g.rtao_frame;
            }
         }
      }
      for (auto* rtv : rtvs)
         if (rtv)
            rtv->Release();
      if (dsv)
         dsv->Release();
   }

   static ID3D11Resource* RTDiagResource(ID3D11View* view)
   {
      if (!view)
         return nullptr;
      ID3D11Resource* resource = nullptr;
      view->GetResource(&resource);
      view->Release();
      return resource; // AddRef'd
   }

   // Present (immediate context): hotkey, phases, the copies of a capture, the readback of the previous one, the timeline
   static void RTDiagOnPresent(ID3D11Device* native_device, DeviceData& device_data, GameDeviceDataDishonored2& g)
   {
      static bool hotkey_down = false;
      const bool down = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState(VK_PAUSE) & 0x8000);
      if (down && !hotkey_down && !g_rtdiag.active)
         g_rtdiag.start_requested = true;
      hotkey_down = down;

      com_ptr<ID3D11DeviceContext> context;
      native_device->GetImmediateContext(&context);
      const auto now = std::chrono::steady_clock::now();

      // Tracing stretches (menu, then levels after loading screens)
      {
         const uint32_t traces = g.rtsun_pass.TraceCount();
         if (traces != g_rtdiag_last_trace_count)
         {
            if (g_rtdiag_trace_stretches == 0 || std::chrono::duration<double>(now - g_rtdiag_last_trace_time).count() > kRTDiagLoadGapSeconds)
            {
               g_rtdiag_trace_stretches++;
               g_rtdiag_stretch_start = now;
               reshade::log::message(reshade::log::level::info, ("RT diag: tracing stretch " + std::to_string(g_rtdiag_trace_stretches) + (g_rtdiag_trace_stretches == 1 ? " (main menu)" : " (after a load)")).c_str());
            }
            g_rtdiag_last_trace_count = traces;
            g_rtdiag_last_trace_time = now;
         }
         if (g_rtdiag_auto_level_seconds > 0 && !g_rtdiag_auto_level_done && !g_rtdiag.active && g_rtdiag_trace_stretches >= 2 &&
            std::chrono::duration<double>(now - g_rtdiag_last_trace_time).count() < 0.5 &&
            std::chrono::duration<double>(now - g_rtdiag_stretch_start).count() >= double(g_rtdiag_auto_level_seconds))
         {
            g_rtdiag_auto_level_done = true;
            g_rtdiag.start_requested = true;
            reshade::log::message(reshade::log::level::info, "RT diag: level capture starting (RTDiagAutoLevelSeconds)");
         }
      }
      if (g_rtdiag_auto_seconds > 0 && !g_rtdiag_auto_done && !g_rtdiag.active && g.rtsun_pass.TraceCount() > 0 && g.rtsun_shadow_copies.load() > 0)
      {
         if (!g_rtdiag_auto_armed)
         {
            g_rtdiag_auto_armed = true;
            g_rtdiag_auto_at = now + std::chrono::seconds(g_rtdiag_auto_seconds);
            reshade::log::message(reshade::log::level::info, ("RT diag: unattended capture in " + std::to_string(g_rtdiag_auto_seconds) + " s").c_str());
         }
         else if (now >= g_rtdiag_auto_at)
         {
            g_rtdiag_auto_done = true;
            g_rtdiag.start_requested = true;
         }
      }

      if (g_rtdiag.start_requested && !g_rtdiag.active)
      {
         g_rtdiag.start_requested = false;
         wchar_t exe[MAX_PATH] = {};
         GetModuleFileNameW(nullptr, exe, MAX_PATH);
         std::filesystem::path dir = std::filesystem::path(exe).parent_path() / "RTDiag";
         const std::time_t t = std::time(nullptr);
         std::tm tm = {};
         localtime_s(&tm, &t);
         char stamp[32];
         std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);
         dir /= stamp;
         std::error_code ec;
         std::filesystem::create_directories(dir, ec);
         if (ec)
         {
            g_rtdiag.status = "could not create " + dir.string();
            return;
         }
         g_rtdiag.dir = dir.string();
         g_rtdiag.start = now;
         g_rtdiag.next = now + std::chrono::milliseconds(500); // Lets the debug targets come up
         g_rtdiag.index = 0;
         g_rtdiag.frame = 0;
         g_rtdiag.phase = 0;
         g_rtdiag.saved_freeze = g_rtao_freeze_scene;
         g_rtdiag.saved_openables = g_rtdbg_shadow_openables;
         g_rtdiag.saved_young_long = g_rtdbg_shadow_young_long;
         g_rtdiag.saved_near_only = g_rtao_shadow_near_only;
         g_rtdiag.combo = -1;
         g_rtdiag.pending_set = -1;
         g_rtdiag.timeline.clear();
         g_rtdiag.timeline.push_back("frame,t_ms,phase,combo,frozen,screen_shadows_rt,screen_shadows_game,tlas_instances,retained,new,evicted_promoted,evicted_young,frame_meshes,movables_kept,cam_x,cam_y,cam_z,fwd_x,fwd_y,fwd_z,slice0,slice1,slice2,slice3,slice4,slice5,slice6,slice7");
         for (int i = 0; i < 8; i++)
            g_rtdiag.last_slices[i] = g_rtao_shadow_slice_draws[i];
         g_rtdiag.last_bound = g.rtsun_bound_draws;
         g_rtdiag.last_stale = g.rtsun_stale_draws;
         g_rtdiag.active = true;
         {
            const std::lock_guard lock(g_draw_probe_mutex);
            g_draw_probe.clear();
         }
         g_draw_probe_active = true;
         g_rtdiag.status = "capturing into " + g_rtdiag.dir;
         reshade::log::message(reshade::log::level::info, ("RT diag: capture started, " + g_rtdiag.dir).c_str());
      }
      if (!g_rtdiag.active)
         return;

      const double t = std::chrono::duration<double>(now - g_rtdiag.start).count();
      const RTAO::CaptureStats stats = g.rtao_capture.GetStats();
      const uint32_t bound = g.rtsun_bound_draws, stale = g.rtsun_stale_draws;
      const uint32_t bound_now = bound - g_rtdiag.last_bound, stale_now = stale - g_rtdiag.last_stale;
      g_rtdiag.last_bound = bound;
      g_rtdiag.last_stale = stale;
      {
         char line[512];
         const RTAO::RetentionStats& r = stats.retention;
         snprintf(line, sizeof(line), "%u,%.1f,%d,%d,%d,%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f", g_rtdiag.frame, t * 1000.0, g_rtdiag.phase, g_rtdiag.combo, int(g_rtao_freeze_scene),
            bound_now, stale_now, g.rtao_tlas_instances.load(), r.promoted + r.young, r.added_new, r.evicted_promoted, r.evicted_young, (unsigned long long)stats.frame_draws,
            (unsigned long long)stats.frame_retained, g.rtdiag_cam_pos[0], g.rtdiag_cam_pos[1], g.rtdiag_cam_pos[2], g.rtdiag_cam_fwd[0], g.rtdiag_cam_fwd[1], g.rtdiag_cam_fwd[2]);
         std::string row = line;
         for (int i = 0; i < 8; i++)
         {
            const uint64_t total = g_rtao_shadow_slice_draws[i];
            row += "," + std::to_string(total - g_rtdiag.last_slices[i]);
            g_rtdiag.last_slices[i] = total;
         }
         g_rtdiag.timeline.push_back(row);
      }

      // Read back the pending capture (copied at least one present ago) and write it on a worker thread
      if (g_rtdiag.pending_set >= 0 && g_rtdiag.frame > g_rtdiag.pending_frame)
      {
         const int set = g_rtdiag.pending_set;
         struct Out
         {
            std::string name;
            RTDiag::Image image;
         };
         auto outs = std::make_shared<std::vector<Out>>();
         std::string formats;
         for (RTDiagStage& stage : g_rtdiag.stages)
         {
            if (!stage.copied[set] || !stage.staging[set])
               continue;
            stage.copied[set] = false;
            D3D11_MAPPED_SUBRESOURCE mapped = {};
            if (FAILED(context->Map(stage.staging[set].get(), 0, D3D11_MAP_READ, 0, &mapped)))
               continue;
            const bool scrgb = stage.kind == RTDiag::Kind::Color && stage.format == DXGI_FORMAT_R16G16B16A16_FLOAT;
            RTDiag::Image image = RTDiag::Convert(mapped, stage.format, stage.width, stage.height, stage.kind, stage.channel, scrgb);
            context->Unmap(stage.staging[set].get(), 0);
            formats += std::string("format_") + stage.name + "=" + std::to_string(int(stage.format)) + "\n";
            formats += std::string("size_") + stage.name + "=" + std::to_string(image.width) + "," + std::to_string(image.height) + "\n";
            if (image.width)
               outs->push_back({ stage.name, std::move(image) });
         }
         char prefix[32];
         snprintf(prefix, sizeof(prefix), "c%04u", g_rtdiag.pending_index);
         const std::string base = (std::filesystem::path(g_rtdiag.dir) / prefix).string();
         const std::string meta = g_rtdiag.pending_meta + formats;
         std::thread([outs, base, meta]() {
            std::ofstream(base + ".txt") << meta;
            for (const Out& out : *outs)
            {
               const std::string file = base + "_" + out.name;
               const RTDiag::Image& im = out.image;
               if (!im.u16.empty())
               {
                  std::ofstream(file + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(im.u16.data()), im.u16.size() * sizeof(uint16_t));
                  stbi_write_png((file + ".png").c_str(), int(im.width), int(im.height), 1, im.u8.data(), int(im.width));
               }
               else if (im.channels == 0)
               {
                  std::ofstream(file + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(im.f32.data()), im.f32.size() * sizeof(float));
                  // Viewable: reversed Z device depth, sqrt for contrast
                  std::vector<uint8_t> gray(im.f32.size());
                  for (size_t i = 0; i < gray.size(); i++)
                     gray[i] = uint8_t(std::lround(std::sqrt(std::clamp(im.f32[i] * 20.f, 0.f, 1.f)) * 255.f));
                  stbi_write_png((file + ".png").c_str(), int(im.width), int(im.height), 1, gray.data(), int(im.width));
               }
               else
               {
                  std::ofstream(file + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(im.u8.data()), im.u8.size());
                  stbi_write_png((file + ".png").c_str(), int(im.width), int(im.height), int(im.channels), im.u8.data(), int(im.width * im.channels));
               }
            }
         }).detach();
         g_rtdiag.pending_set = -1;
      }

      // Phases: the test matrix, then the frozen scene
      if (t >= kRTDiagMatrixSeconds + kRTDiagFrozenSeconds)
      {
         g_rtao_freeze_scene = g_rtdiag.saved_freeze;
         g_rtdbg_shadow_openables = g_rtdiag.saved_openables;
         g_rtdbg_shadow_young_long = g_rtdiag.saved_young_long;
         g_rtao_shadow_near_only = g_rtdiag.saved_near_only;
         g_rtdiag.active = false;
         g_draw_probe_active = false;
         {
            std::map<std::string, uint64_t> probe;
            {
               const std::lock_guard lock(g_draw_probe_mutex);
               probe.swap(g_draw_probe);
            }
            std::string csv = "mesh_key,ib,first_index,index_count,base_vertex,vs_hash,ps_hash,depth_enable,depth_write,depth_func,blend,src_blend,dst_blend,write_mask,alpha_to_coverage,rtv_count,rtv0_format,dsv_slice,depth_bias,slope_bias,cull,vp_x,vp_y,vp_w,vp_h,dynamic_vb,dsv_resource,dsv_w,dsv_h,dsv_array,dsv_format,draws\n";
            for (const auto& pair : probe)
               csv += pair.first + "," + std::to_string(pair.second) + "\n";
            std::ofstream(std::filesystem::path(g_rtdiag.dir) / "draws.csv") << csv;
            std::map<std::string, uint64_t> copies;
            {
               const std::lock_guard lock(g_copy_probe_mutex);
               copies.swap(g_copy_probe);
            }
            std::string copy_csv = "kind,src,src_w,src_h,src_array,src_format,src_sub,dst,dst_w,dst_h,dst_array,dst_format,dst_sub,count\n";
            for (const auto& pair : copies)
               copy_csv += pair.first + "," + std::to_string(pair.second) + "\n";
            std::ofstream(std::filesystem::path(g_rtdiag.dir) / "copies.csv") << copy_csv;
         }
         std::ofstream(std::filesystem::path(g_rtdiag.dir) / "timeline.csv") << [&]() {
            std::string all;
            for (const std::string& l : g_rtdiag.timeline)
               all += l + "\n";
            return all;
         }();
         g_rtdiag.status = "done: " + std::to_string(g_rtdiag.index) + " captures in " + g_rtdiag.dir;
         reshade::log::message(reshade::log::level::info, ("RT diag: " + g_rtdiag.status).c_str());
         g_rtdiag.frame++;
         return;
      }
      // The current settings as they are, then the scene frozen. No switching and no resets: the game keeps its shadow
      // map content (draws an object into it once), so ray tracing scene objects taken from it can't be dropped and
      // recovered (the 54b8f2b8 capture: 892 -> 292 instances, never back).
      const bool frozen = t >= kRTDiagMatrixSeconds;
      if (int(frozen) != g_rtdiag.phase)
         g_rtdiag.switch_time = now;
      g_rtdiag.combo = 0;
      g_rtdiag.phase = frozen ? 1 : 0;
      g_rtao_freeze_scene = frozen ? true : g_rtdiag.saved_freeze;

      // A capture: copy every stage of this frame into the free staging set
      if (now >= g_rtdiag.next && g_rtdiag.pending_set < 0)
      {
         g_rtdiag.next = now + std::chrono::milliseconds(int(kRTDiagInterval * 1000.0));
         const int set = int(g_rtdiag.index & 1);
         com_ptr<ID3D11Resource> sources[9];
         {
            const std::lock_guard lock(g_rtdiag.mutex);
            // Drawn this frame (rtao_frame has already advanced at present)
            if (g.rtdiag_game_shadow && g.rtdiag_game_shadow_frame + 1 >= g.rtao_frame)
               sources[0] = g.rtdiag_game_shadow.get();
         }
         sources[1] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtsun_pass.GetDebugSRV(0)), true);
         sources[2] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtsun_pass.GetSRV()), true);
         sources[3] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtsun_pass.GetDebugSRV(1)), true);
         sources[6] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtsun_pass.GetDebugSRV(2)), true);
         sources[7] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtsun_pass.GetDebugSRV(3)), true);
         sources[8] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtsun_pass.GetDebugSRV(4)), true);
         if (g.rtao.IsInitialized())
            sources[4] = com_ptr<ID3D11Resource>(RTDiagResource(g.rtao.GetDepthRTV()), true);
         {
            const std::shared_lock lock(device_data.mutex);
            if (!device_data.swapchains.empty())
            {
               IDXGISwapChain* swapchain = (IDXGISwapChain*)((*device_data.swapchains.begin())->get_native());
               com_ptr<ID3D11Texture2D> back_buffer;
               if (swapchain && SUCCEEDED(swapchain->GetBuffer(0, IID_PPV_ARGS(&back_buffer))))
                  sources[5] = back_buffer.get();
            }
         }
         int copied = 0;
         for (int s = 0; s < 9; s++)
         {
            RTDiagStage& stage = g_rtdiag.stages[s];
            if (!sources[s])
               continue;
            com_ptr<ID3D11Texture2D> texture;
            if (FAILED(sources[s]->QueryInterface(&texture)))
               continue;
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            D3D11_TEXTURE2D_DESC have = {};
            if (stage.staging[set])
               stage.staging[set]->GetDesc(&have);
            if (!stage.staging[set] || have.Width != desc.Width || have.Height != desc.Height || have.Format != desc.Format)
            {
               stage.staging[set].reset();
               D3D11_TEXTURE2D_DESC staging = desc;
               staging.MipLevels = 1;
               staging.ArraySize = 1;
               staging.SampleDesc = { 1, 0 };
               staging.Usage = D3D11_USAGE_STAGING;
               staging.BindFlags = 0;
               staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
               staging.MiscFlags = 0;
               if (FAILED(native_device->CreateTexture2D(&staging, nullptr, &stage.staging[set])))
                  continue;
            }
            if (desc.MipLevels == 1 && desc.ArraySize == 1 && desc.SampleDesc.Count == 1)
               context->CopyResource(stage.staging[set].get(), texture.get());
            else
               context->CopySubresourceRegion(stage.staging[set].get(), 0, 0, 0, 0, texture.get(), 0, nullptr);
            stage.format = desc.Format;
            stage.width = desc.Width;
            stage.height = desc.Height;
            stage.copied[set] = true;
            copied++;
         }
         if (copied)
         {
            const RTAO::RetentionStats& r = stats.retention;
            std::string meta;
            char line[256];
            auto add = [&](const char* format, auto... args) {
               snprintf(line, sizeof(line), format, args...);
               meta += line;
            };
            add("index=%u\nframe=%u\nt_ms=%.1f\nphase=%d\nfrozen=%d\n", g_rtdiag.index, g_rtdiag.frame, t * 1000.0, g_rtdiag.phase, int(g_rtao_freeze_scene));
            add("view_proj=");
            for (int i = 0; i < 16; i++)
               add(i ? ",%.9g" : "%.9g", g.rtdiag_view_proj[i]);
            add("\ncam_pos=%.6g,%.6g,%.6g\ncam_fwd=%.6g,%.6g,%.6g\n", g.rtdiag_cam_pos[0], g.rtdiag_cam_pos[1], g.rtdiag_cam_pos[2], g.rtdiag_cam_fwd[0], g.rtdiag_cam_fwd[1], g.rtdiag_cam_fwd[2]);
            float light[3] = {};
            g.rtsun_light.GetDirection(light);
            add("light=%.6g,%.6g,%.6g\nrt_output=%d\nrt_mode=%d\n", light[0], light[1], light[2], g_rtsun_output, g_rtsun_mode);
            add("screen_shadows_rt=%u\nscreen_shadows_game=%u\ngame_shadow_captured=%d\n", bound_now, stale_now, int(bool(sources[0])));
            add("tlas_instances=%u\nretained=%u\npromoted=%u\nyoung=%u\nmain_only=%u\nshadow_only=%u\nboth=%u\nnew=%u\nevicted=%u\n", g.rtao_tlas_instances.load(), r.promoted + r.young, r.promoted, r.young,
               r.main_only, r.shadow_only, r.both, r.added_new, r.evicted_promoted + r.evicted_young);
            add("frame_meshes=%llu\nmovables_kept=%llu\n", (unsigned long long)stats.frame_draws, (unsigned long long)stats.frame_retained);
            float fars[4] = {};
            g.rtsun_light.GetSliceFars(fars);
            add("slice_fars=%.4g,%.4g,%.4g,%.4g\n", fars[0], fars[1], fars[2], fars[3]);
            add("shadow_bursts=%u\nlast_burst_frame=%u\nlast_burst_size=%u\ndynamic_meshes=%u\ndynamic_dropped=%u\nshadow_static=%u\n", r.bursts, r.last_burst_frame, r.last_burst_size,
               r.dynamic_meshes, r.dynamic_dropped, r.shadow_static);
            // Shadow pass audit since the last capture
            {
               std::map<ShadowAuditKey, uint64_t> audit;
               {
                  const std::lock_guard lock(g_shadow_audit_mutex);
                  audit.swap(g_shadow_audit);
               }
               std::string csv = "slice,outcome,vs_hash,ps_hash,index_count,draws\n";
               for (const auto& pair : audit)
               {
                  char row[160];
                  snprintf(row, sizeof(row), "%u,%s,%08X,%08X,%u,%llu\n", pair.first.slice, pair.first.outcome.c_str(), pair.first.vs, pair.first.ps, pair.first.indices, (unsigned long long)pair.second);
                  csv += row;
               }
               char name[40];
               snprintf(name, sizeof(name), "c%04u_shadow_audit.csv", g_rtdiag.index);
               const std::string path = (std::filesystem::path(g_rtdiag.dir) / name).string();
               std::thread([csv = std::move(csv), path]() { std::ofstream(path) << csv; }).detach();
            }
            // Shadow map reproduction: the latest read back (RTSun::kReproCascades x kReproGrid^2 uints, see rtsun_trace.hlsl)
            {
               std::vector<uint32_t> repro;
               uint32_t repro_trace = 0;
               if (g.rtsun_pass.GetShadowRepro(repro, repro_trace))
               {
                  uint64_t counts[RTSun::kReproCascades][2][3] = {};
                  const size_t per = size_t(RTSun::kReproGrid) * RTSun::kReproGrid;
                  for (size_t i = 0; i < repro.size(); i++)
                  {
                     const size_t c = i / per;
                     counts[c][0][repro[i] & 3]++;
                     counts[c][1][(repro[i] >> 2) & 3]++;
                  }
                  add("repro_trace=%u\nrepro_now=%u\n", repro_trace, g.rtsun_pass.TraceCount());
                  for (uint32_t c = 0; c < RTSun::kReproCascades; c++)
                     add("repro_c%u=shadow_only match %llu missing %llu extra %llu | all match %llu missing %llu extra %llu\n", c,
                        (unsigned long long)counts[c][0][0], (unsigned long long)counts[c][0][1], (unsigned long long)counts[c][0][2],
                        (unsigned long long)counts[c][1][0], (unsigned long long)counts[c][1][1], (unsigned long long)counts[c][1][2]);
                  char name[40];
                  snprintf(name, sizeof(name), "c%04u_repro.bin", g_rtdiag.index);
                  const std::string path = (std::filesystem::path(g_rtdiag.dir) / name).string();
                  std::thread([repro = std::move(repro), path]() { std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(repro.data()), std::streamsize(repro.size() * 4)); }).detach();
               }
            }
            // Every retained TLAS instance of this frame, for the duplicate check (cNNNN_instances.csv)
            {
               std::string csv = "blas_key,sources,min_slice,x,y,z,index_count,vertex_count,first_frame,last_frame,seen_frames,vs_hash,ps_hash\n";
               char row[192];
               for (const auto& i : g.rtao_tlas.LastBuildOrder()) // In TLAS index order: row r = blocker id r + 1
               {
                  uint32_t indices = 0, vertices = 0, vs_hash = 0, ps_hash = 0;
                  g.rtao_capture.GetMeshSize(i.blas_key, indices, vertices, &vs_hash, &ps_hash);
                  snprintf(row, sizeof(row), "%016llX,%u,%u,%.3f,%.3f,%.3f,%u,%u,%u,%u,%u,%08X,%08X\n", (unsigned long long)i.blas_key, i.sources, i.min_slice, i.pos[0], i.pos[1], i.pos[2], indices, vertices, i.first_frame, i.last_frame, i.seen_frames, vs_hash, ps_hash);
                  csv += row;
               }
               char name[32];
               snprintf(name, sizeof(name), "c%04u_tlas.csv", g_rtdiag.index);
               const std::string path = (std::filesystem::path(g_rtdiag.dir) / name).string();
               std::thread([csv = std::move(csv), path]() { std::ofstream(path) << csv; }).detach();
            }
            add("casters_only=%d\nskinned_hold=%d\nafter_ao_records=%llu\nafter_ao_redraw_kept=%llu\n", int(g_rtsun_casters_only), g_rtao_skinned_hold,
               (unsigned long long)stats.after_ao_records, (unsigned long long)stats.after_ao_redraw_kept);
            add("shadow_bias=%d\nshadow_slope_bias=%.3f\nshadow_depth_format=%u\nshadow_records_held=%llu\nshadow_records_lost=%llu\nrecords_deferred=%llu\n", g_shadow_depth_bias.load(), g_shadow_slope_bias.load(), g_shadow_depth_format.load(),
               (unsigned long long)stats.shadow_records_held, (unsigned long long)stats.shadow_records_lost, (unsigned long long)stats.records_deferred);
            add("shadow_check=%d\nshadow_check_tolerance=%.3f\nshadow_copied_frame=%u\nshadow_reversed=%d\nshadow_copies=%llu\n", int(g_rtsun_shadow_consult), g_rtsun_shadow_tolerance,
               g.rtsun_shadow_copied_frame.load(), int(g.rtsun_shadow_reversed.load()), (unsigned long long)g.rtsun_shadow_copies.load());
            // Per frame meshes of the last complete frame: per mesh, its events in order (see RTAO::FrameMeshEvent)
            {
               const std::vector<RTAO::FrameMeshEvent> trace = g.rtao_capture.GetLastFrameMeshTrace();
               std::vector<std::pair<uint64_t, std::string>> per_mesh;
               for (const RTAO::FrameMeshEvent& e : trace)
               {
                  auto it = std::find_if(per_mesh.begin(), per_mesh.end(), [&](const auto& p) { return p.first == e.key; });
                  if (it == per_mesh.end())
                  {
                     per_mesh.emplace_back(e.key, e.skinned ? "skin" : "dyn");
                     it = per_mesh.end() - 1;
                  }
                  char step[8];
                  snprintf(step, sizeof(step), " %c%02X", e.event, e.context);
                  it->second += step;
               }
               for (const auto& p : per_mesh)
                  add("frame_mesh=%016llX %s\n", (unsigned long long)p.first, p.second.c_str());
            }
            add("near_only=%d\nnear_slices=%d\nfar_skipped=%llu\n", int(g_rtao_shadow_near_only), g_rtao_shadow_near_slices, (unsigned long long)g_rtao_shadow_far_skipped.load());
            add("slice_draws=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", (unsigned long long)g_rtao_shadow_slice_draws[0].load(), (unsigned long long)g_rtao_shadow_slice_draws[1].load(),
               (unsigned long long)g_rtao_shadow_slice_draws[2].load(), (unsigned long long)g_rtao_shadow_slice_draws[3].load(), (unsigned long long)g_rtao_shadow_slice_draws[4].load(),
               (unsigned long long)g_rtao_shadow_slice_draws[5].load(), (unsigned long long)g_rtao_shadow_slice_draws[6].load(), (unsigned long long)g_rtao_shadow_slice_draws[7].load());
            add("test_shadow_openables=%d\ntest_shadow_young_long=%d\ncombo=%d\nsettle=%d\n", int(g_rtdbg_shadow_openables), int(g_rtdbg_shadow_young_long), g_rtdiag.combo,
               int(std::chrono::duration<double>(now - g_rtdiag.switch_time).count() < kRTDiagSettle));
            // Per frame mesh probes: key, flags (1 from a shadow map, 2 skinned), vertex 0 world position
            for (const auto& probe : g.rtao_capture.GetProbes())
               add("probe=%08X,%u,%.4f,%.4f,%.4f\n", probe.key, probe.flags, probe.pos[0], probe.pos[1], probe.pos[2]);
            g_rtdiag.pending_meta = meta;
            g_rtdiag.pending_set = set;
            g_rtdiag.pending_index = g_rtdiag.index;
            g_rtdiag.pending_frame = g_rtdiag.frame;
            g_rtdiag.index++;
         }
      }
      g_rtdiag.frame++;
   }

   static void DrawAOView(ID3D11Device* native_device, DeviceData& device_data, GameDeviceDataDishonored2& game_device_data)
   {
      const auto copy_vs = device_data.native_vertex_shaders.find(CompileTimeStringHash("Copy VS"));
      const auto view_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 AO View PS"));
      if (copy_vs == device_data.native_vertex_shaders.end() || !copy_vs->second || view_ps == device_data.native_pixel_shaders.end() || !view_ps->second)
         return;

      // The AO texture
      com_ptr<ID3D11ShaderResourceView> srv;
      float2 uv_scale = { 1.f, 1.f };
      // RT sun "Scene view" outputs are colors (RGB565 in the sun visibility): always shown, unpacked by the shader
      const bool scene_view = RTSunSceneView();
      const int view_mode = scene_view ? 3 : g_ao_view_mode;
      if (view_mode == 2)
      {
         if (!game_device_data.rtao.IsInitialized())
            return;
         ID3D11ShaderResourceView* rtao_srv = game_device_data.rtao.GetAOSRV();
         if (!rtao_srv)
            return;
         srv = rtao_srv;
         rtao_srv->Release();
      }
      else if (view_mode == 3)
      {
         ID3D11ShaderResourceView* sun_srv = game_device_data.rtsun_pass.GetSRV();
         if (!sun_srv)
            return;
         srv = sun_srv;
         sun_srv->Release();
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
      const float params[8] = { 1.f / float(back_buffer_desc.Width), 1.f / float(back_buffer_desc.Height), uv_scale.x, uv_scale.y, scrgb ? 203.f / 80.f : 1.f, scene_view ? 1.f : 0.f, 0.f, 0.f };
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
      const auto skin_cs = device_data.native_compute_shaders.find(CompileTimeStringHash("DS2 RTAO Skin CS"));
      ID3D11ComputeShader* skin_shader = skin_cs != device_data.native_compute_shaders.end() ? skin_cs->second.get() : nullptr;
      std::string error;
      auto& rtao = game_device_data.rtao;
      if (!game_device_data.rtao_capture.Init(native_device, rtao, capture_cs->second.get(), skin_shader, error) || !game_device_data.rtao_pass.Init(rtao.GetDevice(), error) ||
         !game_device_data.rtao_timer.Init(rtao.GetDevice(), rtao.GetTimestampFrequency(), error))
      {
         game_device_data.rtao_rt_failed = true;
         SetRTAOStatus("ray tracing failed to start, showing the depth test instead: " + error, true);
         return false;
      }
      game_device_data.rtao_rt_ready = true;
      // Texture exact cutouts: optional (without them cutouts stay stochastic)
      const auto strip_cs = device_data.native_compute_shaders.find(CompileTimeStringHash("DS2 RTAO Alpha Strip CS"));
      std::string cutout_error;
      if (strip_cs == device_data.native_compute_shaders.end() || !strip_cs->second)
         SetRTAOStatus("texture exact cutouts unavailable (alpha strip shader not compiled), cutouts stay stochastic", true);
      else if (!game_device_data.rtao_capture.InitCutouts(native_device, rtao, strip_cs->second.get(), cutout_error))
         SetRTAOStatus("texture exact cutouts unavailable (" + cutout_error + "), cutouts stay stochastic", true);
      if (game_device_data.rtao_capture.IsSkinningAvailable())
         SetRTAOStatus("ray tracing ready (geometry capture incl. skinned and dynamic meshes, BLAS/TLAS, RayQuery AO + denoiser)", false);
      else
         SetRTAOStatus("ray tracing ready, without skinned/dynamic meshes (" + (skin_shader ? game_device_data.rtao_capture.skinning_error : std::string("skin shader not compiled")) + ")", false);
      return true;
   }

   // Camera from the game's per view constants (the copy Luma keeps, see UpdateGlobalCB). The game's vertex shaders
   // compute clip = world.x * VP[0] + world.y * VP[1] + world.z * VP[2] + VP[3] on the matrix registers as stored,
   // which is the row vector convention RTAO::PassSettings uses, so the bytes are taken as they are.
   // The per view constants the SSAO draw uses: the last ones mapped on its recording thread (the game maps the per
   // view cbuffer on each deferred context before its passes). Luma's shared copy (cb_per_view_global) is whichever view
   // any thread mapped last, which can be another frame's (the camera a frame off: the traced shadows wobble when the
   // camera moves, and their TAA jitter differs from the depth's even when it doesn't).
   static const CBPerViewGlobals& GetRTAOPerView()
   {
      bool valid = t_rtao_per_view_valid;
      // Another camera (e.g. a spot light's perspective shadow view mapped last on this thread): not the SSAO's view
      if (valid)
      {
         const float dx = t_rtao_per_view.cb_wscampos.x - cb_per_view_global.cb_wscampos.x, dy = t_rtao_per_view.cb_wscampos.y - cb_per_view_global.cb_wscampos.y,
                     dz = t_rtao_per_view.cb_wscampos.z - cb_per_view_global.cb_wscampos.z;
         if (dx * dx + dy * dy + dz * dz > 4.f)
         {
            valid = false;
            g_rtao_view_other_camera++;
         }
      }
      const CBPerViewGlobals& per_view = valid ? t_rtao_per_view : cb_per_view_global;
      g_rtao_view_checks++;
      if (valid && t_rtao_per_view.cb_framenumber != cb_per_view_global.cb_framenumber)
         g_rtao_view_other_frame++; // The shared copy was another frame's
      if (!valid)
         g_rtao_view_no_thread_copy++;
      else if (std::memcmp(&t_rtao_per_view.cb_viewprojectionmatrix, &cb_per_view_global.cb_viewprojectionmatrix, sizeof(cb_per_view_global.cb_viewprojectionmatrix)) != 0)
      {
         g_rtao_view_mismatch++;
         const float* a = reinterpret_cast<const float*>(&t_rtao_per_view.cb_viewprojectionmatrix);
         const float* b = reinterpret_cast<const float*>(&cb_per_view_global.cb_viewprojectionmatrix);
         float diff = 0.f;
         for (int i = 0; i < 16; i++)
            diff = (std::max)(diff, std::fabs(a[i] - b[i]));
         const uint32_t diff_bits = uint32_t((std::min)(diff * 1e6f, 4e9f)); // Max element difference x 1e6
         uint32_t prev = g_rtao_view_mismatch_max.load();
         while (diff_bits > prev && !g_rtao_view_mismatch_max.compare_exchange_weak(prev, diff_bits)) {}
      }
      return per_view;
   }

   static RTAO::PassSettings MakeRTAOSettings(int output)
   {
      RTAO::PassSettings settings;
      const CBPerViewGlobals& per_view = GetRTAOPerView();
      static_assert(sizeof(settings.view_proj) == sizeof(per_view.cb_viewprojectionmatrix));
      std::memcpy(settings.view_proj, &per_view.cb_viewprojectionmatrix, sizeof(settings.view_proj));
      RTAO::InvertMatrix(settings.view_proj, settings.inv_view_proj);
      std::memcpy(settings.cam_pos, &per_view.cb_wscampos, sizeof(settings.cam_pos));
      std::memcpy(settings.cam_fwd, &per_view.cb_wscamforwarddir, sizeof(settings.cam_fwd));
      settings.reversed_z = true; // Dishonored 2 renders with inverted depth (see the DLSS settings)
      settings.radius = (std::max)(g_rtao_radius, 0.01f);
      const int rays = std::clamp(g_rtao_rays, 1, 64);
      settings.ray_count = uint32_t(g_rtao_quality == RTAO_QUALITY_MEDIUM ? (std::max)(1, (rays + 1) / 2) : rays);
      settings.filter_mode = (output == RTAO_OUTPUT_AO) ? RTAO::FilterMode::Temporal : RTAO::FilterMode::Off;
      // Half resolution needs the denoiser's upsample: the debug outputs stay at full resolution
      settings.half_res = g_rtao_quality == RTAO_QUALITY_LOW && output == RTAO_OUTPUT_AO;
      settings.debug_mode = output == RTAO_OUTPUT_COVERAGE ? 3 : 0;
      settings.strength = g_rtao_strength;
      settings.cutout_opacity = std::clamp(g_rtao_cutout_opacity, 0.f, 1.f);
      settings.cutout_mode = g_rtao_fast_cutouts ? 2 : 0;
      settings.cutout_exact = g_rtao_cutouts_exact;
      return settings;
   }

   static RTAO::Recorder MakeRTAORecorder(GameDeviceDataDishonored2& game_device_data, const RTAO::PassSettings& settings, const RTSun::SunSettings* sun = nullptr)
   {
      const bool has_sun = sun != nullptr;
      const RTSun::SunSettings sun_settings = sun ? *sun : RTSun::SunSettings{};
      return [&game_device_data, settings, has_sun, sun_settings](ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, std::string& error) {
         auto& g = game_device_data;
         g.rtao_retirer.Collect(info.completed_fence_value);
         g.rtao_timer.Collect(info.frame_slot);
         g.rtao_timer.Begin(list, info.frame_slot);
         // "Freeze ray tracing scene" (debug): no capture consumption, no TLAS rebuild, the last TLAS is traced again. If
         // the shadows still change while frozen, the cause is in the trace/denoiser/camera, not in the scene.
         const bool frozen = g_rtao_freeze_scene && g.rtao_last_top != 0;
         if (!frozen && !g.rtao_capture.Record(list, info, g.rtao_blases, g.rtao_tlas, g.rtao_retirer, uint32_t(std::clamp(g_rtao_blas_budget, 1, 4096)), error))
            return false;
         g.rtao_timer.Mark(list, info.frame_slot);
         const RTAO::BuildContext ctx = { info.device, list, &g.rtao_retirer, info.submit_fence_value };
         D3D12_GPU_VIRTUAL_ADDRESS top = g.rtao_last_top;
         uint32_t instances = g.rtao_tlas_instances;
         if (!frozen)
         {
            if (!g.rtao_tlas.Build(ctx, g.rtao_blases, info.frame_slot, top, instances, error))
               return false;
            g.rtao_last_top = top;
         }
         g.rtao_timer.Mark(list, info.frame_slot);
         if (!g.rtao_pass.Record(list, info, top, settings, g.rtao_retirer, error, &g.rtao_timer, &g.rtao_capture.GetCutoutBindings())) // Marks the end of the trace
            return false;
         g.rtao_timer.End(list, info.frame_slot);
         g.rtao_blas_count = g.rtao_blases.Count();
         g.rtao_blas_bytes = g.rtao_blases.ResultBytes();
         g.rtao_tlas_instances = instances;
         g.rtao_tlas_cutouts = g.rtao_tlas.LastCutoutCount();
         if (!frozen)
            g.rtao_tlas_missing_blas = g.rtao_tlas.LastMissingBlas();
         g.rtao_gpu_us = uint32_t(g.rtao_timer.LastMs() * 1000.0);
         g.rtao_gpu_capture_us = uint32_t(g.rtao_timer.LastSpanMs(0) * 1000.0);
         g.rtao_gpu_tlas_us = uint32_t(g.rtao_timer.LastSpanMs(1) * 1000.0);
         g.rtao_gpu_trace_us = uint32_t(g.rtao_timer.LastSpanMs(2) * 1000.0);
         g.rtao_gpu_filter_us = uint32_t(g.rtao_timer.LastSpanMs(3) * 1000.0);
         const RTAO::AOPass::CutoutCounters counters = g.rtao_pass.GetCutoutCounters();
         g.rtao_cutout_tests = counters.exact_tests;
         g.rtao_cutout_hits = counters.exact_hits;
         g.rtao_cutout_fallback_tests = counters.fallback_tests;
         if (has_sun)
            RecordRTSun(g, list, info, top, sun_settings); // Its own timer; a failure never fails the AO
         return true;
      };
   }

   // RT sun shadows, at the SSAO draw (render thread of that draw): starts the pass, sizes its output like the AO, and
   // fills the settings from the AO camera and the captured light direction. False: no sun trace this frame.
   static bool PrepareRTSun(ID3D11Device* native_device, GameDeviceDataDishonored2& game_device_data, const RTAO::PassSettings& ao_settings, uint32_t width, uint32_t height, RTSun::SunSettings& out)
   {
      if (g_rtsun_mode == RTSUN_MODE_OFF || game_device_data.rtsun_failed)
         return false;
      std::string error;
      auto& rtao = game_device_data.rtao;
      if (!game_device_data.rtsun_pass.IsInitialized())
      {
         if (!game_device_data.rtsun_pass.Init(rtao.GetDevice(), rtao.GetTimestampFrequency(), error))
         {
            game_device_data.rtsun_failed = true;
            SetRTSunStatus("failed to start, the game's shadows stay: " + error, true);
            return false;
         }
         SetRTSunStatus("ready, waiting for the light direction (screen shadows pass)", false);
      }
      out.debug_outputs = g_rtdiag.active;
      {
         const uint32_t copied = game_device_data.rtsun_shadow_copied_frame.load();
         out.shadow_repro = (g_rtdiag.active || game_device_data.rtao_frame % kReproLogFrames == 0) && copied != UINT32_MAX && game_device_data.rtao_frame - copied <= 2;
      }
      if (!game_device_data.rtsun_pass.EnsureTargets(native_device, width, height, error))
      {
         if (!game_device_data.rtsun_logged_error.exchange(true))
            SetRTSunStatus("targets failed: " + error, true);
         return false;
      }
      if (!game_device_data.rtsun_light.GetDirection(out.light_dir))
         return false;
      std::memcpy(out.view_proj, ao_settings.view_proj, sizeof(out.view_proj));
      std::memcpy(out.inv_view_proj, ao_settings.inv_view_proj, sizeof(out.inv_view_proj));
      std::memcpy(out.cam_pos, ao_settings.cam_pos, sizeof(out.cam_pos));
      std::memcpy(out.cam_fwd, ao_settings.cam_fwd, sizeof(out.cam_fwd));
      out.reversed_z = ao_settings.reversed_z;
      out.bias = ao_settings.bias;
      out.cutout_opacity = ao_settings.cutout_opacity;
      out.cutout_mode = ao_settings.cutout_mode; // Fast cutouts
      out.cutout_exact = ao_settings.cutout_exact; // Texture exact masks (the sun always traces at full resolution, whatever RTAO Quality)
      out.angular_diameter_deg = std::clamp(g_rtsun_angle, 0.f, 10.f);
      out.ray_count = uint32_t(std::clamp(g_rtsun_rays, 1, 16));
      out.casters_only = g_rtsun_casters_only;
      {
         // The copy from the last frame's screen shadows draw (this frame's comes after the trace)
         const uint32_t copied = game_device_data.rtsun_shadow_copied_frame.load();
         out.shadow_consult = g_rtsun_shadow_consult && copied != UINT32_MAX && game_device_data.rtao_frame - copied <= 2;
         out.shadow_reversed = game_device_data.rtsun_shadow_reversed.load();
         out.shadow_tolerance = g_rtsun_shadow_tolerance;
         // The game's rasterizer bias in depth units: DepthBias * r (r: the depth format's unit; a 32 bit float map's is
         // ~2^-24 at the far end, negligible) and SlopeScaledDepthBias
         const DXGI_FORMAT depth_format = DXGI_FORMAT(g_shadow_depth_format.load());
         const double r = depth_format == DXGI_FORMAT_D16_UNORM ? 1.0 / 65535.0 : (depth_format == DXGI_FORMAT_D24_UNORM_S8_UINT ? 1.0 / 16777215.0 : 1.0 / 16777216.0);
         out.shadow_bias_constant = float(double(g_shadow_depth_bias.load()) * r);
         out.shadow_bias_slope = g_shadow_slope_bias.load();
      }
      const int output = std::clamp(g_rtsun_output, 0, int(RTSun::Output::Count) - 1);
      out.output = RTSun::Output(output);
      out.reset_history = output != game_device_data.rtsun_last_output || g_rtdiag.reset_history.exchange(false);
      game_device_data.rtsun_last_output = output;
      return true;
   }

   // Present: reads back the light direction copies, logs it when it changes (more than ~2 degrees)
   static void PollRTSunLight(ID3D11Device* native_device, GameDeviceDataDishonored2& game_device_data)
   {
      com_ptr<ID3D11DeviceContext> immediate;
      native_device->GetImmediateContext(&immediate);
      game_device_data.rtsun_light.Poll(immediate.get(), game_device_data.rtao_frame);
      float dir[3], sun[3];
      if (!game_device_data.rtsun_light.GetDirection(dir, nullptr, sun))
         return;
      float* logged = game_device_data.rtsun_logged_dir;
      if (dir[0] * logged[0] + dir[1] * logged[1] + dir[2] * logged[2] > 0.9994f)
         return;
      std::memcpy(logged, dir, sizeof(game_device_data.rtsun_logged_dir));
      char buffer[256];
      snprintf(buffer, sizeof(buffer), "light direction (toward the light, from the shadow map axis) %.3f %.3f %.3f, atmosphere sun %.3f %.3f %.3f (angle between %.1f deg)",
         dir[0], dir[1], dir[2], sun[0], sun[1], sun[2], std::acos(std::clamp(dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2], -1.f, 1.f)) * 57.29578f);
      reshade::log::message(reshade::log::level::info, (std::string("RT sun: ") + buffer).c_str());
   }

   // Inside the RTAO recorder, after the AO (same TLAS)
   static void RecordRTSun(GameDeviceDataDishonored2& g, ID3D12GraphicsCommandList4* list, const RTAO::RecordInfo& info, D3D12_GPU_VIRTUAL_ADDRESS top, const RTSun::SunSettings& sun_settings)
   {
      std::string error;
      if (g.rtsun_pass.Record(list, info, top, sun_settings, g.rtao_retirer, error, &g.rtao_capture.GetCutoutBindings()))
      {
         g.rtsun_gpu_us = uint32_t(g.rtsun_pass.LastGpuMs() * 1000.0);
         if (g.rtsun_pass.TraceCount() == 1)
            SetRTSunStatus("tracing", false);
         return;
      }
      g.rtsun_record_failures++;
      if (!g.rtsun_logged_error.exchange(true))
         SetRTSunStatus("trace failed (the visibility is not updated): " + error, true);
   }

   // The game's screen shadows draw (replaced pixel shaders ScreenShadows_*.hlsl): captures the light direction, and
   // binds this frame's traced visibility at t3 for the draw. Returns true if it drew (state restored after).
   // Shadow map check: copies the game's sun cascades (PS t0 of the screen shadows draw) and their light matrices (its PS
   // b2, still bound) into the texture shared with the RT sun trace (Shaders/Dishonored 2/Luma_RTSunShadowCopy.hlsl)
   static void RTSunCopyShadowMap(ID3D11Device* native_device, ID3D11DeviceContext* context, DeviceData& device_data, GameDeviceDataDishonored2& g)
   {
      if (!g.rtsun_pass.IsInitialized())
         return;
      const auto copy_vs = device_data.native_vertex_shaders.find(CompileTimeStringHash("Copy VS"));
      const auto copy_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 RTSun Shadow Copy PS"));
      const auto pack_ps = device_data.native_pixel_shaders.find(CompileTimeStringHash("DS2 RTSun Shadow Pack PS"));
      if (copy_vs == device_data.native_vertex_shaders.end() || !copy_vs->second || copy_ps == device_data.native_pixel_shaders.end() || !copy_ps->second ||
         pack_ps == device_data.native_pixel_shaders.end() || !pack_ps->second)
         return;
      com_ptr<ID3D11ShaderResourceView> shadow_srv;
      context->PSGetShaderResources(0, 1, &shadow_srv);
      if (!shadow_srv)
         return;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
      shadow_srv->GetDesc(&srv_desc);
      com_ptr<ID3D11Resource> resource;
      shadow_srv->GetResource(&resource);
      com_ptr<ID3D11Texture2D> texture;
      if (srv_desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2DARRAY || !resource || FAILED(resource->QueryInterface(&texture)))
         return;
      D3D11_TEXTURE2D_DESC desc = {};
      texture->GetDesc(&desc);
      std::string error;
      if (!g.rtsun_pass.EnsureShadowTarget(native_device, desc.Width, desc.Height, error))
      {
         if (!g.rtsun_shadow_logged_error.exchange(true))
            reshade::log::message(reshade::log::level::warning, ("[Luma] RT sun: shadow map check unavailable: " + error).c_str());
         return;
      }
      bool packed = false;
      ID3D11RenderTargetView* rtv = g.rtsun_pass.GetShadowRTV(&packed);
      if (!rtv)
         return;
      // Which way the game compares (its PCF sampler, s0): GREATER means nearer the light is larger
      com_ptr<ID3D11SamplerState> sampler;
      context->PSGetSamplers(0, 1, &sampler);
      if (sampler)
      {
         D3D11_SAMPLER_DESC sampler_desc = {};
         sampler->GetDesc(&sampler_desc);
         g.rtsun_shadow_reversed = sampler_desc.ComparisonFunc == D3D11_COMPARISON_GREATER || sampler_desc.ComparisonFunc == D3D11_COMPARISON_GREATER_EQUAL;
      }
      DrawStateStack<DrawStateStackType::FullGraphics> draw_state_stack;
      draw_state_stack.Cache(context, device_data.uav_max_count);
      DrawCustomPixelShader(context, device_data.default_depth_stencil_state.get(), device_data.default_blend_state.get(), nullptr, copy_vs->second.get(),
         packed ? pack_ps->second.get() : copy_ps->second.get(), shadow_srv.get(), rtv, desc.Width, desc.Height * RTSun::kShadowCascades + 1);
      draw_state_stack.Restore(context);
      rtv->Release();
      g.rtsun_shadow_copied_frame = g.rtao_frame;
      g.rtsun_shadow_copies++;
   }

   static bool DrawRTSunScreenShadows(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, GameDeviceDataDishonored2& game_device_data, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      if (g_rtsun_mode == RTSUN_MODE_OFF || GetEffectiveAOMode() != AO_MODE_RTAO)
         return false;
      std::string error;
      if (!original_draw_dispatch_func || !*original_draw_dispatch_func || !game_device_data.rtsun_light.Init(native_device, error))
         return false;
      const uint32_t frame = game_device_data.rtao_frame;
      game_device_data.rtsun_light.OnScreenShadowsDraw(native_device_context, frame);
      // The copy: for the shadow map check, captures, and the periodic reproduction (the frame before it)
      if (g_rtsun_shadow_consult || g_rtdiag.active || (game_device_data.rtao_frame + 2) % kReproLogFrames <= 1)
         RTSunCopyShadowMap(native_device, native_device_context, device_data, game_device_data);
      // Traced by this frame's SSAO draw (or the previous frame's, if the recording threads raced): otherwise the game's
      if (frame - game_device_data.rtsun_scheduled_frame.load() > 1 || game_device_data.rtsun_pass.TraceCount() == 0)
      {
         game_device_data.rtsun_stale_draws++;
         return false;
      }
      ID3D11ShaderResourceView* visibility = game_device_data.rtsun_pass.GetSRV();
      if (!visibility)
         return false;
      com_ptr<ID3D11ShaderResourceView> previous;
      native_device_context->PSGetShaderResources(3, 1, &previous);
      native_device_context->PSSetShaderResources(3, 1, &visibility);
      // The core doesn't set Luma's cbuffers for draws the game code replaces (RTSunMode is in LumaSettings)
      SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaSettings);
      updated_cbuffers = true;
      (*original_draw_dispatch_func)();
      if (g_rtdiag.active)
         RTDiagDrawGameShadow(native_device, native_device_context, game_device_data, original_draw_dispatch_func);
      ID3D11ShaderResourceView* const restore = previous.get();
      native_device_context->PSSetShaderResources(3, 1, &restore);
      visibility->Release();
      game_device_data.rtsun_bound_draws++;
      return true;
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
            const bool ok = RTAOOutputIsRayTraced(pending.output) ? game_device_data.rtao.Run(context.get(), MakeRTAORecorder(game_device_data, pending.settings, pending.sun ? &pending.sun_settings : nullptr), error) : game_device_data.rtao.Run(context.get(), pending.constants, error);
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
      if (g_rtsun_mode != RTSUN_MODE_OFF)
         PollRTSunLight(native_device, game_device_data);
      if (game_device_data.rtao_capture.IsInitialized() && g_rtao_capture_active.load(std::memory_order_relaxed))
      {
         com_ptr<ID3D11DeviceContext> immediate;
         native_device->GetImmediateContext(&immediate);
         game_device_data.rtao_capture.PollCounters(immediate.get());
      }
      if (game_device_data.rtao.IsInitialized())
      {
         static bool removal_logged = false;
         if (!removal_logged)
         {
            const std::string report = game_device_data.rtao.DeviceRemovedReport();
            if (!report.empty())
            {
               removal_logged = true;
               reshade::log::message(reshade::log::level::error, ("RTAO: " + report).c_str());
            }
         }
         RTDiagOnPresent(native_device, device_data, game_device_data);
      }
      if (g_ao_view_mode != 0 || RTSunSceneView())
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
      // Shadow map draws per frame, and every 600 frames a line with the scene's retention and the camera source
      // (popping and wobble diagnostics, see RTAO_NOTES.md "Shadow map casters")
      if (game_device_data.rtao.IsInitialized() && g_rtao_capture_active.load(std::memory_order_relaxed))
      {
         const uint64_t shadow_draws = g_rtao_shadow_view_draws.load();
         const uint64_t delta = shadow_draws - g_rtao_shadow_draws_at_present;
         g_rtao_shadow_draws_at_present = shadow_draws;
         g_rtao_shadow_frames++;
         g_rtao_shadow_zero_frames += delta == 0;
         g_rtao_shadow_max_per_frame = (std::max)(g_rtao_shadow_max_per_frame, delta);
         // Shadow map reproduction score (cascades 0 and 1, every instance / shadow map instances only), each new result
         if (!g_rtdiag.active)
         {
            std::vector<uint32_t> repro;
            uint32_t repro_trace = 0;
            if (game_device_data.rtsun_pass.GetShadowRepro(repro, repro_trace) && repro_trace != g_repro_logged_trace)
            {
               g_repro_logged_trace = repro_trace;
               const size_t per = size_t(RTSun::kReproGrid) * RTSun::kReproGrid;
               double all[RTSun::kReproCascades] = {}, only[RTSun::kReproCascades] = {};
               uint64_t missing0 = 0, extra0 = 0;
               for (size_t i = 0; i < repro.size(); i++)
               {
                  const size_t c = i / per;
                  all[c] += ((repro[i] >> 2) & 3) == 0;
                  only[c] += (repro[i] & 3) == 0;
                  missing0 += c == 0 && ((repro[i] >> 2) & 3) == 1;
                  extra0 += c == 0 && ((repro[i] >> 2) & 3) == 2;
               }
               char text[260];
               snprintf(text, sizeof(text), "RTAO diag: shadow map reproduction (pass %u): cascade 0 %.2f%% match (missing %llu, extra %llu texels), cascade 1 %.2f%%; shadow map instances only: %.2f%% / %.2f%%",
                  repro_trace, 100.0 * all[0] / per, (unsigned long long)missing0, (unsigned long long)extra0, 100.0 * all[1] / per, 100.0 * only[0] / per, 100.0 * only[1] / per);
               reshade::log::message(reshade::log::level::info, text);
            }
         }
         // Every shadow map redraw burst (first 200): when the game puts static content into its shadow maps
         if (game_device_data.rtao_capture.IsInitialized())
         {
            static uint32_t logged_bursts = 0;
            const RTAO::RetentionStats burst = game_device_data.rtao_tlas.GetRetentionStats();
            if (burst.bursts != logged_bursts && logged_bursts < 200)
            {
               logged_bursts = burst.bursts;
               char text[200];
               snprintf(text, sizeof(text), "RTAO diag: shadow map redraw burst %u at TLAS frame %u (Luma frame %u): %u new shadow map instances, %u static kept, %u dynamic meshes",
                  burst.bursts, burst.last_burst_frame, game_device_data.rtao_frame, burst.last_burst_size, burst.shadow_static, burst.dynamic_meshes);
               reshade::log::message(reshade::log::level::info, text);
            }
         }
         if (g_rtao_shadow_frames >= 600 && game_device_data.rtao_capture.IsInitialized())
         {
            const RTAO::CaptureStats stats = game_device_data.rtao_capture.GetStats();
            const RTAO::RetentionStats& r = stats.retention;
            char line[1800];
            snprintf(line, sizeof(line),
               "RTAO diag: shadow map draws in %u frames: %u frames with none, max %llu in a frame | TLAS retained %u (promoted %u, young %u; main view only %u, shadow maps only %u, both %u), "
               "last readback: new %u, evicted %u promoted + %u young, shadow young %u | BLAS: meshes %llu, built %llu, failed %llu, TLAS instances %u, left out (BLAS not built yet) %u, budget %d/frame, arena %llu MB, draws refused: arena full %llu, buffer %llu, ring %llu | MVP instances %llu, rekeyed %llu (fuzzy entries %llu) | "
               "camera: SSAO draws %llu, shared copy from another frame %llu, view projection differed %llu (max element diff %.6f), no thread copy %llu, thread copy another camera %llu | game writes into captured buffers %llu (%llu buffers, %llu re-copied) | indirect draws (not captured): main view %llu, shadow maps %llu | matrix tag mismatches %llu of %llu | movable objects kept while not drawn %llu (dropped: moved %llu, freed %llu, cap %llu) | per frame meshes (GPU, total): shadow maps %llu (matrix rejected %llu), main view %llu (rejected %llu)",
               g_rtao_shadow_frames, g_rtao_shadow_zero_frames, (unsigned long long)g_rtao_shadow_max_per_frame, r.promoted + r.young, r.promoted, r.young, r.main_only, r.shadow_only, r.both,
               r.added_new, r.evicted_promoted, r.evicted_young, r.shadow_young, (unsigned long long)stats.meshes, (unsigned long long)stats.meshes_ready, (unsigned long long)stats.blas_failed,
               game_device_data.rtao_tlas_instances.load(), game_device_data.rtao_tlas_missing_blas.load(), g_rtao_blas_budget, (unsigned long long)(stats.arena_vertex_bytes >> 20),
               (unsigned long long)stats.skipped[RTAO::kSkipArenaFull], (unsigned long long)stats.skipped[RTAO::kSkipBuffer], (unsigned long long)stats.skipped[RTAO::kSkipRingFull], (unsigned long long)stats.mvp_instances, (unsigned long long)stats.mvp_rekeyed, (unsigned long long)stats.fuzzy_entries,
               (unsigned long long)g_rtao_view_checks.load(), (unsigned long long)g_rtao_view_other_frame.load(), (unsigned long long)g_rtao_view_mismatch.load(), g_rtao_view_mismatch_max.load() / 1e6,
               (unsigned long long)g_rtao_view_no_thread_copy.load(), (unsigned long long)g_rtao_view_other_camera.load(),
               (unsigned long long)stats.written_after_copy, (unsigned long long)stats.written_buffers, (unsigned long long)stats.written_forgotten,
               (unsigned long long)g_rtao_indirect_draws.load(), (unsigned long long)g_rtao_shadow_indirect_draws.load(), (unsigned long long)stats.tag_mismatch, (unsigned long long)stats.tag_checked, (unsigned long long)stats.frame_retained, (unsigned long long)stats.hold_superseded, (unsigned long long)stats.hold_owner_freed, (unsigned long long)stats.hold_capped,
               (unsigned long long)stats.gpu_shadow_frame, (unsigned long long)stats.gpu_shadow_frame_rejected, (unsigned long long)stats.gpu_main_frame, (unsigned long long)stats.gpu_main_frame_rejected);
            reshade::log::message(reshade::log::level::info, line);
            {
               // Shadow map layout over this window: draws per slice and viewport (x,y wxh), then reset
               std::string layout = "RTAO diag: shadow map layout (slice: viewport x,y wxh = draws):";
               const std::lock_guard lock(g_rtao_shadow_layout_mutex);
               for (const auto& [k, n] : g_rtao_shadow_layout)
                  layout += " s" + std::to_string(std::get<0>(k)) + ": " + std::to_string(std::get<1>(k)) + "," + std::to_string(std::get<2>(k)) + " " + std::to_string(std::get<3>(k)) + "x" + std::to_string(std::get<4>(k)) + " = " + std::to_string(n) + ";";
               layout += " near only " + std::to_string(int(g_rtao_shadow_near_only)) + " (" + std::to_string(g_rtao_shadow_near_slices) + " slices), far skipped total " + std::to_string(g_rtao_shadow_far_skipped.load());
               g_rtao_shadow_layout.clear();
               reshade::log::message(reshade::log::level::info, layout.c_str());
            }
            g_rtao_shadow_frames = g_rtao_shadow_zero_frames = 0;
            g_rtao_shadow_max_per_frame = 0;
            g_rtao_view_checks = g_rtao_view_no_thread_copy = g_rtao_view_mismatch = g_rtao_view_other_frame = g_rtao_view_other_camera = 0;
            g_rtao_view_mismatch_max = 0;
         }
      }
      if (game_device_data.rtao.IsInitialized() && (rtao_frames == 300 || rtao_frames == 3000 || rtao_frames == 30000))
      {
         char buffer[512];
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
            snprintf(buffer, sizeof(buffer), "RTAO: per frame meshes: skinned draws %llu main view / %llu captured (instanced %llu, copy pending %llu), dynamic %llu captured (instanced %llu, too large %llu), last frame %llu draws %llu vertices, over budget %llu, BLAS builds %llu refits %llu failed %llu",
               (unsigned long long)g_rtao_draws_skinned.load(), (unsigned long long)stats.skinned_captured, (unsigned long long)stats.skinned_instanced, (unsigned long long)stats.skinned_waiting,
               (unsigned long long)stats.dynamic_captured, (unsigned long long)stats.dynamic_instanced, (unsigned long long)stats.dynamic_too_large, (unsigned long long)stats.frame_draws,
               (unsigned long long)stats.frame_vertices, (unsigned long long)stats.frame_overflow, (unsigned long long)stats.frame_blas_builds, (unsigned long long)stats.frame_blas_refits, (unsigned long long)stats.frame_blas_failed);
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
            char foliage[768];
            snprintf(foliage, sizeof(foliage), "RTAO: foliage: wind draws captured %llu (%llu instances), alpha tested draws %llu (captured %llu, %llu instances), cutouts in the TLAS %u, pixel shaders alpha tested %llu of %llu with discard (%llu reflected)",
               (unsigned long long)stats.wind_draws, (unsigned long long)stats.wind_instances, (unsigned long long)g_rtao_draws_alpha_tested.load(), (unsigned long long)stats.cutout_draws,
               (unsigned long long)stats.cutout_instances, game_device_data.rtao_tlas_cutouts.load(), (unsigned long long)g_rtao_ps_alpha_tested.load(), (unsigned long long)g_rtao_ps_discard.load(),
               (unsigned long long)g_rtao_ps_reflected.load());
            reshade::log::message(reshade::log::level::info, foliage);
            snprintf(foliage, sizeof(foliage), "RTAO: shadow map casters %s: shadow map draws %llu (last map %ux%u, %u slices), captured %llu (%llu instances, %llu per frame meshes), skipped (characters) %llu, MVP instances rejected %llu, readback overflow %llu",
               g_rtao_shadow_casters ? "on" : "off", (unsigned long long)g_rtao_shadow_view_draws.load(), g_rtao_shadow_view_width.load(), g_rtao_shadow_view_height.load(), g_rtao_shadow_view_slices.load(),
               (unsigned long long)stats.shadow_draws, (unsigned long long)stats.shadow_instances, (unsigned long long)stats.shadow_frame_meshes, (unsigned long long)stats.shadow_skipped,
               (unsigned long long)stats.shadow_rejected, (unsigned long long)stats.readback_overflow);
            reshade::log::message(reshade::log::level::info, foliage);
            static const char* quality_names[RTAO_QUALITY_COUNT] = { "low", "medium", "high" };
            snprintf(buffer, sizeof(buffer), "RTAO: GPU last frame %.2f ms: capture %.2f, TLAS %.2f, trace %.2f, denoise %.2f (quality %s, %d rays, fast cutouts %d); per frame duplicates skipped: %llu draws, %llu per frame meshes",
               game_device_data.rtao_gpu_us.load() / 1000.0, game_device_data.rtao_gpu_capture_us.load() / 1000.0, game_device_data.rtao_gpu_tlas_us.load() / 1000.0,
               game_device_data.rtao_gpu_trace_us.load() / 1000.0, game_device_data.rtao_gpu_filter_us.load() / 1000.0, quality_names[std::clamp(g_rtao_quality, 0, int(RTAO_QUALITY_COUNT) - 1)], g_rtao_rays,
               int(g_rtao_fast_cutouts), (unsigned long long)stats.dedup_draws, (unsigned long long)stats.dedup_frame_meshes);
            reshade::log::message(reshade::log::level::info, buffer);
            const RTAO::CutoutStats cutout = game_device_data.rtao_capture.GetCutoutStats();
            snprintf(foliage, sizeof(foliage), "RTAO: texture exact cutouts %s: draws with a mask %llu / stochastic %llu, materials %llu (masked %llu, pending %llu), atlas slices %llu/%u (overflow %llu), records %llu (overflow %llu), strips %llu rendered %llu copied %llu deferred %llu abandoned, last frame candidates %llu masked (%llu solid) %llu stochastic, pixel shaders replayable %llu",
               g_rtao_cutouts_exact ? (game_device_data.rtao_capture.IsCutoutsAvailable() ? "on" : "unavailable") : "off", (unsigned long long)cutout.draws_exact, (unsigned long long)cutout.draws_no_material,
               (unsigned long long)cutout.materials, (unsigned long long)cutout.materials_ready, (unsigned long long)cutout.materials_pending, (unsigned long long)cutout.atlas_slices_used, RTAO::CutoutMaterials::kAtlasSlices,
               (unsigned long long)cutout.atlas_overflow, (unsigned long long)cutout.records, (unsigned long long)cutout.record_overflow, (unsigned long long)cutout.strips_rendered, (unsigned long long)cutout.strips_copied,
               (unsigned long long)cutout.strips_deferred, (unsigned long long)cutout.strips_abandoned, (unsigned long long)game_device_data.rtao_cutout_tests.load(), (unsigned long long)game_device_data.rtao_cutout_hits.load(),
               (unsigned long long)game_device_data.rtao_cutout_fallback_tests.load(), (unsigned long long)g_rtao_ps_texture_exact.load());
            reshade::log::message(reshade::log::level::info, foliage);
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

      if (original_shader_hashes.Contains(shader_hashes_ScreenShadows))
      {
         if (DrawRTSunScreenShadows(native_device, native_device_context, cmd_list_data, device_data, game_device_data, updated_cbuffers, original_draw_dispatch_func))
            return DrawOrDispatchOverrideType::Replaced;
         return DrawOrDispatchOverrideType::None;
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
            const uint32_t hash = Shader::BinToHash(static_cast<const uint8_t*>(desc->code), desc->code_size);
            if (info.skip_reason && strcmp(info.skip_reason, "no VERTEX_POSITION") == 0)
            {
               std::string signature;
               RTAO::VertexShaderInfo unused;
               RTAO::ReflectVertexShader(desc->code, desc->code_size, unused, &signature);
               char buffer[512];
               snprintf(buffer, sizeof(buffer), "RTAO: vertex shader 0x%08X has no VERTEX_POSITION input (inputs: %s)", hash, signature.empty() ? "none" : signature.c_str());
               reshade::log::message(reshade::log::level::info, buffer);
            }
            if (info.skinned)
               g_rtao_vs_skinned++;
            const std::unique_lock lock(g_rtao_shader_mutex);
            g_rtao_vertex_shaders[pipeline.handle] = info;
            g_rtao_vertex_shader_hashes[pipeline.handle] = hash;
         }
         else if (subobject.type == reshade::api::pipeline_subobject_type::pixel_shader && subobject.data)
         {
            // Alpha tested pixel shaders make their draws cutouts (RTAO_NOTES.md "Alpha tested geometry")
            const auto* desc = static_cast<const reshade::api::shader_desc*>(subobject.data);
            if (!desc->code || desc->code_size == 0)
               continue;
            {
               const std::unique_lock lock(g_rtao_shader_mutex);
               g_rtao_pixel_shader_hashes[pipeline.handle] = Shader::BinToHash(static_cast<const uint8_t*>(desc->code), desc->code_size);
            }
            RTAO::PixelShaderInfo info;
            RTAO::ReflectPixelShader(desc->code, desc->code_size, info);
            g_rtao_ps_reflected++;
            g_rtao_ps_discard += info.discard;
            g_rtao_ps_alpha_tested += info.alpha_tested;
            g_rtao_ps_texture_exact += info.texture_exact;
            if (info.alpha_tested)
            {
               const std::unique_lock lock(g_rtao_shader_mutex);
               g_rtao_alpha_tested_ps[pipeline.handle] = info;
            }
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
      g_rtao_vertex_shader_hashes.erase(pipeline.handle);
      g_rtao_pixel_shader_hashes.erase(pipeline.handle);
      g_rtao_input_layouts.erase(pipeline.handle);
      g_rtao_alpha_tested_ps.erase(pipeline.handle);
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

   // A shadow map draw: depth only (no render target) with depth writes on, into a depth texture array (the sun's
   // cascades, the local lights' atlas) that isn't the main view's depth. Whatever is drawn there is world geometry.
   static bool IsShadowViewDepthWrite(ID3D11DeviceContext* context, uint32_t* out_slice = nullptr)
   {
      com_ptr<ID3D11RenderTargetView> rtv;
      com_ptr<ID3D11DepthStencilView> dsv;
      context->OMGetRenderTargets(1, &rtv, &dsv);
      if (!dsv || rtv)
         return false;
      com_ptr<ID3D11Resource> resource;
      dsv->GetResource(&resource);
      com_ptr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource->QueryInterface(&texture)))
         return false;
      D3D11_TEXTURE2D_DESC desc;
      texture->GetDesc(&desc);
      const bool main_size = desc.Width == g_rtao_main_depth_width.load(std::memory_order_relaxed) && desc.Height == g_rtao_main_depth_height.load(std::memory_order_relaxed);
      const bool flat = desc.ArraySize == 1 && g_rtao_shadow_flat && desc.Width >= 1024 && desc.Width == desc.Height;
      if ((desc.ArraySize < 2 && !flat) || main_size)
         return false;
      com_ptr<ID3D11DepthStencilState> state;
      UINT stencil_ref = 0;
      context->OMGetDepthStencilState(&state, &stencil_ref);
      if (state)
      {
         D3D11_DEPTH_STENCIL_DESC state_desc;
         state->GetDesc(&state_desc);
         if (!state_desc.DepthEnable || state_desc.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ALL)
            return false;
      }
      D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc;
      dsv->GetDesc(&dsv_desc);
      const uint32_t slice = flat ? kFlatShadowSlice : (dsv_desc.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DARRAY ? dsv_desc.Texture2DArray.FirstArraySlice : 0);
      {
         D3D11_VIEWPORT viewport = {};
         UINT viewports = 1;
         context->RSGetViewports(&viewports, &viewport);
         const std::lock_guard lock(g_rtao_shadow_layout_mutex);
         if (g_rtao_shadow_layout.size() < 64)
            g_rtao_shadow_layout[{ slice, int(viewport.TopLeftX), int(viewport.TopLeftY), int(viewport.Width), int(viewport.Height) }]++;
         else
         {
            auto it = g_rtao_shadow_layout.find({ slice, int(viewport.TopLeftX), int(viewport.TopLeftY), int(viewport.Width), int(viewport.Height) });
            if (it != g_rtao_shadow_layout.end())
               it->second++;
         }
      }
      if (out_slice)
         *out_slice = slice;
      {
         com_ptr<ID3D11RasterizerState> raster;
         context->RSGetState(&raster);
         if (raster)
         {
            D3D11_RASTERIZER_DESC raster_desc;
            raster->GetDesc(&raster_desc);
            g_shadow_depth_bias.store(raster_desc.DepthBias, std::memory_order_relaxed);
            g_shadow_slope_bias.store(raster_desc.SlopeScaledDepthBias, std::memory_order_relaxed);
         }
         g_shadow_depth_format.store(uint32_t(dsv_desc.Format), std::memory_order_relaxed);
      }
      g_rtao_shadow_view_width.store(desc.Width, std::memory_order_relaxed);
      g_rtao_shadow_view_height.store(desc.Height, std::memory_order_relaxed);
      g_rtao_shadow_view_slices.store(desc.ArraySize, std::memory_order_relaxed);
      return true;
   }

   // Logs an uncaptured main view draw once per (vertex shader, reason), so the log names what is missing
   static void LogRTAOUncaptured(ID3D11DeviceContext* context, const char* reason, const char* detail = nullptr)
   {
      com_ptr<ID3D11VertexShader> vs;
      context->VSGetShader(&vs, nullptr, nullptr);
      uint32_t hash = 0;
      {
         const std::shared_lock lock(g_rtao_shader_mutex);
         auto it = g_rtao_vertex_shader_hashes.find(uint64_t(uintptr_t(vs.get())));
         if (it != g_rtao_vertex_shader_hashes.end())
            hash = it->second;
      }
      const uint64_t key = (uint64_t(hash) << 32) ^ uint64_t(uintptr_t(reason));
      {
         const std::lock_guard lock(g_rtao_diag_mutex);
         if (g_rtao_diag_logged.size() >= 200 || !g_rtao_diag_logged.insert(key).second)
            return;
      }
      com_ptr<ID3D11PixelShader> ps;
      context->PSGetShader(&ps, nullptr, nullptr);
      char buffer[256];
      snprintf(buffer, sizeof(buffer), "RTAO: not captured: VS 0x%08X (PS object %p): %s%s%s", hash, (void*)ps.get(), reason, detail ? " " : "", detail ? detail : "");
      reshade::log::message(reshade::log::level::info, buffer);
   }

   static void ShadowAudit(ID3D11DeviceContext* context, uint32_t slice, const char* outcome, uint32_t indices)
   {
      com_ptr<ID3D11VertexShader> vs;
      context->VSGetShader(&vs, nullptr, nullptr);
      com_ptr<ID3D11PixelShader> ps;
      context->PSGetShader(&ps, nullptr, nullptr);
      uint32_t vs_hash = 0, ps_hash = 0;
      {
         const std::shared_lock lock(g_rtao_shader_mutex);
         auto vh = g_rtao_vertex_shader_hashes.find(uint64_t(uintptr_t(vs.get())));
         vs_hash = vh != g_rtao_vertex_shader_hashes.end() ? vh->second : 0;
         auto ph = ps ? g_rtao_pixel_shader_hashes.find(uint64_t(uintptr_t(ps.get()))) : g_rtao_pixel_shader_hashes.end();
         ps_hash = ph != g_rtao_pixel_shader_hashes.end() ? ph->second : 0;
      }
      const std::lock_guard lock(g_shadow_audit_mutex);
      ShadowAuditKey key = { slice, outcome, vs_hash, ps_hash, indices };
      auto it = g_shadow_audit.find(key);
      if (it != g_shadow_audit.end())
         it->second++;
      else if (g_shadow_audit.size() < 50000)
         g_shadow_audit.emplace(std::move(key), 1);
   }

   static void DrawProbe(ID3D11DeviceContext* context, uint32_t index_count, uint32_t first_index, int32_t vertex_offset)
   {
      com_ptr<ID3D11InputLayout> layout;
      context->IAGetInputLayout(&layout);
      com_ptr<ID3D11VertexShader> vs;
      context->VSGetShader(&vs, nullptr, nullptr);
      com_ptr<ID3D11PixelShader> ps;
      context->PSGetShader(&ps, nullptr, nullptr);
      RTAO::InputLayoutInfo layout_info;
      uint32_t vs_hash = 0, ps_hash = 0;
      {
         const std::shared_lock lock(g_rtao_shader_mutex);
         auto layout_it = g_rtao_input_layouts.find(uint64_t(uintptr_t(layout.get())));
         if (layout_it != g_rtao_input_layouts.end())
            layout_info = layout_it->second;
         auto vh = g_rtao_vertex_shader_hashes.find(uint64_t(uintptr_t(vs.get())));
         vs_hash = vh != g_rtao_vertex_shader_hashes.end() ? vh->second : 0;
         auto ph = ps ? g_rtao_pixel_shader_hashes.find(uint64_t(uintptr_t(ps.get()))) : g_rtao_pixel_shader_hashes.end();
         ps_hash = ph != g_rtao_pixel_shader_hashes.end() ? ph->second : 0;
      }
      com_ptr<ID3D11Buffer> vb, ib;
      UINT stride = 0, offset = 0, index_offset = 0;
      DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
      context->IAGetVertexBuffers(layout_info.position_slot, 1, &vb, &stride, &offset);
      context->IAGetIndexBuffer(&ib, &index_format, &index_offset);
      uint64_t key = 0;
      const uint32_t index_size = index_format == DXGI_FORMAT_R16_UINT ? 2 : (index_format == DXGI_FORMAT_R32_UINT ? 4 : 0);
      bool dynamic = false;
      if (vb && ib && layout_info.has_position && index_size)
      {
         D3D11_BUFFER_DESC vb_desc = {};
         vb->GetDesc(&vb_desc);
         dynamic = vb_desc.Usage == D3D11_USAGE_DYNAMIC;
         const int64_t vertex_zero = int64_t(offset) + int64_t(vertex_offset) * int64_t(stride);
         if (vertex_zero >= 0)
            key = RTAO::ComputeMeshKey(vb.get(), ib.get(), uint64_t(vertex_zero) + layout_info.position_offset, stride,
               dynamic ? layout_info.position_format : RTAO::DxrVertexFormat(layout_info.position_format), uint64_t(index_offset) + uint64_t(first_index) * index_size, index_count, index_format);
      }
      com_ptr<ID3D11BlendState> blend;
      FLOAT factor[4];
      UINT sample_mask = 0;
      context->OMGetBlendState(&blend, factor, &sample_mask);
      D3D11_BLEND_DESC blend_desc = {};
      if (blend)
         blend->GetDesc(&blend_desc);
      com_ptr<ID3D11DepthStencilState> depth;
      UINT stencil_ref = 0;
      context->OMGetDepthStencilState(&depth, &stencil_ref);
      D3D11_DEPTH_STENCIL_DESC depth_desc = { TRUE, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_COMPARISON_LESS };
      if (depth)
         depth->GetDesc(&depth_desc);
      com_ptr<ID3D11RasterizerState> raster;
      context->RSGetState(&raster);
      D3D11_RASTERIZER_DESC raster_desc = {};
      raster_desc.CullMode = D3D11_CULL_BACK;
      if (raster)
         raster->GetDesc(&raster_desc);
      ID3D11RenderTargetView* rtvs[8] = {};
      ID3D11DepthStencilView* dsv = nullptr;
      context->OMGetRenderTargets(8, rtvs, &dsv);
      uint32_t rtv_count = 0;
      DXGI_FORMAT rtv0_format = DXGI_FORMAT_UNKNOWN;
      for (uint32_t i = 0; i < 8; i++)
         if (rtvs[i])
         {
            if (rtv_count == 0)
            {
               D3D11_RENDER_TARGET_VIEW_DESC d;
               rtvs[i]->GetDesc(&d);
               rtv0_format = d.Format;
            }
            rtv_count++;
            rtvs[i]->Release();
         }
      int dsv_slice = -1;
      uintptr_t dsv_resource = 0;
      D3D11_TEXTURE2D_DESC dsv_tex = {};
      if (dsv)
      {
         D3D11_DEPTH_STENCIL_VIEW_DESC d;
         dsv->GetDesc(&d);
         dsv_slice = d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DARRAY ? int(d.Texture2DArray.FirstArraySlice) : -2;
         com_ptr<ID3D11Resource> resource;
         dsv->GetResource(&resource);
         dsv_resource = uintptr_t(resource.get());
         com_ptr<ID3D11Texture2D> texture;
         if (resource && SUCCEEDED(resource->QueryInterface(&texture)))
            texture->GetDesc(&dsv_tex);
         dsv->Release();
      }
      D3D11_VIEWPORT viewport = {};
      UINT viewports = 1;
      context->RSGetViewports(&viewports, &viewport);
      const D3D11_RENDER_TARGET_BLEND_DESC& rt0 = blend_desc.RenderTarget[0];
      char row[400];
      snprintf(row, sizeof(row), "%016llX,%llX,%u,%u,%d,%08X,%08X,%d,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d,%d,%.3f,%d,%.0f,%.0f,%.0f,%.0f,%d",
         (unsigned long long)key, (unsigned long long)uintptr_t(ib.get()), first_index, index_count, vertex_offset, vs_hash, ps_hash,
         int(depth_desc.DepthEnable), int(depth_desc.DepthWriteMask), int(depth_desc.DepthFunc), int(rt0.BlendEnable), int(rt0.SrcBlend), int(rt0.DestBlend), int(rt0.RenderTargetWriteMask), int(blend_desc.AlphaToCoverageEnable),
         rtv_count, int(rtv0_format), dsv_slice, int(raster_desc.DepthBias), raster_desc.SlopeScaledDepthBias, int(raster_desc.CullMode),
         viewport.TopLeftX, viewport.TopLeftY, viewport.Width, viewport.Height, int(dynamic));
      {
         char target[96];
         snprintf(target, sizeof(target), ",%llX,%u,%u,%u,%d", (unsigned long long)dsv_resource, dsv_tex.Width, dsv_tex.Height, dsv_tex.ArraySize, int(dsv_tex.Format));
         strncat_s(row, sizeof(row), target, _TRUNCATE);
      }
      const std::lock_guard lock(g_draw_probe_mutex);
      auto it = g_draw_probe.find(row);
      if (it != g_draw_probe.end())
         it->second++;
      else if (g_draw_probe.size() < 200000)
         g_draw_probe.emplace(row, 1);
   }

   static bool OnRTAODrawIndexed(reshade::api::command_list* cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
   {
      if (g_draw_probe_active.load(std::memory_order_relaxed) && cmd_list->get_device()->get_api() == reshade::api::device_api::d3d11)
         DrawProbe(reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native()), index_count, first_index, vertex_offset);
      if (!g_rtao_capture_active.load(std::memory_order_relaxed))
         return false;
      auto* device = cmd_list->get_device();
      if (device->get_api() != reshade::api::device_api::d3d11)
         return false;
      auto* context = reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native());
      // Main view draws, or shadow map draws (sun casters out of view). Shadow map draws feed no main view diagnostics.
      bool shadow_view = false;
      uint32_t shadow_slice = 0xFF;
      if (!IsMainViewDepthWrite(context, g_rtao_capture_pass == 0))
      {
         uint32_t slice = 0;
         if (!g_rtao_shadow_casters || !IsShadowViewDepthWrite(context, &slice))
            return false;
         g_rtao_shadow_slice_draws[(std::min)(slice, 7u)]++;
         shadow_slice = slice;
         if (g_rtao_shadow_near_only && slice >= uint32_t((std::max)(g_rtao_shadow_near_slices, 1)) && slice != kFlatShadowSlice)
         {
            ShadowAudit(context, slice, "far_slice", index_count);
            g_rtao_shadow_far_skipped++;
            return false;
         }
         shadow_view = true;
         g_rtao_shadow_view_draws++;
      }
      D3D11_PRIMITIVE_TOPOLOGY topology;
      context->IAGetPrimitiveTopology(&topology);
      if (shadow_view && topology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
      {
         ShadowAudit(context, shadow_slice, "topology", index_count);
         return false;
      }
      if (topology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
      {
         g_rtao_draws_not_triangle_list++;
         char detail[48];
         snprintf(detail, sizeof(detail), "(topology %d, %u indices, %u instances)", int(topology), index_count, instance_count);
         LogRTAOUncaptured(context, "not a triangle list", detail);
         return false;
      }

      com_ptr<ID3D11VertexShader> vs;
      context->VSGetShader(&vs, nullptr, nullptr);
      com_ptr<ID3D11InputLayout> layout;
      context->IAGetInputLayout(&layout);
      com_ptr<ID3D11PixelShader> ps;
      context->PSGetShader(&ps, nullptr, nullptr);
      RTAO::VertexShaderInfo vs_info;
      RTAO::InputLayoutInfo layout_info;
      RTAO::PixelShaderInfo ps_info;
      const char* shadow_audit_outcome = nullptr;
      // Audited once the shader lock is released (ShadowAudit takes it too)
      struct AuditOnExit
      {
         const char*& outcome;
         ID3D11DeviceContext* context;
         uint32_t slice, indices;
         ~AuditOnExit()
         {
            if (outcome)
               ShadowAudit(context, slice, outcome, indices);
         }
      } audit_on_exit{ shadow_audit_outcome, context, shadow_slice, index_count };
      bool alpha_tested = false;
      {
         const std::shared_lock lock(g_rtao_shader_mutex);
         auto vs_it = g_rtao_vertex_shaders.find(uint64_t(uintptr_t(vs.get())));
         if (vs_it == g_rtao_vertex_shaders.end())
         {
            if (!shadow_view)
               g_rtao_draws_unknown_shader++;
            else
               shadow_audit_outcome = "vs_not_reflected";
            return false;
         }
         vs_info = vs_it->second;
         auto layout_it = g_rtao_input_layouts.find(uint64_t(uintptr_t(layout.get())));
         if (layout_it != g_rtao_input_layouts.end())
            layout_info = layout_it->second;
         if (ps)
         {
            auto ps_it = g_rtao_alpha_tested_ps.find(uint64_t(uintptr_t(ps.get())));
            if (ps_it != g_rtao_alpha_tested_ps.end())
            {
               alpha_tested = true;
               ps_info = ps_it->second;
            }
         }
      }
      if (shadow_view)
      {
         // Same family toggles as the main view, no diagnostics. Skinned draws are left out (see DrawInfo::shadow_view).
         if ((vs_info.wind && !g_rtao_wind_meshes) || (alpha_tested && !g_rtao_cutouts) || vs_info.source == RTAO::MatrixSource::None || (vs_info.skinned && !g_rtao_shadow_skinned))
         {
            ShadowAudit(context, shadow_slice, vs_info.source == RTAO::MatrixSource::None ? (vs_info.skip_reason ? vs_info.skip_reason : "no_matrix_source") : (vs_info.skinned ? "skinned" : "family_off"), index_count);
            return false;
         }
         if (!layout_info.has_position || layout_info.position_per_instance)
         {
            ShadowAudit(context, shadow_slice, "layout", index_count);
            return false;
         }
         if (vs_info.skinned)
            for (uint32_t set = 0; set < vs_info.skin_sets; set++)
            {
               const auto& indices = layout_info.skin_indices[set];
               const auto& weights = layout_info.skin_weights[set];
               if (!indices.present || !weights.present || indices.per_instance || weights.per_instance)
               {
                  ShadowAudit(context, shadow_slice, "skin_layout", index_count);
                  return false;
               }
            }
      }
      else
      {
         g_rtao_draws_alpha_tested += alpha_tested;
         // Foliage family toggles (UI): left out of the scene when off
         if ((vs_info.wind && !g_rtao_wind_meshes) || (alpha_tested && !g_rtao_cutouts))
         {
            const std::lock_guard lock(g_rtao_skip_reason_mutex);
            g_rtao_skip_reasons[(vs_info.wind && !g_rtao_wind_meshes) ? "wind (off in the UI)" : "alpha tested (off in the UI)"]++;
            return false;
         }
         g_rtao_draws_by_source[(std::min)(uint32_t(vs_info.source), 4u)]++;
         if (vs_info.skinned)
            g_rtao_draws_skinned++;
         if (vs_info.source == RTAO::MatrixSource::None || (vs_info.skinned && !g_rtao_skinned))
         {
            const char* reason = vs_info.source == RTAO::MatrixSource::None ? (vs_info.skip_reason ? vs_info.skip_reason : "unknown") : "skinned (off)";
            {
               const std::lock_guard lock(g_rtao_skip_reason_mutex);
               g_rtao_skip_reasons[reason]++;
            }
            LogRTAOUncaptured(context, reason);
            return false;
         }
         if (!layout_info.has_position || layout_info.position_per_instance)
         {
            g_rtao_draws_unusable_layout++;
            return false;
         }
         if (vs_info.skinned)
         {
            for (uint32_t set = 0; set < vs_info.skin_sets; set++)
            {
               const auto& indices = layout_info.skin_indices[set];
               const auto& weights = layout_info.skin_weights[set];
               if (!indices.present || !weights.present || indices.per_instance || weights.per_instance)
               {
                  g_rtao_draws_unusable_layout++;
                  return false;
               }
            }
         }
      }

      RTAO::DrawInfo draw;
      draw.vs = &vs_info;
      draw.shadow_view = shadow_view;
      {
         const std::shared_lock lock(g_rtao_shader_mutex);
         auto vh = g_rtao_vertex_shader_hashes.find(uint64_t(uintptr_t(vs.get())));
         draw.debug_vs_hash = vh != g_rtao_vertex_shader_hashes.end() ? vh->second : 0;
         auto ph = ps ? g_rtao_pixel_shader_hashes.find(uint64_t(uintptr_t(ps.get()))) : g_rtao_pixel_shader_hashes.end();
         draw.debug_ps_hash = ph != g_rtao_pixel_shader_hashes.end() ? ph->second : 0;
         draw.redraw_pass = draw.debug_vs_hash == 0xABCEF358u || draw.debug_ps_hash == 0xBBBE692Eu; // The motion vector pass
      }
      draw.shadow_slice = uint8_t((std::min)(shadow_slice, 0xFEu));
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
      // Skinned: the bone palette and the vertex streams of the skin indices/weights (often another slot)
      com_ptr<ID3D11ShaderResourceView> bones;
      com_ptr<ID3D11Buffer> skin_buffers[4];
      if (vs_info.skinned)
      {
         context->VSGetShaderResources(vs_info.bones_slot, 1, &bones);
         draw.bones = bones.get();
         for (uint32_t set = 0; set < vs_info.skin_sets; set++)
         {
            const RTAO::InputLayoutInfo::Element* elements[2] = { &layout_info.skin_indices[set], &layout_info.skin_weights[set] };
            RTAO::DrawInfo::Stream* streams[2] = { &draw.skin_indices[set], &draw.skin_weights[set] };
            for (uint32_t k = 0; k < 2; k++)
            {
               UINT skin_stride = 0, skin_offset = 0;
               com_ptr<ID3D11Buffer>& buffer = skin_buffers[set * 2 + k];
               context->IAGetVertexBuffers(elements[k]->slot, 1, &buffer, &skin_stride, &skin_offset);
               streams[k]->buffer = buffer.get();
               streams[k]->stride = skin_stride;
               streams[k]->offset = skin_offset;
               streams[k]->element_offset = elements[k]->offset;
               streams[k]->format = elements[k]->format;
            }
         }
      }
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
      draw.cutout = alpha_tested;
      // Texture exact cutouts: the pixel shader's alpha texture and material cbuffer, and the TEXCOORD0 stream
      com_ptr<ID3D11ShaderResourceView> alpha_srv;
      com_ptr<ID3D11Buffer> material_cb, uv_buffer;
      if (alpha_tested && g_rtao_cutouts_exact && ps_info.texture_exact && ps_info.alpha_slot != RTAO::kNoOffset && layout_info.texcoord0.present && !layout_info.texcoord0.per_instance)
      {
         context->PSGetShaderResources(ps_info.alpha_slot, 1, &alpha_srv);
         if (ps_info.material_cb_slot != RTAO::kNoOffset)
         {
            com_ptr<ID3D11DeviceContext1> context1;
            UINT first_constant = 0, num_constants = 0;
            if (SUCCEEDED(context->QueryInterface(&context1)))
               context1->PSGetConstantBuffers1(ps_info.material_cb_slot, 1, &material_cb, &first_constant, &num_constants);
            else
               context->PSGetConstantBuffers(ps_info.material_cb_slot, 1, &material_cb);
            draw.material_cb_first_constant = first_constant;
            draw.material_cb_num_constants = num_constants;
         }
         UINT uv_stride = 0, uv_offset = 0;
         context->IAGetVertexBuffers(layout_info.texcoord0.slot, 1, &uv_buffer, &uv_stride, &uv_offset);
         // A reference or scale/bias the shader reads from a cbuffer that isn't bound: no material
         if (material_cb || (ps_info.alpha_ref_cb == RTAO::kNoOffset && ps_info.uv_scalebias_cb == RTAO::kNoOffset))
         {
            draw.ps = &ps_info;
            draw.alpha_srv = alpha_srv.get();
            draw.material_cb = material_cb.get();
            draw.uv.buffer = uv_buffer.get();
            draw.uv.stride = uv_stride;
            draw.uv.offset = uv_offset;
            draw.uv.element_offset = layout_info.texcoord0.offset;
            draw.uv.format = layout_info.texcoord0.format;
         }
      }
      if (g_rtao_dedup)
      {
         uint64_t version = RTAODiscardVersion(vertex_buffer.get()); // DYNAMIC vertex buffers (per frame meshes)
         version = version * 0x9E3779B97F4A7C15ull + RTAODiscardVersion(instance_cb.get());
         version = version * 0x9E3779B97F4A7C15ull + RTAODiscardVersion(instance_buffer.get());
         version = version * 0x9E3779B97F4A7C15ull + RTAODiscardVersion(instance_index.get());
         version = version * 0x9E3779B97F4A7C15ull + RTAODiscardVersion(bones.get());
         version = version * 0x9E3779B97F4A7C15ull + RTAODiscardVersion(per_view_cb.get());
         draw.binding_version = version;
      }

      auto& device_data = *device->get_private_data<DeviceData>();
      const bool captured = device_data.game && GetGameDeviceData(device_data).rtao_capture.CaptureDraw(context, draw);
      if (shadow_view)
         ShadowAudit(context, shadow_slice, device_data.game ? RTAO::GeometryCapture::LastOutcome() : "no_game", index_count);
      if (device_data.game && !captured && !shadow_view)
      {
         D3D11_BUFFER_DESC vb_desc = {}, ib_desc = {};
         if (vertex_buffer)
            vertex_buffer->GetDesc(&vb_desc);
         if (index_buffer)
            index_buffer->GetDesc(&ib_desc);
         char detail[96];
         snprintf(detail, sizeof(detail), "(VB usage %d, IB usage %d, position format %d, index format %d)", int(vb_desc.Usage), int(ib_desc.Usage), int(layout_info.position_format), int(index_format));
         LogRTAOUncaptured(context, "capture refused", detail);
      }
      return false;
   }

   // GPU culled geometry (arkcullgeometry) is drawn indirectly: counted only for now (see RTAO_NOTES.md)
   static bool OnRTAODrawIndirect(reshade::api::command_list* cmd_list, reshade::api::indirect_command type, reshade::api::resource buffer, uint64_t offset, uint32_t draw_count, uint32_t stride)
   {
      if (!g_rtao_capture_active.load(std::memory_order_relaxed) || type != reshade::api::indirect_command::draw_indexed)
         return false;
      if (cmd_list->get_device()->get_api() != reshade::api::device_api::d3d11)
         return false;
      auto* context = reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native());
      if (IsMainViewDepthWrite(context, false))
         g_rtao_indirect_draws += draw_count;
      else if (IsShadowViewDepthWrite(context))
         g_rtao_shadow_indirect_draws += draw_count; // GPU culled shadow casters: not captured
      return false;
   }

   static void OnRTAODestroyResource(reshade::api::device* device, reshade::api::resource resource)
   {
      if (device->get_api() != reshade::api::device_api::d3d11 || resource.handle == 0)
         return;
      auto* native = reinterpret_cast<ID3D11Resource*>(resource.handle);
      D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      native->GetType(&dimension);
      if (dimension != D3D11_RESOURCE_DIMENSION_BUFFER && dimension != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
         return;
      auto* device_data = device->get_private_data<DeviceData>();
      if (device_data && device_data->game)
      {
         auto& game_device_data = GetGameDeviceData(*device_data);
         if (!game_device_data.rtao_capture.IsInitialized())
            return;
         if (dimension == D3D11_RESOURCE_DIMENSION_BUFFER)
            game_device_data.rtao_capture.OnDestroyBuffer(static_cast<ID3D11Buffer*>(native));
         else
            game_device_data.rtao_capture.OnDestroyTexture(native); // Its cutout mask (if any) goes, the pointer may come back
      }
   }

   // H3 (wrong geometry with stable instance counts): the capture copies a game buffer into its arena once, at the first
   // draw that uses it. A DEFAULT buffer the game rewrites later (UpdateSubresource, a copy into it: streamed or pooled
   // meshes) would leave stale geometry in the ray tracing scene. Counted always; with "RTAORecopyWrittenBuffers" the
   // buffer is forgotten (its meshes' BLASes evicted) so the next draw copies it again.
   static void OnRTAOBufferWritten(reshade::api::device* device, reshade::api::resource dest)
   {
      if (device->get_api() != reshade::api::device_api::d3d11 || dest.handle == 0 || !g_rtao_capture_active.load(std::memory_order_relaxed))
         return;
      auto* native = reinterpret_cast<ID3D11Resource*>(dest.handle);
      D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      native->GetType(&dimension);
      if (dimension != D3D11_RESOURCE_DIMENSION_BUFFER)
         return;
      auto* device_data = device->get_private_data<DeviceData>();
      if (device_data && device_data->game)
         GetGameDeviceData(*device_data).rtao_capture.OnBufferWritten(static_cast<ID3D11Buffer*>(native), g_rtao_recopy_written);
   }
   static bool OnRTAOUpdateBuffer(reshade::api::device* device, const void*, reshade::api::resource dest, uint64_t, uint64_t)
   {
      OnRTAOBufferWritten(device, dest);
      return false;
   }
   static bool OnRTAOUpdateBufferCommand(reshade::api::command_list* cmd_list, const void*, reshade::api::resource dest, uint64_t, uint64_t)
   {
      OnRTAOBufferWritten(cmd_list->get_device(), dest);
      return false;
   }
   // Diagnostic capture: copies involving a texture of at least 1024 (depth shadow maps and anything else that size)
   static void CopyProbe(const char* kind, reshade::api::resource source, uint32_t source_sub, reshade::api::resource dest, uint32_t dest_sub)
   {
      if (!g_draw_probe_active.load(std::memory_order_relaxed))
         return;
      auto describe = [](reshade::api::resource r, D3D11_TEXTURE2D_DESC& out) {
         out = {};
         auto* resource = reinterpret_cast<ID3D11Resource*>(r.handle);
         com_ptr<ID3D11Texture2D> texture;
         if (resource && SUCCEEDED(resource->QueryInterface(&texture)))
            texture->GetDesc(&out);
      };
      D3D11_TEXTURE2D_DESC s, d;
      describe(source, s);
      describe(dest, d);
      if ((std::max)(s.Width, d.Width) < 1024)
         return;
      char row[200];
      snprintf(row, sizeof(row), "%s,%llX,%u,%u,%u,%d,%u,%llX,%u,%u,%u,%d,%u", kind, (unsigned long long)source.handle, s.Width, s.Height, s.ArraySize, int(s.Format), source_sub,
         (unsigned long long)dest.handle, d.Width, d.Height, d.ArraySize, int(d.Format), dest_sub);
      const std::lock_guard lock(g_copy_probe_mutex);
      if (g_copy_probe.size() < 10000)
         g_copy_probe[row]++;
   }

   static bool OnRTAOCopyTextureRegion(reshade::api::command_list*, reshade::api::resource source, uint32_t source_subresource, const reshade::api::subresource_box*, reshade::api::resource dest, uint32_t dest_subresource, const reshade::api::subresource_box*, reshade::api::filter_mode)
   {
      CopyProbe("region", source, source_subresource, dest, dest_subresource);
      return false;
   }

   static bool OnRTAOCopyResource(reshade::api::command_list* cmd_list, reshade::api::resource source, reshade::api::resource dest)
   {
      CopyProbe("resource", source, 0, dest, 0);
      OnRTAOBufferWritten(cmd_list->get_device(), dest);
      return false;
   }
   static bool OnRTAOCopyBufferRegion(reshade::api::command_list* cmd_list, reshade::api::resource, uint64_t, reshade::api::resource dest, uint64_t, uint64_t)
   {
      OnRTAOBufferWritten(cmd_list->get_device(), dest);
      return false;
   }

   static void OnMapBufferRegion(reshade::api::device* device, reshade::api::resource resource, uint64_t offset, uint64_t size, reshade::api::map_access access, void** data)
   {
      if (device->get_api() != reshade::api::device_api::d3d11) // e.g. RTAO's D3D12 device
         return;
      ID3D11Buffer* buffer = reinterpret_cast<ID3D11Buffer*>(resource.handle);
      if (access == reshade::api::map_access::write_discard && g_rtao_dedup && g_rtao_capture_active.load(std::memory_order_relaxed))
         RTAONoteDiscard(buffer);
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
         // RTAO/RT sun: this thread's last per view constants (as the GPU will see them), for the SSAO draw recorded next
         // on this thread (see t_rtao_per_view)
         const CBPerViewGlobals& written = *reinterpret_cast<const CBPerViewGlobals*>(device_data.cb_per_view_global_buffer_map_data);
         if (MatrixIsProjection(written.cb_projectionmatrix))
         {
            t_rtao_per_view = written;
            t_rtao_per_view_valid = true;
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
      reshade::get_config_value(runtime, NAME, "RTAOSkinned", g_rtao_skinned);
      reshade::get_config_value(runtime, NAME, "RTAODynamic", g_rtao_dynamic);
      reshade::get_config_value(runtime, NAME, "RTAOShadowCasters", g_rtao_shadow_casters);
      reshade::get_config_value(runtime, NAME, "RTSunShadowCastersOnly", g_rtsun_casters_only);
      reshade::get_config_value(runtime, NAME, "RTSunShadowMapCheck", g_rtsun_shadow_consult);
      reshade::get_config_value(runtime, NAME, "RTDiagAutoStartSeconds", g_rtdiag_auto_seconds);
      reshade::get_config_value(runtime, NAME, "RTDiagAutoLevelSeconds", g_rtdiag_auto_level_seconds);
      reshade::get_config_value(runtime, NAME, "RTAOShadowFlatTargets", g_rtao_shadow_flat);
      reshade::get_config_value(runtime, NAME, "RTAOShadowSkinned", g_rtao_shadow_skinned);
      reshade::get_config_value(runtime, NAME, "RTSunShadowMapTolerance", g_rtsun_shadow_tolerance);
      reshade::get_config_value(runtime, NAME, "RTAOShadowNearSlices", g_rtao_shadow_near_slices);
      reshade::get_config_value(runtime, NAME, "RTAOShadowNearOnly", g_rtao_shadow_near_only);
      reshade::get_config_value(runtime, NAME, "RTAOStrength", g_rtao_strength);
      g_rtao_strength = std::clamp(g_rtao_strength, 0.25f, 4.f);
      g_rtao_radius = std::clamp(g_rtao_radius, 0.05f, 50.f);
      g_rtao_rays = std::clamp(g_rtao_rays, 1, 64);
      g_rtao_capture_pass = std::clamp(g_rtao_capture_pass, 0, 1);
      reshade::get_config_value(runtime, NAME, "RTAOWindMeshes", g_rtao_wind_meshes);
      reshade::get_config_value(runtime, NAME, "RTAOCutouts", g_rtao_cutouts);
      reshade::get_config_value(runtime, NAME, "RTAOCutoutOpacity", g_rtao_cutout_opacity);
      g_rtao_cutout_opacity = std::clamp(g_rtao_cutout_opacity, 0.f, 1.f);
      reshade::get_config_value(runtime, NAME, "RTSunMode", g_rtsun_mode);
      reshade::get_config_value(runtime, NAME, "RTSunAngle", g_rtsun_angle);
      reshade::get_config_value(runtime, NAME, "RTSunRays", g_rtsun_rays);
      reshade::get_config_value(runtime, NAME, "RTSunOutput", g_rtsun_output);
      g_rtsun_mode = std::clamp(g_rtsun_mode, 0, int(RTSUN_MODE_COUNT) - 1);
      g_rtsun_angle = std::clamp(g_rtsun_angle, 0.f, 10.f);
      g_rtsun_rays = std::clamp(g_rtsun_rays, 1, 16);
      g_rtsun_output = std::clamp(g_rtsun_output, 0, int(RTSun::Output::Count) - 1);
      cb_luma_global_settings.GameSettings.RTSunMode = uint32_t(g_rtsun_mode);
      reshade::get_config_value(runtime, NAME, "RTAOQuality", g_rtao_quality);
      g_rtao_quality = std::clamp(g_rtao_quality, 0, int(RTAO_QUALITY_COUNT) - 1);
      reshade::get_config_value(runtime, NAME, "RTAOFastCutouts", g_rtao_fast_cutouts);
      reshade::get_config_value(runtime, NAME, "RTAODedup", g_rtao_dedup);
      reshade::get_config_value(runtime, NAME, "RTAOSkinnedHoldFrames", g_rtao_skinned_hold);
      reshade::get_config_value(runtime, NAME, "RTAOBlasBudget", g_rtao_blas_budget);
      reshade::get_config_value(runtime, NAME, "RTAORecopyWrittenBuffers", g_rtao_recopy_written);
      reshade::get_config_value(runtime, NAME, "RTAOCutoutsExact", g_rtao_cutouts_exact);

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
      static const char* ao_view_names[] = { "Off", "Final AO (after the game's blur)", "RTAO output (before the game's blur)", "RT sun visibility (before the game's blur)" };
      ImGui::Combo("View AO only (debug)", &g_ao_view_mode, ao_view_names, IM_ARRAYSIZE(ao_view_names));
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Shows the AO buffer fullscreen in grey (white = no occlusion) instead of the game.\n\"Final AO\" works with every AO mode (the game's SSAO, XeGTAO, RTAO), so you can compare them directly.\nWith RTAO, the RTAO Output choice (e.g. BLAS coverage, no denoiser) is what shows here.\n\"RT sun visibility\" shows the ray traced sun/moon shadows (white = lit), see RT Sun Shadows below.");
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
         static const char* rtao_quality_names[RTAO_QUALITY_COUNT] = { "Low (half resolution trace)", "Medium (half the rays)", "High" };
         if (ImGui::Combo("RTAO Quality", &g_rtao_quality, rtao_quality_names, RTAO_QUALITY_COUNT))
            reshade::set_config_value(runtime, NAME, "RTAOQuality", g_rtao_quality);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Ray tracing cost of RTAO (the scene is the same in all of them).\nLow: \"RTAO Rays\" at one pixel of each 2x2 block (cycling over 4 frames), upsampled by the denoiser: about a quarter of the trace cost.\nMedium: every pixel, half of \"RTAO Rays\".\nHigh: every pixel, all of \"RTAO Rays\".");
         DrawResetButton(g_rtao_quality, int(RTAO_QUALITY_LOW), "RTAOQuality");
         if (ImGui::SliderInt("RTAO Rays", &g_rtao_rays, 1, 16))
            g_rtao_rays = std::clamp(g_rtao_rays, 1, 64);
         if (ImGui::IsItemDeactivatedAfterEdit())
            reshade::set_config_value(runtime, NAME, "RTAORays", g_rtao_rays);
         static const char* capture_pass_names[] = { "Depth only draws", "All depth writing draws" };
         if (ImGui::Combo("RTAO Capture", &g_rtao_capture_pass, capture_pass_names, IM_ARRAYSIZE(capture_pass_names)))
            reshade::set_config_value(runtime, NAME, "RTAOCapturePass", g_rtao_capture_pass);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Which main view draws feed the ray tracing scene. Use \"All\" if the coverage view shows missing (grey) geometry.");
         if (ImGui::Checkbox("RTAO: skinned characters", &g_rtao_skinned))
            reshade::set_config_value(runtime, NAME, "RTAOSkinned", g_rtao_skinned);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Characters (skinned meshes) occlude in the ray traced AO: their vertices are skinned again in a compute pass every frame\nand get a BLAS refit per draw. Costs some GPU time per visible character.");
         if (ImGui::Checkbox("RTAO: dynamic meshes (doors, windows)", &g_rtao_dynamic))
            reshade::set_config_value(runtime, NAME, "RTAODynamic", g_rtao_dynamic);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Movable objects whose vertices the game rewrites every frame (openable doors, windows, cabinets):\ncopied at each draw and built into a per frame BLAS.");
         if (ImGui::Checkbox("RTAO: shadow map casters", &g_rtao_shadow_casters))
            reshade::set_config_value(runtime, NAME, "RTAOShadowCasters", g_rtao_shadow_casters);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Also captures what the game draws into its shadow maps: walls, roofs and window frames behind or beside the camera,\nwhich the camera never draws but which cast the sun's shadows into view (RT sun shadows). Characters are not taken from there.\nChanges apply to newly captured geometry; what is already in the scene ages out within a few seconds.");
         if (ImGui::SliderFloat("RTAO Strength", &g_rtao_strength, 0.25f, 4.f, "%.2f"))
            g_rtao_strength = std::clamp(g_rtao_strength, 0.25f, 4.f);
         if (ImGui::IsItemDeactivatedAfterEdit())
            reshade::set_config_value(runtime, NAME, "RTAOStrength", g_rtao_strength);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Deepens (> 1) or lightens (< 1) the ray traced AO: a power curve on the result. 1 = as traced, 2 = default.");
         DrawResetButton(g_rtao_strength, 2.f, "RTAOStrength");
         if (ImGui::Checkbox("RTAO: wind meshes", &g_rtao_wind_meshes))
            reshade::set_config_value(runtime, NAME, "RTAOWindMeshes", g_rtao_wind_meshes);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Vertex animated vegetation (plants, trees, hanging things) in the ray tracing scene, in its rest pose (the wind sway is not replayed).\nChanges apply to newly captured geometry; what is already in the scene ages out within a few seconds.");
         if (ImGui::Checkbox("RTAO: alpha tested cutouts", &g_rtao_cutouts))
            reshade::set_config_value(runtime, NAME, "RTAOCutouts", g_rtao_cutouts);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Alpha tested geometry (leaves, grates, fences, hair cards) in the ray tracing scene, as partially occluding cutouts (see the opacity below).\nChanges apply to newly captured geometry; what is already in the scene ages out within a few seconds.");
         if (ImGui::SliderFloat("RTAO Cutout Opacity", &g_rtao_cutout_opacity, 0.f, 1.f, "%.2f"))
            g_rtao_cutout_opacity = std::clamp(g_rtao_cutout_opacity, 0.f, 1.f);
   if (ImGui::IsItemDeactivatedAfterEdit())
            reshade::set_config_value(runtime, NAME, "RTAOCutoutOpacity", g_rtao_cutout_opacity);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How much alpha tested surfaces occlude: the chance that a ray crossing one stops there.\nWith \"texture exact\" on, only for the cutouts that have no alpha mask (yet).\n0 = left out, 1 = as solid as the rest (cheapest).");
         DrawResetButton(g_rtao_cutout_opacity, 0.5f, "RTAOCutoutOpacity");
         if (ImGui::Checkbox("RTAO: fast cutouts", &g_rtao_fast_cutouts))
            reshade::set_config_value(runtime, NAME, "RTAOFastCutouts", g_rtao_fast_cutouts);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Decides once per ray whether cutouts are solid or absent (instead of at every crossed leaf), which skips the per triangle\ncandidate handling: cheaper in foliage (about a quarter of the trace there), slightly different AO under dense vegetation.\nWith \"texture exact\" on, the masks take precedence: the per ray coin only decides cutouts without a mask, so it saves nothing.");
         if (ImGui::Checkbox("RTAO: skip repeated draws", &g_rtao_dedup))
            reshade::set_config_value(runtime, NAME, "RTAODedup", g_rtao_dedup);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Captures an object once per frame even if several passes draw it (depth prepass, G-buffer): saves the per draw\ncapture work on the game's render threads and duplicate per frame BLASes of characters. Turn off if geometry goes missing.");
         if (ImGui::SliderInt("RTAO: skinned mesh hold (frames)", &g_rtao_skinned_hold, 0, 120))
            reshade::set_config_value(runtime, NAME, "RTAOSkinnedHoldFrames", g_rtao_skinned_hold);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How many frames a skinned mesh (a character, a swinging window flap) keeps its last pose when a frame misses its\ndraw. A newer draw of the same mesh replaces it at once. 0: current frame only.");
         if (ImGui::Checkbox("RTAO cutouts: texture exact", &g_rtao_cutouts_exact))
            reshade::set_config_value(runtime, NAME, "RTAOCutoutsExact", g_rtao_cutouts_exact);
         if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Alpha tested surfaces occlude exactly where their texture is solid (leaf shaped AO), like the game's alpha test:\neach material's alpha is rendered once into a mask the rays test per hit.\nOff: every cutout uses the opacity above.");

         auto& game_device_data = GetGameDeviceData(device_data);
         if (game_device_data.rtao_capture.IsInitialized() && ImGui::TreeNode("RTAO stats"))
         {
            const RTAO::CaptureStats stats = game_device_data.rtao_capture.GetStats();
            ImGui::Text("GPU time %.2f ms: capture (readback, BLAS builds, per frame meshes) %.2f, TLAS build %.2f, trace %.2f, denoise %.2f", game_device_data.rtao_gpu_us.load() / 1000.0,
               game_device_data.rtao_gpu_capture_us.load() / 1000.0, game_device_data.rtao_gpu_tlas_us.load() / 1000.0, game_device_data.rtao_gpu_trace_us.load() / 1000.0, game_device_data.rtao_gpu_filter_us.load() / 1000.0);
            ImGui::Text("Draws seen %llu, captured %llu (%llu instances), repeated in the frame (skipped) %llu + %llu per frame meshes", (unsigned long long)stats.draws_seen, (unsigned long long)stats.draws_captured,
               (unsigned long long)stats.instances_captured, (unsigned long long)stats.dedup_draws, (unsigned long long)stats.dedup_frame_meshes);
            ImGui::Text("Shadow map draws %llu (last map %ux%u, %u slices): captured %llu (%llu instances, %llu per frame meshes), skipped (characters) %llu, MVP instances rejected %llu",
               (unsigned long long)g_rtao_shadow_view_draws.load(), g_rtao_shadow_view_width.load(), g_rtao_shadow_view_height.load(), g_rtao_shadow_view_slices.load(), (unsigned long long)stats.shadow_draws,
               (unsigned long long)stats.shadow_instances, (unsigned long long)stats.shadow_frame_meshes, (unsigned long long)stats.shadow_skipped, (unsigned long long)stats.shadow_rejected);
            {
               const RTAO::RetentionStats& r = stats.retention;
               ImGui::Text("TLAS retained %u (promoted %u, young %u; main view only %u, shadow maps only %u, both %u), last readback: new %u, evicted %u + %u young, shadow young %u; left out (BLAS not built) %u; MVP rekeyed %llu of %llu",
                  r.promoted + r.young, r.promoted, r.young, r.main_only, r.shadow_only, r.both, r.added_new, r.evicted_promoted, r.evicted_young, r.shadow_young, game_device_data.rtao_tlas_missing_blas.load(),
                  (unsigned long long)stats.mvp_rekeyed, (unsigned long long)stats.mvp_instances);
            }
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
            ImGui::Text("Vertex shaders reflected: none %llu, direct %llu, indexed %llu, cbuffer %llu, MVP %llu (skinned %llu)", (unsigned long long)g_rtao_vs_by_source[0].load(), (unsigned long long)g_rtao_vs_by_source[1].load(),
               (unsigned long long)g_rtao_vs_by_source[2].load(), (unsigned long long)g_rtao_vs_by_source[3].load(), (unsigned long long)g_rtao_vs_by_source[4].load(), (unsigned long long)g_rtao_vs_skinned.load());
            ImGui::Text("Foliage: wind draws captured %llu (%llu instances), alpha tested draws %llu (captured %llu, %llu instances), cutouts read back last frame %llu, in the TLAS %u",
               (unsigned long long)stats.wind_draws, (unsigned long long)stats.wind_instances, (unsigned long long)g_rtao_draws_alpha_tested.load(), (unsigned long long)stats.cutout_draws,
               (unsigned long long)stats.cutout_instances, (unsigned long long)stats.frame_cutout_instances, game_device_data.rtao_tlas_cutouts.load());
            ImGui::Text("Pixel shaders reflected %llu: with discard %llu, alpha tested %llu (texture exact replayable %llu)", (unsigned long long)g_rtao_ps_reflected.load(), (unsigned long long)g_rtao_ps_discard.load(),
               (unsigned long long)g_rtao_ps_alpha_tested.load(), (unsigned long long)g_rtao_ps_texture_exact.load());
            {
               const RTAO::CutoutStats cutout = game_device_data.rtao_capture.GetCutoutStats();
               ImGui::Text("Texture exact cutouts (%s): draws with a mask %llu, stochastic %llu; materials %llu (masked %llu, pending %llu)", game_device_data.rtao_capture.IsCutoutsAvailable() ? "available" : "unavailable",
                  (unsigned long long)cutout.draws_exact, (unsigned long long)cutout.draws_no_material, (unsigned long long)cutout.materials, (unsigned long long)cutout.materials_ready, (unsigned long long)cutout.materials_pending);
               ImGui::Text("Alpha atlas slices %llu/%u (overflow %llu), records %llu (overflow %llu), strips rendered %llu, copied %llu, deferred %llu, abandoned %llu", (unsigned long long)cutout.atlas_slices_used,
                  RTAO::CutoutMaterials::kAtlasSlices, (unsigned long long)cutout.atlas_overflow, (unsigned long long)cutout.records, (unsigned long long)cutout.record_overflow, (unsigned long long)cutout.strips_rendered,
                  (unsigned long long)cutout.strips_copied, (unsigned long long)cutout.strips_deferred, (unsigned long long)cutout.strips_abandoned);
               ImGui::Text("Cutout candidates last frame: masked %llu (%llu solid), stochastic %llu", (unsigned long long)game_device_data.rtao_cutout_tests.load(), (unsigned long long)game_device_data.rtao_cutout_hits.load(),
                  (unsigned long long)game_device_data.rtao_cutout_fallback_tests.load());
            }
            ImGui::Text("Skinned draws: main view %llu, captured %llu, skipped: instanced %llu, copy pending %llu", (unsigned long long)g_rtao_draws_skinned.load(), (unsigned long long)stats.skinned_captured,
               (unsigned long long)stats.skinned_instanced, (unsigned long long)stats.skinned_waiting);
            ImGui::Text("Dynamic meshes: captured %llu, skipped: instanced %llu, copy pending %llu, too large %llu", (unsigned long long)stats.dynamic_captured, (unsigned long long)stats.dynamic_instanced,
               (unsigned long long)stats.dynamic_waiting, (unsigned long long)stats.dynamic_too_large);
            ImGui::Text("Per frame meshes last frame: %llu draws, %llu vertices; soup %llu / indexed %llu, over budget %llu; BLAS builds %llu, refits %llu, failed %llu, live %llu (%.1f MB)",
               (unsigned long long)stats.frame_draws, (unsigned long long)stats.frame_vertices, (unsigned long long)stats.frame_soup, (unsigned long long)stats.frame_indexed, (unsigned long long)stats.frame_overflow,
               (unsigned long long)stats.frame_blas_builds, (unsigned long long)stats.frame_blas_refits, (unsigned long long)stats.frame_blas_failed, (unsigned long long)stats.frame_blas_count, stats.frame_blas_bytes / 1048576.0);
            ImGui::TreePop();
         }
      }

      DrawRTSunSettings(device_data);
   }

   void DrawRTSunSettings(DeviceData& device_data)
   {
      reshade::api::effect_runtime* runtime = nullptr;
      ImGui::SeparatorText("Sun Shadows");
      static const char* rtsun_mode_names[RTSUN_MODE_COUNT] = { "Off (game shadow maps)", "Ray traced (experimental)", "Ray traced x game shadow maps" };
      if (ImGui::Combo("RT Sun Shadows", &g_rtsun_mode, rtsun_mode_names, RTSUN_MODE_COUNT))
      {
         g_rtsun_mode = std::clamp(g_rtsun_mode, 0, int(RTSUN_MODE_COUNT) - 1);
         cb_luma_global_settings.GameSettings.RTSunMode = uint32_t(g_rtsun_mode);
         device_data.cb_luma_global_settings_dirty = true;
         reshade::set_config_value(runtime, NAME, "RTSunMode", g_rtsun_mode);
      }
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Ray traced sun/moon shadows for the world (the game's screen shadows pass), from RTAO's ray tracing scene.\n"
                           "Ray traced: replaces the shadow maps. Ray traced x game: both (an object missing from the ray tracing scene still casts its shadow map shadow).\n"
                           "Characters' skin and hair, and transparent surfaces, keep the game's shadow maps (they don't read the screen shadows).\nNeeds Ambient Occlusion = RTAO.");
      if (g_rtsun_mode == RTSUN_MODE_OFF)
         return;
      if (GetEffectiveAOMode() != AO_MODE_RTAO)
      {
         ImGui::TextWrapped("Needs Ambient Occlusion = RTAO (the sun shares its ray tracing scene and D3D12 pass). Using the game's shadows.");
         return;
      }
      if (ImGui::SliderFloat("RT Sun Angular Size", &g_rtsun_angle, 0.f, 5.f, "%.2f deg"))
         g_rtsun_angle = std::clamp(g_rtsun_angle, 0.f, 10.f);
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "RTSunAngle", g_rtsun_angle);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("The light's apparent diameter: the softness of the penumbra (wider with the distance to the caster).\n0.53 is the real sun, 0 is a hard shadow. Wider costs denoiser noise, not trace time.");
      DrawResetButton(g_rtsun_angle, 0.75f, "RTSunAngle");
      if (ImGui::SliderInt("RT Sun Rays", &g_rtsun_rays, 1, 8))
         g_rtsun_rays = std::clamp(g_rtsun_rays, 1, 16);
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "RTSunRays", g_rtsun_rays);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Rays per pixel per frame (in the penumbra the denoiser averages them over frames).");
      static const char* rtsun_output_names[int(RTSun::Output::Count)] = { "Denoised", "Raw (no denoiser)", "Geometric N.L (no rays, debug)", "Denoiser history (debug)", "Raw hard (no denoiser, no penumbra)",
         "Scene view: from the camera (debug)", "Scene view: from the sun (debug)", "Blocker ids (debug)", "Blocker ids, all objects (debug)" };
      if (ImGui::Combo("RT Sun Output", &g_rtsun_output, rtsun_output_names, int(RTSun::Output::Count)))
      {
         g_rtsun_output = std::clamp(g_rtsun_output, 0, int(RTSun::Output::Count) - 1);
         reshade::set_config_value(runtime, NAME, "RTSunOutput", g_rtsun_output);
      }
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("What the trace writes. Check it with \"View AO only\" = RT sun visibility.");
      // Debug (shared with RTAO's scene), next to the sun output where they are used
      if (ImGui::Button(g_rtdiag.active ? "RT diagnostic capture: running..." : "RT diagnostic capture (24 s, Ctrl+Shift+Pause)") && !g_rtdiag.active)
         g_rtdiag.start_requested = true;
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Saves ~120 captures over 24 s into RTDiag\\<time> next to the game exe: 20 s with the current settings, then 4 s with the ray tracing scene frozen.\n"
                           "Each has, from the same frame: the game's own shadow, the RT raw and denoised shadow, the scene view, depth and the final image.\n"
                           "Stand where the problem shows and turn slowly (or hold still) while it runs.");
      ImGui::SameLine();
      ImGui::TextUnformatted(g_rtdiag.status.c_str());
      if (ImGui::Checkbox("RT debug: re-copy buffers the game rewrites", &g_rtao_recopy_written))
         reshade::set_config_value(runtime, NAME, "RTAORecopyWrittenBuffers", g_rtao_recopy_written);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Test: when the game writes into a vertex/index buffer the ray tracing scene already copied (streamed or pooled\nmeshes), forget the copy so the next draw copies the new contents. Off: the count is still logged (\"RTAO diag\").");
      ImGui::Checkbox("RT test: openables from the shadow map", &g_rtdbg_shadow_openables);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Off: openable windows/doors only enter the ray tracing scene from the camera's draws (not saved).");
      if (ImGui::Checkbox("RT sun: shadow map check", &g_rtsun_shadow_consult))
         reshade::set_config_value(runtime, NAME, "RTSunShadowMapCheck", g_rtsun_shadow_consult);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("An object the game never drew into its shadow map (window glass, fake light shafts) blocks the ray traced sun only\n"
            "where the game's own shadow map has a surface (or a shadow) there. Objects drawn into the shadow map, characters\nand openable doors/windows always block.");
      if (ImGui::SliderFloat("RT sun: shadow map check tolerance", &g_rtsun_shadow_tolerance, 0.f, 1.f, "%.2f"))
         reshade::set_config_value(runtime, NAME, "RTSunShadowMapTolerance", g_rtsun_shadow_tolerance);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("How far (world units, plus two shadow map texels) a hit may be behind the shadow map's surface and still count.");
      if (ImGui::Checkbox("RT sun: only shadow casting objects block it", &g_rtsun_casters_only))
         reshade::set_config_value(runtime, NAME, "RTSunShadowCastersOnly", g_rtsun_casters_only);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Objects the game draws for the camera but never into its shadow maps (window glass, fake light shafts) don't block\nthe ray traced sun, like the game's own shadows. They still occlude the AO.");
      if (ImGui::Checkbox("RT: shadow map casters from near slices only", &g_rtao_shadow_near_only))
         reshade::set_config_value(runtime, NAME, "RTAOShadowNearOnly", g_rtao_shadow_near_only);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("The game draws simplified copies of buildings (no proper window openings) into its far shadow cascades.\nOn: only the near cascades' draws enter the ray tracing scene.");
      if (ImGui::SliderInt("RT: near shadow slices", &g_rtao_shadow_near_slices, 1, 8))
         g_rtao_shadow_near_slices = std::clamp(g_rtao_shadow_near_slices, 1, 8);
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "RTAOShadowNearSlices", g_rtao_shadow_near_slices);
      ImGui::Checkbox("RT test: old shadow map retention", &g_rtdbg_shadow_young_long);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("On: the previous rule (3 sightings to keep a shadow map object, else dropped after 4 frames) instead of keeping static ones until unloaded (not saved).");
      ImGui::Checkbox("RT debug: freeze ray tracing scene", &g_rtao_freeze_scene);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("Keeps tracing the current ray tracing scene (no new geometry, no removals) until unchecked. Not saved.\n"
                           "If the RT shadows still change while frozen, the cause is in the trace or the denoiser, not in what is captured.");
      if (ImGui::SliderInt("RT BLAS builds per frame", &g_rtao_blas_budget, 8, 1024))
         g_rtao_blas_budget = std::clamp(g_rtao_blas_budget, 8, 1024);
      if (ImGui::IsItemDeactivatedAfterEdit())
         reshade::set_config_value(runtime, NAME, "RTAOBlasBudget", g_rtao_blas_budget);
      if (ImGui::IsItemHovered())
         ImGui::SetTooltip("New meshes built for ray tracing per frame. A mesh waiting for its build is missing from the scene (late casters).");

      auto& game_device_data = GetGameDeviceData(device_data);
      std::string status;
      {
         const std::lock_guard lock(g_rtsun_status_mutex);
         status = g_rtsun_status;
      }
      ImGui::TextWrapped("RT sun status: %s", status.empty() ? "waiting for the RTAO pass" : status.c_str());
      if (ImGui::TreeNode("RT sun stats"))
      {
         float dir[3] = {}, sun[3] = {};
         uint32_t dir_frame = 0;
         if (game_device_data.rtsun_light.GetDirection(dir, &dir_frame, sun))
         {
            ImGui::Text("Light direction (toward the light) %.3f %.3f %.3f, %u frames old", dir[0], dir[1], dir[2], game_device_data.rtao_frame - dir_frame);
            ImGui::Text("Atmosphere sun %.3f %.3f %.3f", sun[0], sun[1], sun[2]);
         }
         else
         {
            ImGui::Text("Light direction: not captured yet (no screen shadows draw seen)");
         }
         ImGui::Text("GPU time (trace + denoise + copy): %.2f ms", game_device_data.rtsun_gpu_us.load() / 1000.0);
         ImGui::Text("Traces %u, failed %u; screen shadows draws using it %u, using the game's (no fresh trace) %u", game_device_data.rtsun_pass.TraceCount(),
            game_device_data.rtsun_record_failures.load(), game_device_data.rtsun_bound_draws.load(), game_device_data.rtsun_stale_draws.load());
         const RTAO::AOPass::CutoutCounters counters = game_device_data.rtsun_pass.GetCutoutCounters();
         ImGui::Text("Cutout candidates last frame: with a mask %llu (%llu solid), stochastic fallback %llu", (unsigned long long)counters.exact_tests,
            (unsigned long long)counters.exact_hits, (unsigned long long)counters.fallback_tests);
         ImGui::TreePop();
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
      shader_hashes_ScreenShadows.pixel_shaders.emplace(0xF324B47B); // DH2 "screenShadowsNoJitter"
      shader_hashes_ScreenShadows.pixel_shaders.emplace(0x6C8C2397); // DH2 "screenShadows" (dithered)
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
      reshade::unregister_event<reshade::addon_event::update_buffer_region>(Dishonored2::OnRTAOUpdateBuffer);
      reshade::unregister_event<reshade::addon_event::update_buffer_region_command>(Dishonored2::OnRTAOUpdateBufferCommand);
      reshade::unregister_event<reshade::addon_event::copy_resource>(Dishonored2::OnRTAOCopyResource);
      reshade::unregister_event<reshade::addon_event::copy_buffer_region>(Dishonored2::OnRTAOCopyBufferRegion);
      reshade::unregister_event<reshade::addon_event::copy_texture_region>(Dishonored2::OnRTAOCopyTextureRegion);
   }

   CoreMain(hModule, ul_reason_for_call, lpReserved);

   return TRUE;
}