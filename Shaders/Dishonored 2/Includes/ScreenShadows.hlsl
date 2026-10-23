// The game's screen shadows pass (sun/moon visibility mask for the opaque world, see "RT sun shadows" in
// Source/Games/Dishonored 2/RTAO_NOTES.md), ported from the disassembly of 0xF324B47B ("screenShadowsNoJitter") and
// 0x6C8C2397 ("screenShadows", the same plus a 4x4 dither of the lookup toward the camera).
// Output: cloud shadow * cascaded shadow map PCF, in all channels (the game writes the sun term to y).
//
// RT sun shadows: when Luma binds the ray traced visibility at t3 (RTAO on, "RT Sun Shadows" not Off, traced this
// frame), LumaSettings.GameSettings.RTSunMode picks how it replaces the shadow map term:
//   1 = cloud * rt, 2 = cloud * pcf * rt. Without the texture (or mode 0) this is the game's shader.

#include "GameCBuffers.hlsl"
#include "../../Includes/Common.hlsl"

cbuffer PerInstanceCB : register(b2)
{
  float4 cb_parallellightslicefars : packoffset(c0);
  // float4x4 cb_shadowprojections[5], as the registers the game's code indexes: cascade c is c1 + 4c .. c4 + 4c and
  // light space = x * [0] + y * [1] + z * [2] + [3]
  float4 cb_shadowprojections[20] : packoffset(c1);
  float3 cb_atm_sky_sundirection : packoffset(c21);
  float cb_env_atm_sky_cloudaltitude : packoffset(c21.w);
  float cb_shadow_bias : packoffset(c22);
  float cb_screenshadows_stepsize : packoffset(c22.y);
  float cb_modeldepthhack : packoffset(c22.z);
  float cb_env_atm_sky_cloudtilingy : packoffset(c22.w);
  float cb_env_atm_sky_cloudtilingx : packoffset(c23);
  float cb_env_atm_sky_cloudspeedy : packoffset(c23.y);
  float cb_env_atm_sky_cloudspeedx : packoffset(c23.z);
  float cb_env_atm_sky_cloudrotation : packoffset(c23.w);
  float cb_env_atm_sky_clouddensity : packoffset(c24);
}

cbuffer PerViewCB : register(b1)
{
  float4 cb_alwaystweak : packoffset(c0);
  float4 cb_viewrandom : packoffset(c1);
  float4 cb_viewprojectionmatrix[4] : packoffset(c2);
  float4 cb_viewmatrix[4] : packoffset(c6);
  float4 cb_subpixeloffset : packoffset(c10);
  float4 cb_projectionmatrix[4] : packoffset(c11);
  float4 cb_previousviewprojectionmatrix[4] : packoffset(c15);
  float4 cb_previousviewmatrix[4] : packoffset(c19);
  float4 cb_previousprojectionmatrix[4] : packoffset(c23);
  float4 cb_mousecursorposition : packoffset(c27);
  float4 cb_mousebuttonsdown : packoffset(c28);
  float4 cb_jittervectors : packoffset(c29);
  float4 cb_inverseviewprojectionmatrix[4] : packoffset(c30);
  float4 cb_inverseviewmatrix[4] : packoffset(c34);
  float4 cb_inverseprojectionmatrix[4] : packoffset(c38);
  float4 cb_globalviewinfos : packoffset(c42);
  float3 cb_wscamforwarddir : packoffset(c43);
  uint cb_alwaysone : packoffset(c43.w);
  float3 cb_wscamupdir : packoffset(c44);
  uint cb_usecompressedhdrbuffers : packoffset(c44.w);
  float3 cb_wscampos : packoffset(c45);
  float cb_time : packoffset(c45.w);
  float3 cb_wscamleftdir : packoffset(c46);
  float cb_systime : packoffset(c46.w);
  float2 cb_jitterrelativetopreviousframe : packoffset(c47);
  float2 cb_worldtime : packoffset(c47.z);
  float2 cb_shadowmapatlasslicedimensions : packoffset(c48);
  float2 cb_resolutionscale : packoffset(c48.z);
  float2 cb_parallelshadowmapslicedimensions : packoffset(c49);
  float cb_framenumber : packoffset(c49.z);
  uint cb_alwayszero : packoffset(c49.w);
}

SamplerComparisonState smp_shadowhardwarepcfsampler_s : register(s0);
SamplerState smp_bilinearsampler_s : register(s1);
SamplerState smp_pointclamp_s : register(s2);
Texture2DArray<float4> ro_lightdepthcomparemap : register(t0);
Texture2D<float4> ro_env_atm_sky_cloudshadow : register(t1);
Texture2D<float4> ro_screenshadows_depthbuffer : register(t2);
// Luma: ray traced sun visibility (1 = lit), same size as the game's full resolution textures
Texture2D<float> rtsun_visibility : register(t3);

float3 ScreenShadowsWorldPos(float2 uv)
{
  float4 h = (1.0 - uv.y * 2.0) * cb_inverseviewprojectionmatrix[1];
  h = cb_inverseviewprojectionmatrix[0] * (uv.x * 2.0 - 1.0) + h;
  float depth = ro_screenshadows_depthbuffer.SampleLevel(smp_pointclamp_s, uv * cb_resolutionscale, 0).x;
  depth *= cb_modeldepthhack;
  h = cb_inverseviewprojectionmatrix[2] * depth + h;
  h += cb_inverseviewprojectionmatrix[3];
  return h.xyz / h.w;
}

// Cascaded shadow map, 4 tap tent PCF (3x3 texels)
float ScreenShadowsPCF(float3 world)
{
  float view_z = dot(float4(world, 1.0), float4(cb_viewmatrix[0].z, cb_viewmatrix[1].z, cb_viewmatrix[2].z, cb_viewmatrix[3].z));
  float4 beyond = (-view_z >= cb_parallellightslicefars) ? 1.0 : 0.0;
  uint cascade = (uint)(beyond.x + beyond.y + beyond.z + beyond.w);
  uint r = cascade << 2;
  float3 ls = world.y * cb_shadowprojections[r + 1].xyz;
  ls = cb_shadowprojections[r + 0].xyz * world.x + ls;
  ls = cb_shadowprojections[r + 2].xyz * world.z + ls;
  ls += cb_shadowprojections[r + 3].xyz;
  float2 shadow_uv = ls.xy * cb_parallelshadowmapslicedimensions + 0.5;
  float reference = ls.z - cb_shadow_bias;
  float slice = (float)cascade;

  float2 base = floor(shadow_uv);
  float2 f = shadow_uv - base;
  base -= 0.5;
  float2 w0 = 3.0 - 2.0 * f;
  float2 w1 = 2.0 * f + 1.0;
  float2 o0 = (2.0 - f) / w0 - 1.0;
  float2 o1 = f / w1 + 1.0;
  float2 ts = 1.0 / cb_parallelshadowmapslicedimensions;
  float2 b = base * ts;
  float s00 = ro_lightdepthcomparemap.SampleCmpLevelZero(smp_shadowhardwarepcfsampler_s, float3(b + ts * float2(o0.x, o0.y), slice), reference).x;
  float s10 = ro_lightdepthcomparemap.SampleCmpLevelZero(smp_shadowhardwarepcfsampler_s, float3(b + ts * float2(o1.x, o0.y), slice), reference).x;
  float s01 = ro_lightdepthcomparemap.SampleCmpLevelZero(smp_shadowhardwarepcfsampler_s, float3(b + ts * float2(o0.x, o1.y), slice), reference).x;
  float s11 = ro_lightdepthcomparemap.SampleCmpLevelZero(smp_shadowhardwarepcfsampler_s, float3(b + ts * float2(o1.x, o1.y), slice), reference).x;
  float sum = w0.x * w0.y * s00 + w1.x * w0.y * s10;
  sum = w0.x * w1.y * s01 + sum;
  sum = w1.x * w1.y * s11 + sum;
  return saturate(sum * 0.0625);
}

// Cloud layer shadow: the ray toward the sun hits a sphere at the cloud altitude (km), where the cloud texture is read
float ScreenShadowsClouds(float3 world)
{
  float3 p = world * 0.001 + float3(0.0, 0.0, cb_env_atm_sky_cloudaltitude);
  float2 start = world.xy * 0.001;
  float b = dot(cb_atm_sky_sundirection, p);
  float radius = cb_env_atm_sky_cloudaltitude + 2.0;
  float c = dot(p, p) - radius * radius;
  float s = sqrt(max(b * b - c, 0.0));
  float t_far = s - b;
  float t_near = max(-s - b, 0.0);
  float2 hit_near = cb_atm_sky_sundirection.xy * t_near + start;
  float2 hit_far = cb_atm_sky_sundirection.xy * t_far + start;
  float2 hit = (t_far > 0.0) ? hit_far : hit_near;
  float angle = cb_env_atm_sky_cloudrotation * cb_worldtime.x;
  float sn, cs;
  sincos(angle, sn, cs);
  float2 rotated = float2(cs * hit.x - sn * hit.y, sn * hit.x + cs * hit.y);
  float2 cloud_uv = cb_worldtime.x * float2(cb_env_atm_sky_cloudspeedx, cb_env_atm_sky_cloudspeedy) + rotated;
  cloud_uv *= float2(cb_env_atm_sky_cloudtilingx, cb_env_atm_sky_cloudtilingy);
  float density = ro_env_atm_sky_cloudshadow.SampleLevel(smp_bilinearsampler_s, cloud_uv, 0).x * cb_env_atm_sky_clouddensity;
  return min(exp(-density), 1.0);
}

float4 ScreenShadows(float2 uv, float3 world)
{
  float cloud = ScreenShadowsClouds(world);
  uint rt_width = 0, rt_height = 0;
  rtsun_visibility.GetDimensions(rt_width, rt_height);
  uint mode = LumaSettings.GameSettings.RTSunMode;
  bool use_rt = rt_width != 0 && (mode == 1 || mode == 2);
  float rt = 1.0;
  float pcf = 1.0;
  [branch]
  if (use_rt)
    rt = saturate(rtsun_visibility.SampleLevel(smp_pointclamp_s, uv * cb_resolutionscale, 0));
  [branch]
  if (!use_rt || mode == 2)
    pcf = ScreenShadowsPCF(world);
  return (cloud * pcf * rt).xxxx;
}
