// Shared by rtsun_trace.hlsl and rtsun_filter.hlsl (D3D12 side of the RT sun shadows, compiled offline with DXC,
// see compile_rtao_shaders.bat). Must match RTSun::SunConstants in rtsun_pass.h. Kept separate from rtao_common.hlsli
// on purpose (own constants), the depth helpers are the same.
//
// Matrix convention: row vectors, the game's PerViewCB memory layout:
//   clip = mul(float4(world, 1), ViewProj), world_h = mul(float4(ndc, 1), InvViewProj).

cbuffer SunConstants : register(b0)
{
    row_major float4x4 InvViewProj;  // this frame, NDC -> world
    row_major float4x4 PrevViewProj; // previous frame, world -> clip
    float3 CamPos;      float TMax;
    float3 CamFwd;      uint RayCount;
    float3 PrevCamPos;  uint Frame;
    float3 PrevCamFwd;  uint HistValid;
    float3 LightDir;    float ConeTan;   // world space, toward the light; tan of the light's angular radius
    uint2 Size;         uint2 DepthSize;
    uint DepthPacked;   uint ReversedZ;  uint FilterMode;  float AlphaMin;
    float ClampK;       float Bias;      uint DebugMode;   float CutoutOpacity;
    uint CutoutMode;    uint CutoutExact; uint CastersOnly; uint SunPad;  // As RTAO's (rtao_common.hlsli): fast cutouts, texture exact masks; CastersOnly: camera only objects don't block
    // The game's shadow map decides whether a camera only object blocks (rtsun_trace.hlsl GameShadowConfirms)
    uint ShadowConsult; uint ShadowPacked; uint ShadowReversed; float ShadowTolerance; // Tolerance in world units
    uint2 ShadowSize;   uint ShadowCascades; uint ShadowPad;                          // One cascade's size, how many are stacked
    float ShadowBiasConstant; float ShadowBiasSlope; uint2 BiasPad;                  // The game's shadow rasterizer bias (depth units, slope factor)
};

// Must match rtao_accel.h
static const uint kInstanceMaskCutout = 0x02;
static const uint kInstanceMaskCameraOnly = 0x04;
static const uint kInstanceMaskCameraOnlyCutout = 0x08;
static const uint kInstanceIdCutoutBit = 1u << 23;

static const uint kFilterOff = 0;
static const uint kFilterSpatial = 1;
static const uint kFilterTemporal = 2;

Texture2D<float4> DepthTex : register(t1);

// Device depth at an output pixel (the depth texture may be a different size)
float LoadDepth(uint2 px)
{
    int2 p = int2((float2(px) + 0.5) * float2(DepthSize) / float2(Size));
    p = clamp(p, int2(0, 0), int2(DepthSize) - 1);
    float4 v = DepthTex.Load(int3(p, 0));
    uint4 b = uint4(round(saturate(v) * 255.0));
    float unpacked = asfloat(b.x | (b.y << 8) | (b.z << 16) | (b.w << 24));
    return DepthPacked != 0 ? unpacked : v.x;
}

bool IsSky(float d)
{
    return ReversedZ != 0 ? (d <= 0.0) : (d >= 1.0);
}

float3 WorldPosAt(uint2 px, float d)
{
    float2 uv = (float2(px) + 0.5) / float2(Size);
    float4 h = mul(float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0), InvViewProj);
    return h.xyz / h.w;
}

float3 WorldPos(uint2 px)
{
    return WorldPosAt(px, LoadDepth(px));
}

float ViewZ(float3 p)
{
    return dot(p - CamPos, CamFwd);
}
