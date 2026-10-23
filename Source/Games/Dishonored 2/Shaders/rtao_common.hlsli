// Shared by rtao_trace.hlsl and rtao_filter.hlsl (D3D12 side of RTAO, compiled offline with DXC, see
// compile_rtao_shaders.bat). Must match RTAO::PassConstants in rtao_pass.h.
//
// Matrix convention: row vectors, the same memory layout the game's PerViewCB uses:
//   clip = mul(float4(world, 1), ViewProj), world_h = mul(float4(ndc, 1), InvViewProj).

cbuffer PassConstants : register(b0)
{
    row_major float4x4 InvViewProj;  // this frame, NDC -> world
    row_major float4x4 PrevViewProj; // previous frame, world -> clip
    float3 CamPos;      float Radius;
    float3 CamFwd;      uint RayCount;
    float3 PrevCamPos;  uint Frame;
    float3 PrevCamFwd;  uint HistValid;
    uint2 Size;         uint2 DepthSize;
    uint DepthPacked;   uint ReversedZ;  uint FilterMode; float AlphaMin;
    float ClampK;       float Bias;      uint DebugMode;  float Strength;
};

Texture2D<float4> DepthTex : register(t1);

static const uint kFilterOff = 0;
static const uint kFilterSpatial = 1;
static const uint kFilterTemporal = 2;

// Device depth at an AO pixel (the depth texture may be a different size)
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
