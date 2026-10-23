// RT sun shadows denoiser: the RTAO denoiser (rtao_filter.hlsl, itself ported from f4-mods by Sineaggi)
// with its own history: a 3x3 cross-bilateral on view depth plus a plane term, then reprojected temporal
// accumulation into a double-buffered packed history with a neighbourhood clamp. Output: visibility, 1 = lit.
#include "rtsun_common.hlsli"

RWTexture2D<float> VisRaw : register(u0);
RWTexture2D<unorm float> VisFiltered : register(u1);
// Packed: x = asuint(view depth), y = (vis16 << 8) | age8
RWTexture2D<uint2> HistPrev : register(u2);
RWTexture2D<uint2> HistCur : register(u3);

static const uint kDebugHistory = 3; // 1 = history kept, 0 = rejected

static const float kRelDepthTol = 0.02;
static const float kPlaneTolRel = 0.002;
static const float kDisoccludeTol = 0.03;
static const float kConvergedAge = 32.0;
static const float kSpatialMixFloor = 0.33;
static const uint kAgeMax = 255;
static const float kClampSoft = 0.5;

uint2 PackHist(float v, float view_z, uint age)
{
    return uint2(asuint(view_z), (uint(saturate(v) * 65535.0 + 0.5) << 8) | min(age, kAgeMax));
}

void UnpackHist(uint2 h, out float v, out float view_z, out uint age)
{
    view_z = asfloat(h.x);
    v = float(h.y >> 8) / 65535.0;
    age = h.y & 0xFFu;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= Size))
        return;
    float raw = VisRaw[px];
    float d = LoadDepth(px);
    if (IsSky(d) || FilterMode == kFilterOff)
    {
        VisFiltered[px] = raw;
        HistCur[px] = PackHist(raw, 0.0, 0);
        return;
    }
    float3 p0 = WorldPosAt(px, d);
    float z0 = ViewZ(p0);
    uint2 pxR = uint2(min(px.x + 1, Size.x - 1), px.y);
    uint2 pxD = uint2(px.x, min(px.y + 1, Size.y - 1));
    float3 n = cross(WorldPos(pxD) - p0, WorldPos(pxR) - p0);
    float nl = length(n);
    n = nl > 1e-12 ? n / nl : float3(0, 0, 1);

    float sum = 0.0, wsum = 0.0;
    float mn = 1.0, mx = 0.0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx)
    {
        int2 qi = int2(px) + int2(dx, dy);
        if (qi.x < 0 || qi.y < 0 || qi.x >= int(Size.x) || qi.y >= int(Size.y))
            continue;
        uint2 q = uint2(qi);
        float dq = LoadDepth(q);
        if (IsSky(dq))
            continue;
        float3 pq = WorldPosAt(q, dq);
        float dz = abs(ViewZ(pq) - z0) / max(abs(z0), 1e-3);
        float dp = abs(dot(pq - p0, n)) / max(abs(z0) * kPlaneTolRel, 1e-4);
        float w = exp(-dz / kRelDepthTol) * exp(-dp);
        float v = VisRaw[q];
        sum += w * v;
        wsum += w;
        if (w > 0.0)
        {
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    float mean = wsum > 0.0 ? sum / wsum : raw;

    bool rejected = true;
    float prev_v = 0.0;
    uint age = 0;
    if (FilterMode == kFilterTemporal && HistValid != 0)
    {
        float4 pc = mul(float4(p0, 1.0), PrevViewProj);
        if (pc.w > 1e-4)
        {
            float2 ndc = pc.xy / pc.w;
            float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv >= 0.0) && all(uv < 1.0))
            {
                uint2 pp = uint2(uv * float2(Size));
                float hv, hz;
                uint ha;
                UnpackHist(HistPrev[pp], hv, hz, ha);
                float z_prev_cam = dot(p0 - PrevCamPos, PrevCamFwd);
                if (ha > 0u && abs(hz - z_prev_cam) <= kDisoccludeTol * max(abs(z_prev_cam), 1e-3))
                {
                    float lo = mn - ClampK, hi = mx + ClampK;
                    float dist = max(max(lo - hv, hv - hi), 0.0);
                    float pull = saturate(dist / kClampSoft);
                    prev_v = lerp(hv, clamp(hv, lo, hi), pull);
                    age = (uint)max(1.0, lerp((float)ha, 1.0, pull));
                    rejected = false;
                }
            }
        }
    }

    float spatial_mix = rejected ? 1.0 : max(kSpatialMixFloor, saturate(1.0 - (float)age / kConvergedAge));
    float cur = lerp(raw, mean, spatial_mix);
    float out_v;
    uint new_age;
    if (!rejected)
    {
        float a = max(1.0 / ((float)age + 1.0), AlphaMin);
        out_v = lerp(prev_v, cur, a);
        new_age = min(age + 1u, kAgeMax);
    }
    else
    {
        out_v = cur;
        new_age = 1u;
    }
    HistCur[px] = PackHist(out_v, z0, new_age);
    VisFiltered[px] = DebugMode == kDebugHistory ? (rejected ? 0.0 : 1.0) : saturate(out_v);
}
