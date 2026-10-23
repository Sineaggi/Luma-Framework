// RTAO denoiser, ported from the user's Fallout 4 RtShadows denoiser (f4-mods/RtShadows/shadow_filter.hlsl) and
// reduced to one AO channel: a 3x3 cross-bilateral on view depth plus a plane term from depth derivatives,
// then reprojected temporal accumulation into a double-buffered packed history with a neighbourhood clamp.
#include "rtao_common.hlsli"

RWTexture2D<float> AoRaw : register(u0);
RWTexture2D<unorm float> AoFiltered : register(u1);
// Double-buffered: this frame reads HistPrev at the reprojected texel and writes HistCur at its own.
// Packed: x = asuint(view depth), y = (ao16 << 8) | age8.
RWTexture2D<uint2> HistPrev : register(u2);
RWTexture2D<uint2> HistCur : register(u3);

static const float kRelDepthTol = 0.02;     // bilateral depth gate: 2 % of view depth
static const float kPlaneTolRel = 0.002;    // plane term: 0.2 % of view depth off the local plane
static const float kDisoccludeTol = 0.03;   // previous depth within 3 % relative
static const float kConvergedAge = 32.0;    // spatial weight fades out by this age...
static const float kSpatialMixFloor = 0.33; // ...but never below this
static const uint kAgeMax = 255;
static const float kClampSoft = 0.5;        // history distance outside the box at which the pull is total

uint2 PackHist(float ao, float view_z, uint age)
{
    return uint2(asuint(view_z), (uint(saturate(ao) * 65535.0 + 0.5) << 8) | min(age, kAgeMax));
}

void UnpackHist(uint2 h, out float ao, out float view_z, out uint age)
{
    view_z = asfloat(h.x);
    ao = float(h.y >> 8) / 65535.0;
    age = h.y & 0xFFu;
}

// The 3x3 raw texels around each pixel of the 8x8 group, with the world position of the pixel each was traced at:
// loaded once per group (10x10 texels at full resolution, 6x6 at half) instead of 9 times per pixel.
// w: 0 usable, 1 sky, 2 outside the image.
static const uint kTileMax = 10;
groupshared float4 g_pos[kTileMax * kTileMax];
groupshared float g_raw[kTileMax * kTileMax];

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint gi : SV_GroupIndex)
{
    const bool half_res = TraceHalfRes != 0;
    const uint2 trace_size = TraceSize();
    const uint tile_w = half_res ? 6 : 10;
    const int2 tile_origin = int2(group.xy * (half_res ? 4 : 8)) - 1;
    for (uint t = gi; t < tile_w * tile_w; t += 64)
    {
        const int2 ci = tile_origin + int2(t % tile_w, t / tile_w);
        const bool inside = all(ci >= 0) && all(ci < int2(trace_size));
        // Outside texels take the clamped one's position (the normal below relies on that at the image border)
        const uint2 c = uint2(clamp(ci, int2(0, 0), int2(trace_size) - 1));
        const uint2 q = TracedPixel(c);
        const float dq = LoadDepth(q);
        g_pos[t] = float4(WorldPosAt(q, dq), inside ? (IsSky(dq) ? 1.0 : 0.0) : 2.0);
        g_raw[t] = AoRaw[c];
    }
    GroupMemoryBarrierWithGroupSync();

    uint2 px = id.xy;
    if (any(px >= Size))
        return;
    const uint2 own_cell = half_res ? px / 2 : px;
    const int2 local = int2(own_cell) - tile_origin; // 1..8 (1..4 at half resolution)
    float raw = g_raw[local.y * tile_w + local.x];
    float d = LoadDepth(px);
    if (IsSky(d) || FilterMode == kFilterOff)
    {
        AoFiltered[px] = raw;
        HistCur[px] = PackHist(raw, 0.0, 0);
        return;
    }
    float3 p0, n;
    if (half_res)
    {
        p0 = WorldPosAt(px, d);
        uint2 pxR = uint2(min(px.x + 1, Size.x - 1), px.y);
        uint2 pxD = uint2(px.x, min(px.y + 1, Size.y - 1));
        n = cross(WorldPos(pxD) - p0, WorldPos(pxR) - p0);
    }
    else
    {
        p0 = g_pos[local.y * tile_w + local.x].xyz;
        n = cross(g_pos[(local.y + 1) * tile_w + local.x].xyz - p0, g_pos[local.y * tile_w + local.x + 1].xyz - p0);
    }
    float z0 = ViewZ(p0);
    float nl = length(n);
    n = nl > 1e-12 ? n / nl : float3(0, 0, 1);

    // 3x3 cross-bilateral over the raw texels, plus the min/max box for the neighbourhood clamp. At half resolution
    // the texels are 2x2 blocks, each traced at one pixel (TracedPixel): the same weights on that pixel's position,
    // times a spatial falloff, make it a depth aware upsample too.
    float sum = 0.0, wsum = 0.0;
    float mn = 1.0, mx = 0.0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx)
    {
        const uint t = uint(local.y + dy) * tile_w + uint(local.x + dx);
        const float4 e = g_pos[t];
        if (e.w != 0.0)
            continue;
        float3 pq = e.xyz;
        float dz = abs(ViewZ(pq) - z0) / max(abs(z0), 1e-3);
        float dp = abs(dot(pq - p0, n)) / max(abs(z0) * kPlaneTolRel, 1e-4);
        float w = exp(-dz / kRelDepthTol) * exp(-dp);
        if (half_res)
        {
            float2 o = float2(TracedPixel(uint2(int2(own_cell) + int2(dx, dy)))) - float2(px);
            w *= exp(-dot(o, o) * (1.0 / 4.5)); // sigma 1.5 px
        }
        float v = g_raw[t];
        sum += w * v;
        wsum += w;
        if (w > 0.0)
        {
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    float mean = wsum > 1e-6 ? sum / wsum : raw;
    // At half resolution only one pixel of the block has its own sample this frame: the others take the upsample
    if (half_res && any(TracedPixel(own_cell) != px))
        raw = mean;

    bool rejected = true;
    float prev_ao = 0.0;
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
                // Disocclusion: the depth stored last frame at that texel vs this point's depth in last frame's camera
                float z_prev_cam = dot(p0 - PrevCamPos, PrevCamFwd);
                if (ha > 0u && abs(hz - z_prev_cam) <= kDisoccludeTol * max(abs(z_prev_cam), 1e-3))
                {
                    // Neighbourhood clamp with a pull proportional to how far the history is outside the box;
                    // the pull also resets the age, so a corrected ghost converges in a few frames.
                    float lo = mn - ClampK, hi = mx + ClampK;
                    float dist = max(max(lo - hv, hv - hi), 0.0);
                    float pull = saturate(dist / kClampSoft);
                    prev_ao = lerp(hv, clamp(hv, lo, hi), pull);
                    age = (uint)max(1.0, lerp((float)ha, 1.0, pull));
                    rejected = false;
                }
            }
        }
    }

    float spatial_mix = rejected ? 1.0 : max(kSpatialMixFloor, saturate(1.0 - (float)age / kConvergedAge));
    float cur = lerp(raw, mean, spatial_mix);
    float out_ao;
    uint new_age;
    if (!rejected)
    {
        float a = max(1.0 / ((float)age + 1.0), AlphaMin);
        out_ao = lerp(prev_ao, cur, a);
        new_age = min(age + 1u, kAgeMax);
    }
    else
    {
        out_ao = cur;
        new_age = 1u;
    }
    HistCur[px] = PackHist(out_ao, z0, new_age);
    // Strength: a power curve on the output only (the history keeps the linear value)
    AoFiltered[px] = DebugMode == 2 ? (rejected ? 0.0 : 1.0) : pow(saturate(out_ao), max(Strength, 0.01));
}
