// RT sun shadows trace: per pixel, RayCount inline ray queries (DXR 1.1) from the world position reconstructed from
// depth toward the directional light, jittered inside the light's disc (soft penumbra, ConeSampleDisk from Sineaggi's f4-mods RtShadows).
// Writes raw visibility (1 = lit) for rtsun_filter.hlsl. Same TLAS, bias and cutout handling as rtao_trace.hlsl
// (texture exact masks through rtao_cutout.hlsli, the stochastic opacity / fast cutouts for cutouts without a mask).
// Always full resolution (RTAO Quality doesn't apply: a hard shadow edge doesn't survive a half resolution upsample).
#include "rtsun_common.hlsli"
#include "rtao_cutout.hlsli"

RaytracingAccelerationStructure Scene : register(t0);
RWTexture2D<float> VisRaw : register(u0);

static const uint kDebugRaw = 1;       // raw visibility (filter off), see RTSun::SunSettings::debug_mode
static const uint kDebugNdotL = 2;     // geometric N.L from the depth normal (no rays)

uint Hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float2 Random2(uint2 px, uint i)
{
    uint h = Hash(px.x + Hash(px.y + Hash(Frame * 64 + i + 0x2545F491u)));
    uint h2 = Hash(h);
    return float2(h & 0xFFFFFF, h2 & 0xFFFFFF) / 16777216.0;
}

// A direction inside the cone of half angle atan(tan_radius) around L, uniform over the disc (sqrt(u) radius). The
// disc form keeps its precision at the sun's 0.27 degree radius, where 1 - cos^2 forms would cancel.
float3 ConeSampleDisk(float3 L, float tan_radius, float2 u)
{
    float3 t = abs(L.z) < 0.999 ? normalize(cross(L, float3(0, 0, 1))) : float3(1, 0, 0);
    float3 b = cross(L, t);
    float r = sqrt(u.x) * tan_radius, phi = 6.28318530718 * u.y;
    return normalize(L + t * (r * cos(phi)) + b * (r * sin(phi)));
}

// Picks the shorter one-sided difference on each axis so the normal doesn't smear across depth edges
float3 DepthNormal(uint2 px, float3 p0)
{
    uint2 l = uint2(px.x > 0 ? px.x - 1 : px.x, px.y);
    uint2 r = uint2(min(px.x + 1, Size.x - 1), px.y);
    uint2 u = uint2(px.x, px.y > 0 ? px.y - 1 : px.y);
    uint2 d = uint2(px.x, min(px.y + 1, Size.y - 1));
    float3 dl = p0 - WorldPos(l), dr = WorldPos(r) - p0;
    float3 du = p0 - WorldPos(u), dd = WorldPos(d) - p0;
    float3 dx = dot(dl, dl) < dot(dr, dr) ? dl : dr;
    float3 dy = dot(du, du) < dot(dd, dd) ? du : dd;
    float3 n = cross(dy, dx);
    float len = length(n);
    n = len > 1e-12 ? n / len : -CamFwd;
    return dot(n, CamPos - p0) < 0.0 ? -n : n;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= Size))
        return;
    float d = LoadDepth(px);
    if (IsSky(d))
    {
        VisRaw[px] = 1.0;
        return;
    }
    float3 p0 = WorldPosAt(px, d);
    float3 n = DepthNormal(px, p0);
    float3 L = normalize(LightDir);
    float n_dot_l = dot(n, L);
    if (DebugMode == kDebugNdotL)
    {
        VisRaw[px] = saturate(n_dot_l);
        return;
    }
    // Facing away from the light (geometric normal): in its own shadow, like the shadow map says. A small tolerance
    // keeps grazing surfaces, whose reconstructed normal is noisy, on the traced path.
    if (n_dot_l < -0.05)
    {
        VisRaw[px] = 0.0;
        return;
    }

    float view_z = max(ViewZ(p0), 0.0);
    // Same offset as the AO trace: grows with distance to stay above the depth reconstruction error
    float3 origin = p0 + n * (Bias + view_z * 0.002);

    // Cutouts: exactly as rtao_trace.hlsl. A candidate cutout triangle is tested against its material's alpha mask
    // (texture exact); without a mask, it stops the ray with probability CutoutOpacity (per crossing, or per ray with
    // fast cutouts). With masks off, opacity 1 forces everything opaque and 0 leaves cutouts out.
    const uint ray_flags = (CutoutOpacity >= 1.0 && CutoutExact == 0) ? RAY_FLAG_FORCE_OPAQUE : RAY_FLAG_NONE;
    const uint ray_mask = (CutoutOpacity <= 0.0 && CutoutExact == 0) ? (0xFF & ~kInstanceMaskCutout) : 0xFF;
    uint rays = max(RayCount, 1u);
    float lit = 0.0;
    const float coin_offset = float(Hash(px.x * 0x632BE5ABu + Hash(px.y + Hash(Frame + 0x165667B1u))) & 0xFFFF) / 65536.0;
    for (uint i = 0; i < rays; i++)
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = ConeTan > 0.0 ? ConeSampleDisk(L, ConeTan, Random2(px, i)) : L;
        ray.TMin = 0.0;
        ray.TMax = TMax;
        uint flags_i = ray_flags, mask_i = ray_mask;
        int fallback_coin = -1;
        if (CutoutMode != 0 && CutoutOpacity > 0.0 && CutoutOpacity < 1.0)
        {
            const float u = frac((float(i) + coin_offset) / float(rays));
            const float p = CutoutMode == 2 ? 1.0 - (1.0 - CutoutOpacity) * (1.0 - CutoutOpacity) : CutoutOpacity;
            if (CutoutExact != 0)
                fallback_coin = u < p ? 1 : 0;
            else if (u < p)
                flags_i |= RAY_FLAG_FORCE_OPAQUE;
            else
                mask_i &= ~kInstanceMaskCutout;
        }
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(Scene, flags_i, mask_i, ray);
        const uint seed = Hash(px.x * 0x9E3779B9u + Hash(px.y + Hash(Frame * 64 + i + 0x68E31DA4u)));
        while (q.Proceed())
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && (q.CandidateInstanceID() & kInstanceIdCutoutBit) != 0)
            {
                int solid = CutoutAlphaTest(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics());
                if (solid < 0 && fallback_coin >= 0)
                    solid = fallback_coin;
                if (solid < 0)
                {
                    uint h = Hash(seed ^ Hash(q.CandidateInstanceIndex() * 0x8DA6B343u + q.CandidatePrimitiveIndex()));
                    solid = float(h & 0xFFFFFF) / 16777216.0 < CutoutOpacity ? 1 : 0;
                }
                if (solid > 0)
                    q.CommitNonOpaqueTriangleHit();
            }
        }
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            lit += 1.0;
    }
    CutoutFlushCounters();
    VisRaw[px] = lit / float(rays);
}
