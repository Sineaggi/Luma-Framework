// RTAO trace: one cosine distributed hemisphere of inline ray queries (DXR 1.1) per pixel, from the world
// position reconstructed from depth. Writes raw visibility (1 = unoccluded) for rtao_filter.hlsl.
// The normal comes from depth derivatives (the game's G-buffer normal decode is not wired up yet).
#include "rtao_common.hlsli"

RaytracingAccelerationStructure Scene : register(t0);
RWTexture2D<float> AoRaw : register(u0);

uint Hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float2 Random2(uint2 px, uint i)
{
    uint h = Hash(px.x + Hash(px.y + Hash(Frame * 64 + i)));
    uint h2 = Hash(h);
    return float2(h & 0xFFFFFF, h2 & 0xFFFFFF) / 16777216.0;
}

float3 CosineHemisphere(float3 n, float2 u)
{
    float3 t = abs(n.z) < 0.999 ? normalize(cross(n, float3(0, 0, 1))) : float3(1, 0, 0);
    float3 b = cross(n, t);
    float r = sqrt(u.x);
    float phi = 6.28318530718 * u.y;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(saturate(1.0 - u.x)));
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
    const uint2 cell = id.xy; // Raw texel (one per pixel, or per 2x2 block at half resolution)
    if (any(cell >= TraceSize()))
        return;
    const uint2 px = TracedPixel(cell);
    float d = LoadDepth(px);
    if (IsSky(d))
    {
        AoRaw[cell] = 1.0;
        return;
    }
    float3 p0 = WorldPosAt(px, d);

    if (DebugMode == 3)
    {
        // BLAS coverage: a camera ray through the pixel should hit the captured geometry where the depth says.
        // 1 = hit within 2 % of the depth distance, 0.5 = no hit (geometry missing from the TLAS),
        // 0 = hit clearly in front (geometry in the TLAS that the depth doesn't have: misplaced or extra).
        float3 to_p = p0 - CamPos;
        float dist = length(to_p);
        RayDesc primary;
        primary.Origin = CamPos;
        primary.Direction = to_p / max(dist, 1e-6);
        primary.TMin = 0.0;
        primary.TMax = dist * 1.5 + 1.0;
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> pq;
        pq.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, primary);
        pq.Proceed();
        float coverage = 0.5;
        if (pq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            float t = pq.CommittedRayT();
            float tolerance = dist * 0.02 + 0.02;
            coverage = abs(t - dist) <= tolerance ? 1.0 : (t < dist ? 0.0 : 0.5);
        }
        AoRaw[cell] = coverage;
        return;
    }

    float3 n = DepthNormal(px, p0);
    float view_z = max(ViewZ(p0), 0.0);
    // Offset grows with distance to stay above the depth reconstruction error
    float3 origin = p0 + n * (Bias + view_z * 0.002);

    // Cutouts (alpha tested instances, forced non opaque in the TLAS; every other instance is forced opaque): at
    // opacity 1 the ray forces everything opaque (no candidates, the plain opaque trace), at 0 its mask leaves them out.
    // In between, each (ray, cutout triangle) crossing stops the ray with that probability: a stochastic stand-in for
    // the texture's coverage, which the temporal denoiser averages like the rest of the ray noise.
    const uint ray_flags = CutoutOpacity >= 1.0 ? RAY_FLAG_FORCE_OPAQUE : RAY_FLAG_NONE;
    const uint ray_mask = CutoutOpacity <= 0.0 ? (0xFF & ~kInstanceMaskCutout) : 0xFF;
    float occlusion = 0.0;
    uint rays = max(RayCount, 1u);
    const float coin_offset = float(Hash(px.x * 0x632BE5ABu + Hash(px.y + Hash(Frame + 0x27d4eb2du))) & 0xFFFF) / 65536.0;
    for (uint i = 0; i < rays; i++)
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = CosineHemisphere(n, Random2(px, i));
        ray.TMin = 0.0;
        ray.TMax = Radius;
        uint flags_i = ray_flags, mask_i = ray_mask;
        if (CutoutMode != 0 && CutoutOpacity > 0.0 && CutoutOpacity < 1.0)
        {
            // Per ray coin (stratified over the pixel's rays): cutouts are solid for this ray, or not there at all
            const float u = frac((float(i) + coin_offset) / float(rays));
            const float p = CutoutMode == 2 ? 1.0 - (1.0 - CutoutOpacity) * (1.0 - CutoutOpacity) : CutoutOpacity;
            if (u < p)
                flags_i |= RAY_FLAG_FORCE_OPAQUE;
            else
                mask_i &= ~kInstanceMaskCutout;
        }
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(Scene, flags_i, mask_i, ray);
        const uint seed = Hash(px.x * 0x9E3779B9u + Hash(px.y + Hash(Frame * 64 + i + 0x5bd1e995u)));
        while (q.Proceed())
        {
            // Only cutout instances produce candidates
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && (q.CandidateInstanceID() & kInstanceIdCutoutBit) != 0)
            {
                uint h = Hash(seed ^ Hash(q.CandidateInstanceIndex() * 0x8DA6B343u + q.CandidatePrimitiveIndex()));
                if (float(h & 0xFFFFFF) / 16777216.0 < CutoutOpacity)
                    q.CommitNonOpaqueTriangleHit(); // Ends the search (ACCEPT_FIRST_HIT_AND_END_SEARCH)
            }
        }
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            // Soft falloff towards the radius so the AO has no hard edge at Radius
            float t = q.CommittedRayT() / Radius;
            occlusion += 1.0 - t * t;
        }
    }
    float ao = 1.0 - occlusion / float(rays);
    if (DebugMode == 1)
        ao = n.y * 0.5 + 0.5; // Reconstructed normal (Y)
    AoRaw[cell] = ao;
}
