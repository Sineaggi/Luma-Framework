// RT sun shadows trace: per pixel, RayCount inline ray queries (DXR 1.1) from the world position reconstructed from
// depth toward the directional light, jittered inside the light's disc (soft penumbra, ConeSampleDisk from Sineaggi's f4-mods RtShadows).
// Writes raw visibility (1 = lit) for rtsun_filter.hlsl. Same TLAS, bias and cutout handling as rtao_trace.hlsl
// (texture exact masks through rtao_cutout.hlsli, the stochastic opacity / fast cutouts for cutouts without a mask).
// Always full resolution (RTAO Quality doesn't apply: a hard shadow edge doesn't survive a half resolution upsample).
#include "rtsun_common.hlsli"
#include "rtao_cutout.hlsli"

RaytracingAccelerationStructure Scene : register(t0);
RWTexture2D<float> VisRaw : register(u0);
// The game's sun shadow map (copied at the screen shadows draw, Luma_RTSunShadowCopy.hlsl): ShadowCascades cascades of
// ShadowSize stacked vertically, then one row with their light matrices (16 values each), packed RGBA8 if ShadowPacked
Texture2D<float4> ShadowTex : register(t5);

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

// ---- Debug scene view (RTSun::Output::SceneView / SceneSun): what the TLAS holds, colored by where it was captured.
// The color goes out packed as RGB565 in the 16 bit visibility (Luma_RTAO.hlsl ao_view_ps unpacks it).
static const uint kDebugSceneView = 5; // From the camera, through each pixel
static const float kSkinSelfDistance = 0.08; // A shadow ray ignores a character hit closer than this to its origin (self shadow)
static const uint kDebugBlockerAll = 8; // The same with every instance (the casters only mask off)
static const uint kDebugBlocker = 7;    // One hard shadow ray: TLAS index + 1 of the closest blocker, 0 if lit (16 bit id)
static const uint kDebugSceneSun = 6;  // From the sun, orthographic, a 60 m tall area around the camera, 40 m up the light to 40 m past it
// Must match rtao_accel.h
static const uint kSourceMain = 0x01, kSourceShadow = 0x02, kFlagFrameMesh = 0x04, kFlagMvp = 0x08, kFlagSkinned = 0x10, kFlagCameraOnly = 0x20, kFlagWindow = 0x40;

float LoadShadowValue(uint2 p)
{
    float4 v = ShadowTex.Load(int3(p, 0));
    if (ShadowPacked == 0)
        return v.x;
    uint4 b = uint4(round(saturate(v) * 255.0));
    return asfloat(b.x | (b.y << 8) | (b.z << 16) | (b.w << 24));
}

// Row r of cascade c's light matrix (the game's cb_shadowprojections: light space = x * row0 + y * row1 + z * row2 + row3)
float4 ShadowMatrixRow(uint c, uint r)
{
    uint2 p = uint2(c * 16 + r * 4, ShadowSize.y * ShadowCascades);
    return float4(LoadShadowValue(p), LoadShadowValue(p + uint2(1, 0)), LoadShadowValue(p + uint2(2, 0)), LoadShadowValue(p + uint2(3, 0)));
}

// Does the game's own shadow map agree something blocks the sun at p (a hit on an object the game never drew into it,
// on a ray from "receiver")? The cascade is the one the game's screen shadows use for the receiver (how many of its
// slice fars the receiver's view depth reaches): the cascades hold different geometry, so another one can disagree with
// the game's own shadow. True if that cascade holds a surface at p's light depth or nearer the light (p is on a caster,
// or in the shadow of one), within ShadowTolerance plus two texels' worth, over p's 3x3 texels. A sun ray keeps its
// light space xy, so this is the receiver's texel. Beyond the copied cascades or outside it: true (no say).
bool GameShadowConfirms(float3 receiver, float3 p)
{
    const uint2 extra = uint2(ShadowCascades * 16, ShadowSize.y * ShadowCascades);
    float4 fars = float4(LoadShadowValue(extra), LoadShadowValue(extra + uint2(1, 0)), LoadShadowValue(extra + uint2(2, 0)), LoadShadowValue(extra + uint2(3, 0)));
    float4 view_z_axis = float4(LoadShadowValue(extra + uint2(4, 0)), LoadShadowValue(extra + uint2(5, 0)), LoadShadowValue(extra + uint2(6, 0)), LoadShadowValue(extra + uint2(7, 0)));
    float view_z = dot(float4(receiver, 1.0), view_z_axis);
    uint4 beyond = select(-view_z >= fars, uint4(1, 1, 1, 1), uint4(0, 0, 0, 0));
    uint c = beyond.x + beyond.y + beyond.z + beyond.w;
    [branch]
    if (c < ShadowCascades)
    {
        float4 r0 = ShadowMatrixRow(c, 0), r1 = ShadowMatrixRow(c, 1), r2 = ShadowMatrixRow(c, 2), r3 = ShadowMatrixRow(c, 3);
        float3 ls = p.x * r0.xyz + p.y * r1.xyz + p.z * r2.xyz + r3.xyz;
        if (any(ls.xy <= 0.0) || any(ls.xy >= 1.0) || ls.z <= 0.0 || ls.z >= 1.0)
            return true;
        float depth_per_unit = length(float3(r0.z, r1.z, r2.z));
        float texel_units = 1.0 / (float(ShadowSize.x) * max(length(float3(r0.x, r1.x, r2.x)), 1e-9));
        float tolerance = (ShadowTolerance + 2.0 * texel_units) * depth_per_unit;
        float s = ShadowReversed != 0 ? -1.0 : 1.0; // Compared so that smaller is nearer the light
        int2 t = int2(ls.xy * float2(ShadowSize));
        float nearest = 1e30;
        [unroll]
        for (int dy = -1; dy <= 1; dy++)
        {
            [unroll]
            for (int dx = -1; dx <= 1; dx++)
            {
                int2 q = clamp(t + int2(dx, dy), int2(0, 0), int2(ShadowSize) - 1);
                nearest = min(nearest, s * LoadShadowValue(uint2(q.x, uint(q.y) + c * ShadowSize.y)));
            }
        }
        return nearest <= s * ls.z + tolerance;
    }
    return true;
}

// A non opaque candidate: does it block the sun? A cutout per its mask (or the stochastic fallback: fallback_coin, then
// CutoutOpacity), a character beyond its own surface, and, with "consult", an object only the camera saw (glass, light
// shafts, and real casters drawn only in passes we don't take as casters) only where the game's shadow map agrees.
bool CandidateBlocks(uint instance_id, uint flags, uint instance_index, uint primitive, float2 bary, float t, float3 origin, float3 dir, int fallback_coin, uint seed, bool consult)
{
    if ((instance_id & kInstanceIdCutoutBit) != 0)
    {
        // Textured windows: graded, against a random reference per ray (seed 0: the hard rays, 0.5)
        const float reference = (flags & kFlagWindow) != 0 ? (seed != 0 ? (float(Hash(seed ^ 0x2F6B4A1Du ^ (instance_index * 0x9E3779B9u)) & 0xFFFF) + 0.5) / 65536.0 : 0.5) : -1.0;
        int solid = CutoutAlphaTest(instance_id, primitive, bary, reference);
        if (solid < 0 && fallback_coin >= 0)
            solid = fallback_coin;
        if (solid < 0)
        {
            uint h = Hash(seed ^ Hash(instance_index * 0x8DA6B343u + primitive));
            solid = float(h & 0xFFFFFF) / 16777216.0 < CutoutOpacity ? 1 : 0;
        }
        if (solid <= 0)
            return false;
    }
    else if ((flags & kFlagSkinned) != 0)
    {
        return t > kSkinSelfDistance; // A character, beyond its own surface
    }
    if (consult && (flags & kFlagCameraOnly) != 0)
        return GameShadowConfirms(origin, origin + dir * t);
    return true;
}

// ---- Shadow map reproduction (diagnostics, RTSun::Output::ShadowRepro): for a grid of texels of the game's sun cascades
// 0..kReproCascades-1, a ray from the light side through the texel, against our scene, compared with the game's own
// shadow map depth there. Two variants: only instances drawn in a shadow map (variant 0), and everything (variant 1).
// One uint per texel in Repro: bits 0-1 code of variant 0, 2-3 code of variant 1 (0 match, 1 missing: the game has a
// surface nearer the light than ours, 2 extra: ours is nearer than the game's), 4-31 instance index + 1 of the variant 1
// hit when variant 0 is "missing" (what we left out, 0: nothing in our scene), else of the variant 0 hit.
static const uint kDebugShadowRepro = 9;
static const uint kReproGrid = 512; // Per cascade side; must match RTSun::kReproGrid
static const uint kReproCascades = 2;
RWStructuredBuffer<uint> Repro : register(u5);

// World position of light space ls in cascade rows r0..r3 (ls = x * r0 + y * r1 + z * r2 + r3)
float3 ShadowToWorld(float3 r0, float3 r1, float3 r2, float3 r3, float3 ls)
{
    // Row vector: (ls - r3) = p * A, A's rows r0, r1, r2; p = (ls - r3) * inverse(A)
    // inverse(A) has the columns c0 = r1 x r2, c1 = r2 x r0, c2 = r0 x r1 over det, so p.j = dot(ls - r3, cj) / det
    float3 c0 = cross(r1, r2), c1 = cross(r2, r0), c2 = cross(r0, r1);
    float det = dot(r0, c0);
    float3 v = ls - r3;
    return float3(dot(v, c0), dot(v, c1), dot(v, c2)) / det;
}

void ShadowRepro(uint2 id)
{
    uint c = id.x / kReproGrid;
    uint2 g = uint2(id.x % kReproGrid, id.y);
    if (c >= kReproCascades || c >= ShadowCascades || g.y >= kReproGrid || ShadowConsult == 0)
        return;
    uint step = max(ShadowSize.x / kReproGrid, 1u);
    uint2 texel = min(g * step + step / 2, ShadowSize - 1);
    float4 r0 = ShadowMatrixRow(c, 0), r1 = ShadowMatrixRow(c, 1), r2 = ShadowMatrixRow(c, 2), r3 = ShadowMatrixRow(c, 3);
    float2 uv = (float2(texel) + 0.5) / float2(ShadowSize);
    float z_start = ShadowReversed != 0 ? 1.0 : 0.0; // The light side
    float3 a = ShadowToWorld(r0.xyz, r1.xyz, r2.xyz, r3.xyz, float3(uv, z_start));
    float3 b = ShadowToWorld(r0.xyz, r1.xyz, r2.xyz, r3.xyz, float3(uv, 1.0 - z_start));
    float3 d = b - a;
    float len = max(length(d), 1e-6);
    // Distances along the ray in units of the cascade's depth range (s = 0 at the light side plane, 1 at the far one).
    // The ray starts 50 world units before the plane (the game clamps casters in front of it onto it).
    float back = 50.0 / len;
    // The game's depth over the texel's 3x3 neighbourhood, as s
    float gmin = 1e30, gmax = -1e30;
    [unroll]
    for (int dy = -1; dy <= 1; dy++)
    {
        [unroll]
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 q = clamp(int2(texel) + int2(dx, dy), int2(0, 0), int2(ShadowSize) - 1);
            float v = LoadShadowValue(uint2(q.x, uint(q.y) + c * ShadowSize.y));
            float s = ShadowReversed != 0 ? 1.0 - v : v;
            gmin = min(gmin, s);
            gmax = max(gmax, s);
        }
    }
    float tolerance = (ShadowTolerance + 0.05) / len;
    uint packed = 0;
    uint id0 = 0, id1 = 0;
    uint codes[2];
    [unroll]
    for (uint v = 0; v < 2; v++)
    {
        RayDesc ray;
        ray.Origin = a - d * back;
        ray.Direction = d;
        ray.TMin = 0.0;
        ray.TMax = 1.0 + back;
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE> q;
        q.TraceRayInline(Scene, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
        uint guard = 0;
        while (q.Proceed() && guard++ < 1024)
        {
            uint flags = q.CandidateInstanceContributionToHitGroupIndex();
            if (v == 0 && (flags & kSourceShadow) == 0)
                continue;
            if ((q.CandidateInstanceID() & kInstanceIdCutoutBit) != 0 && (flags & kFlagFrameMesh) == 0)
            {
                int solid = CutoutAlphaTest(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics());
                if (solid == 0)
                    continue;
            }
            q.CommitNonOpaqueTriangleHit();
        }
        bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        float s = hit ? max(q.CommittedRayT() - back, 0.0) : 1.0;
        // The game's map holds depth + its rasterizer bias (constant + slope factor x the depth slope per texel), so a
        // surface reads nearer the light than the map by up to that much: not "extra"
        float bias = abs(ShadowBiasConstant) + abs(ShadowBiasSlope) * 0.5 * (gmax - gmin);
        codes[v] = s > gmax + tolerance ? 1u : (s < gmin - tolerance - bias ? 2u : 0u);
        uint hid = hit ? min(q.CommittedInstanceIndex() + 1, 0x0FFFFFFFu) : 0;
        if (v == 0)
            id0 = hid;
        else
            id1 = hid;
    }
    uint shown = codes[0] == 1 ? id1 : id0;
    Repro[(c * kReproGrid + g.y) * kReproGrid + g.x] = codes[0] | (codes[1] << 2) | (shown << 4);
    CutoutFlushCounters();
}

float PackRGB565(float3 c)
{
    uint3 q = uint3(round(saturate(c) * float3(31.0, 63.0, 31.0)));
    return float((q.r << 11) | (q.g << 5) | q.b) / 65535.0;
}

float3 SourceColor(uint flags, bool cutout)
{
    float3 c;
    if (flags & kFlagSkinned)
        c = float3(1.0, 0.25, 0.25); // Skinned character (per frame): red
    else if (flags & kFlagFrameMesh)
        c = (flags & 3u) == kSourceShadow ? float3(0.5, 0.25, 1.0) : float3(0.95, 0.3, 0.95); // Dynamic vertex buffer (doors, windows; per frame): magenta, from a shadow map purple
    else if ((flags & 3u) == kSourceShadow)
        c = (flags & kFlagMvp) ? float3(1.0, 1.0, 0.2) : float3(1.0, 0.6, 0.1); // Only seen in shadow maps: orange (MVP: yellow)
    else if ((flags & 3u) == (kSourceMain | kSourceShadow))
        c = (flags & kFlagMvp) ? float3(0.3, 0.95, 0.95) : float3(0.35, 0.55, 1.0); // Main view and shadow maps: blue (MVP: cyan)
    else
        c = (flags & kFlagMvp) ? float3(0.7, 1.0, 0.5) : float3(0.25, 0.85, 0.3); // Only seen by the main view: green (MVP: light green)
    return cutout ? c * 0.6 : c; // Alpha tested: darker
}

void DebugScene(uint2 px, bool from_sun)
{
    float2 uv = (float2(px) + 0.5) / float2(Size);
    RayDesc ray;
    if (from_sun)
    {
        float3 L = normalize(LightDir);
        float3 t = abs(L.z) < 0.999 ? normalize(cross(L, float3(0, 0, 1))) : float3(1, 0, 0);
        float3 b = cross(L, t);
        float2 xy = (uv * 2.0 - 1.0) * float2(30.0 * float(Size.x) / float(Size.y), -30.0);
        ray.Origin = CamPos + L * 40.0 + t * xy.x + b * xy.y;
        ray.Direction = -L;
        ray.TMax = 80.0;
    }
    else
    {
        float4 h = mul(float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0), InvViewProj);
        ray.Origin = CamPos;
        ray.Direction = normalize(h.xyz / h.w - CamPos);
        ray.TMax = 200.0;
    }
    ray.TMin = 0.0;
    // Closest hit, everything forced opaque (cutouts show as solid, darker): no candidate loop, so a debug view stays
    // cheap and bounded (a long ray through many alpha tested layers once hung the GPU into a device removal)
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(Scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();
    float3 c;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        c = float3(0.12, 0.12, 0.12); // Nothing in the ray tracing scene
    else
    {
        uint flags = q.CommittedInstanceContributionToHitGroupIndex();
        bool cutout = (q.CommittedInstanceID() & kInstanceIdCutoutBit) != 0 && (flags & kFlagFrameMesh) == 0;
        // Per triangle shade (shows the mesh), plus a little depth falloff
        float shade = 0.7 + 0.3 * float(Hash(q.CommittedPrimitiveIndex() * 0x9E3779B9u + q.CommittedInstanceIndex()) & 255) / 255.0;
        shade *= lerp(1.0, 0.6, saturate(q.CommittedRayT() / (from_sun ? 80.0 : 150.0)));
        c = SourceColor(flags, cutout) * shade;
    }
    VisRaw[px] = PackRGB565(c);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (DebugMode == kDebugShadowRepro)
    {
        ShadowRepro(px);
        return;
    }
    if (any(px >= Size))
        return;
    if (DebugMode == kDebugSceneView || DebugMode == kDebugSceneSun)
    {
        DebugScene(px, DebugMode == kDebugSceneSun);
        CutoutFlushCounters();
        return;
    }
    float d = LoadDepth(px);
    if (IsSky(d))
    {
        VisRaw[px] = (DebugMode == kDebugBlocker || DebugMode == kDebugBlockerAll) ? 0.0 : 1.0; // Sky: lit (blocker id 0)
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
    // Facing away from the light (geometric normal): traced from just behind the surface, which is what the game's
    // shadow map does (it compares depths, not facing). A solid's back face starts inside it and the ray hits its far
    // side (no culling): shadowed. A thin sheet (curtains, cloth, paper) has nothing behind it: lit when nothing else
    // is in the way, like the game, which lights such sheets from behind.
    if (n_dot_l < -0.05)
        n = -n;

    if (DebugMode == kDebugBlocker || DebugMode == kDebugBlockerAll)
    {
        // The visibility's ray (same origin and bias), toward the light's center, closest hit with the cutout test
        float3 o = p0 + n * (Bias + max(ViewZ(p0), 0.0) * 0.002);
        RayDesc ray;
        ray.Origin = o;
        ray.Direction = L;
        ray.TMin = 0.0;
        ray.TMax = TMax;
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(Scene, RAY_FLAG_NONE, (CastersOnly != 0 && DebugMode == kDebugBlocker) ? (0xFF & ~(kInstanceMaskCameraOnly | kInstanceMaskCameraOnlyCutout)) : 0xFF, ray);
        // "All": every object blocks (no shadow map check), cutouts without a mask solid
        const bool consult = ShadowConsult != 0 && DebugMode == kDebugBlocker;
        uint guard = 0;
        while (q.Proceed() && guard++ < 256)
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
                CandidateBlocks(q.CandidateInstanceID(), q.CandidateInstanceContributionToHitGroupIndex(), q.CandidateInstanceIndex(), q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics(), q.CandidateTriangleRayT(), ray.Origin, ray.Direction, 1, 0, consult))
                q.CommitNonOpaqueTriangleHit();
        }
        VisRaw[px] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? float(min(q.CommittedInstanceIndex() + 1, 65535u)) / 65535.0 : 0.0;
        CutoutFlushCounters();
        return;
    }

    float view_z = max(ViewZ(p0), 0.0);
    // Same offset as the AO trace: grows with distance to stay above the depth reconstruction error
    float3 origin = p0 + n * (Bias + view_z * 0.002);

    // Cutouts: exactly as rtao_trace.hlsl. A candidate cutout triangle is tested against its material's alpha mask
    // (texture exact); without a mask, it stops the ray with probability CutoutOpacity (per crossing, or per ray with
    // fast cutouts). With masks off, opacity 1 forces everything opaque and 0 leaves cutouts out.
    // The shadow map check needs camera only objects as candidates: never force everything opaque with it
    const bool consult = ShadowConsult != 0;
    const uint ray_flags = (!consult && CutoutOpacity >= 1.0 && CutoutExact == 0) ? RAY_FLAG_FORCE_OPAQUE : RAY_FLAG_NONE;
    const uint caster_mask = CastersOnly != 0 ? (0xFF & ~(kInstanceMaskCameraOnly | kInstanceMaskCameraOnlyCutout)) : 0xFF;
    const uint ray_mask = caster_mask & ((CutoutOpacity <= 0.0 && CutoutExact == 0) ? (0xFF & ~(kInstanceMaskCutout | kInstanceMaskCameraOnlyCutout)) : 0xFF);
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
            if (CutoutExact != 0 || consult)
                fallback_coin = u < p ? 1 : 0;
            else if (u < p)
                flags_i |= RAY_FLAG_FORCE_OPAQUE;
            else
                mask_i &= ~kInstanceMaskCutout;
        }
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(Scene, flags_i, mask_i, ray);
        const uint seed = Hash(px.x * 0x9E3779B9u + Hash(px.y + Hash(Frame * 64 + i + 0x68E31DA4u)));
        // Textured windows let a fraction through: their transmission (1 - the mask's opacity at the hit) multiplies into
        // the ray, which goes on (no random pass/stop: one ray gives the smooth value, no fizzle)
        float transmission = 1.0;
        uint guard = 0;
        while (q.Proceed() && guard++ < 256)
        {
            if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
                continue;
            const uint flags = q.CandidateInstanceContributionToHitGroupIndex();
            if ((flags & kFlagWindow) != 0 && (q.CandidateInstanceID() & kInstanceIdCutoutBit) != 0)
            {
                float alpha_ref;
                float opacity = CutoutAlphaSample(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(), alpha_ref);
                // The least transmissive layer, not the product: a window is often two meshes at the same place (both
                // carrying the glass), which a product would count twice (squared light). A ray through two separate
                // windows is rare (and gets the darker one).
                transmission = min(transmission, opacity < 0.0 ? 1.0 : saturate(1.0 - opacity)); // No mask yet: clear
                continue;
            }
            if (CandidateBlocks(q.CandidateInstanceID(), flags, q.CandidateInstanceIndex(), q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics(), q.CandidateTriangleRayT(), ray.Origin, ray.Direction, fallback_coin, seed, consult))
                q.CommitNonOpaqueTriangleHit();
        }
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            lit += transmission;
    }
    float vis = lit / float(rays);
    // Textured windows, sharpness: a thin lead line is thinner than the sun's penumbra over the window to receiver
    // distance, so the soft rays only half shade it. One more hard ray toward the light's center, against the windows
    // only, darkens the pixel by WindowSharpness where it meets lead (a stylised crisper pattern, 0: off)
    if (WindowSharpness > 0.0 && vis > 0.0)
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = L;
        ray.TMin = 0.0;
        ray.TMax = TMax;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(Scene, RAY_FLAG_NONE, kInstanceMaskWindow, ray);
        uint guard = 0;
        while (q.Proceed() && guard++ < 64)
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
                CandidateBlocks(q.CandidateInstanceID(), q.CandidateInstanceContributionToHitGroupIndex(), q.CandidateInstanceIndex(), q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics(), q.CandidateTriangleRayT(), ray.Origin, ray.Direction, 1, 0, false))
                q.CommitNonOpaqueTriangleHit();
        }
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            vis *= 1.0 - saturate(WindowSharpness);
    }
    CutoutFlushCounters();
    VisRaw[px] = vis;
}
