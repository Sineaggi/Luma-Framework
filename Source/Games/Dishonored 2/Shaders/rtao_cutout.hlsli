// Texture exact cutouts (see rtao_alpha.h): resolves a cutout candidate hit of an AO ray against the material's alpha mask.
// Included by rtao_trace.hlsl (after rtao_common.hlsli). Bindings: AOPass root signature in rtao_pass.cpp.

// Must match RTAO::CutoutGpuRecord (rtao_alpha.h)
struct CutoutRecord
{
    uint IndexOffset; // Arena byte offset of the mesh's first index (the BLAS's primitive 0)
    uint UvOffset; // Arena byte offset of TEXCOORD0 of the BLAS's vertex 0
    uint UvStrideIndex32; // stride | index32 << 31
    uint UvCodeSlice; // RTAO::VertexFormatCode | atlas slice << 16 (0xFFFF: no mask yet)
    float4 ScaleBias; // uv = TEXCOORD0 * xy + zw (the game's cb_layer0_scalebias)
    float AlphaRef; // The game discards where alpha < this
    uint3 RecordPad;
};

StructuredBuffer<CutoutRecord> CutoutRecords : register(t2);
ByteAddressBuffer CutoutArena : register(t3); // The geometry arena (the game's vertex/index buffers, copied whole)
Texture2DArray<float> CutoutAtlas : register(t4); // One 256x256 R8 mask per material (the raw alpha channel)
SamplerState CutoutSampler : register(s0); // Linear, wrap
RWByteAddressBuffer CutoutCounters : register(u4); // [0] masked tests, [1] masked hits, [2] fallback (stochastic) tests

static const uint kNoCutoutRecord = kInstanceIdCutoutBit - 1; // RTAO::kNoCutoutRecord
static uint g_cutout_exact_tests = 0;
static uint g_cutout_exact_hits = 0;
static uint g_cutout_fallback_tests = 0;

uint CutoutLoad(uint address) // Any byte alignment
{
    uint aligned = address & ~3u;
    uint shift = (address & 3u) * 8u;
    uint lo = CutoutArena.Load(aligned);
    if (shift == 0)
        return lo;
    uint hi = CutoutArena.Load(aligned + 4);
    return (lo >> shift) | (hi << (32u - shift));
}

// Kinds of RTAO::VertexFormatCode (kind | components << 4 | bits << 8) a uv can have: float, unorm, snorm, half
float2 CutoutDecodeUV(uint address, uint code)
{
    uint kind = code & 15u, bits = (code >> 8) & 0xFFu;
    if (bits == 32)
        return asfloat(uint2(CutoutLoad(address), CutoutLoad(address + 4))); // float
    uint packed = CutoutLoad(address);
    if (bits == 16)
    {
        uint2 raw = uint2(packed & 0xFFFFu, packed >> 16);
        if (kind == 6) // half
            return f16tof32(raw);
        if (kind == 2) // unorm
            return float2(raw) / 65535.0;
        return max(float2(int2(raw << 16) >> 16) / 32767.0, -1.0); // snorm
    }
    uint2 raw = uint2(packed & 0xFFu, (packed >> 8) & 0xFFu); // 8 bit
    if (kind == 2)
        return float2(raw) / 255.0;
    return max(float2(int2(raw << 24) >> 24) / 127.0, -1.0);
}

// 1: the candidate is solid at the hit point, 0: it is a hole there, -1: no mask for it (the caller's stochastic fallback)
// reference_override >= 0: compared instead of the record's reference (textured windows: a per ray random number, so an
// opacity of 0.3 stops 30% of the rays)
int CutoutAlphaTest(uint instance_id, uint primitive, float2 barycentrics, float reference_override = -1.0)
{
    uint id = instance_id & kNoCutoutRecord;
    if (CutoutExact == 0 || id == kNoCutoutRecord)
    {
        g_cutout_fallback_tests++;
        return -1;
    }
    CutoutRecord r = CutoutRecords[id];
    uint slice = r.UvCodeSlice >> 16;
    if (slice == 0xFFFFu)
    {
        g_cutout_fallback_tests++;
        return -1;
    }
    bool index32 = (r.UvStrideIndex32 >> 31) != 0;
    uint stride = r.UvStrideIndex32 & 0x7FFFFFFFu;
    uint3 indices;
    if (index32)
    {
        uint base = r.IndexOffset + primitive * 12;
        indices = uint3(CutoutLoad(base), CutoutLoad(base + 4), CutoutLoad(base + 8));
    }
    else
    {
        uint base = r.IndexOffset + primitive * 6;
        uint a = CutoutLoad(base), b = CutoutLoad(base + 4);
        indices = uint3(a & 0xFFFFu, a >> 16, b & 0xFFFFu);
    }
    uint code = r.UvCodeSlice & 0xFFFFu;
    float2 uv0 = CutoutDecodeUV(r.UvOffset + indices.x * stride, code);
    float2 uv1 = CutoutDecodeUV(r.UvOffset + indices.y * stride, code);
    float2 uv2 = CutoutDecodeUV(r.UvOffset + indices.z * stride, code);
    float2 uv = uv0 * (1.0 - barycentrics.x - barycentrics.y) + uv1 * barycentrics.x + uv2 * barycentrics.y;
    uv = uv * r.ScaleBias.xy + r.ScaleBias.zw;
    float alpha = CutoutAtlas.SampleLevel(CutoutSampler, float3(uv, float(slice)), 0.0);
    bool solid = alpha >= (reference_override >= 0.0 ? reference_override : r.AlphaRef);
    g_cutout_exact_tests++;
    g_cutout_exact_hits += solid ? 1u : 0u;
    return solid ? 1 : 0;
}

// Once per thread, at the end (one atomic per wave and counter)
void CutoutFlushCounters()
{
    uint3 sums = uint3(WaveActiveSum(g_cutout_exact_tests), WaveActiveSum(g_cutout_exact_hits), WaveActiveSum(g_cutout_fallback_tests));
    if (WaveIsFirstLane() && any(sums != 0))
    {
        uint unused;
        CutoutCounters.InterlockedAdd(0, sums.x, unused);
        CutoutCounters.InterlockedAdd(4, sums.y, unused);
        CutoutCounters.InterlockedAdd(8, sums.z, unused);
    }
}
