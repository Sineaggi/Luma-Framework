// RTAO D3D11 side passes. The D3D12 side lives in "Source/Games/Dishonored 2/rtao_*.cpp" and "Source/Games/Dishonored 2/Shaders".
//
// depth_copy_ps / depth_pack_ps: copy the SSAO pass depth (PS t0 of 0x94445D2D, device depth) into the texture shared with D3D12.
//    Drawn with "Luma_Copy_VS" at the depth texture's size. The "pack" version is used when the driver can't share
//    R32_FLOAT: it stores the 32 bits of depth in the 4 channels of an R8G8B8A8_UNORM target (bit-exact).
// composite_ps: replaces the game's SSAO pixel shader. Same output layout XeGTAO uses: x = visibility, y = 1.
// capture_instances_cs: dispatched next to a captured game draw. Resolves each instance's object to world matrix the
//    same way the game's vertex shader does (see RTAO_NOTES.md) and writes it to the capture ring shared with D3D12.
// skin_vertices_cs: dispatched next to a captured skinned (or DYNAMIC vertex buffer) draw. Skins its vertices like the game's VS and
//    writes them in world space into the skinned vertex ring shared with D3D12.
// alpha_strip_cs: renders one channel of a cutout material's alpha texture into a 256x256 R8 tile (texture exact cutouts,
//    see "Source/Games/Dishonored 2/rtao_alpha.h"), copied from there into the strip shared with D3D12.

// Depth (copy passes) or the shared AO (composite)
Texture2D<float> tex : register(t0);

float4 depth_copy_ps(float4 pos : SV_Position) : SV_Target
{
    return float4(tex.Load(int3(pos.xy, 0)), 0.0, 0.0, 0.0);
}

float4 depth_pack_ps(float4 pos : SV_Position) : SV_Target
{
    uint bits = asuint(tex.Load(int3(pos.xy, 0)));
    return float4(bits & 0xFF, (bits >> 8) & 0xFF, (bits >> 16) & 0xFF, bits >> 24) / 255.0;
}

float4 composite_ps(float4 pos : SV_Position) : SV_Target
{
    return float4(tex.Load(int3(pos.xy, 0)), 1.0, 0.0, 0.0);
}

// "View AO only" debug: an AO texture (AO in x) drawn fullscreen as grey over the back buffer
cbuffer AOViewParams : register(b0)
{
    float2 InvTargetSize;
    float2 UVScale; // Used region of the AO texture (render resolution / texture size)
    float Brightness; // 1 for SDR back buffers, paper white / 80 nits for scRGB
    float ColorPacked; // 1: the texture holds RGB565 colors in its 16 bits (RT sun "Scene view" debug outputs)
    float2 AOViewPad;
};
SamplerState ao_view_sampler : register(s0);

float4 ao_view_ps(float4 pos : SV_Position) : SV_Target
{
    float2 uv = pos.xy * InvTargetSize * UVScale;
    if (ColorPacked != 0.0)
    {
        uint2 size;
        tex.GetDimensions(size.x, size.y);
        uint v = uint(round(saturate(tex.Load(int3(min(uint2(uv * float2(size)), size - 1), 0))) * 65535.0));
        float3 c = float3((v >> 11) & 31u, (v >> 5) & 63u, v & 31u) / float3(31.0, 63.0, 31.0);
        return float4(pow(c, 2.2) * Brightness, 1.0); // sRGB-ish debug colors into a linear back buffer
    }
    float ao = saturate(tex.SampleLevel(ao_view_sampler, uv, 0));
    return float4(ao.xxx * Brightness, 1.0);
}

// ---- Instance capture. Must match RTAO::CaptureParams in rtao_capture.h.

#define RTAO_SOURCE_DIRECT_INSTANCE_BUFFER 1  // StructuredBuffer, stride 96, index = SV_InstanceID + cb_instancebufferoffset
#define RTAO_SOURCE_INDEXED_INSTANCE_BUFFER 2 // Buffer<uint> index (low 24 bits) into a StructuredBuffer, stride 48
#define RTAO_SOURCE_CONSTANT_BUFFER 3         // PerInstanceCB cb_modelmatrixx/y/z rows
#define RTAO_SOURCE_MVP_CONSTANT_BUFFER 4     // PerInstanceCB cb_mvpmatrixx/y/z/w, with the draw's PerViewCB view projection

#define RTAO_NO_OFFSET 0xFFFFFFFF
#define RTAO_FLAG_VALID 1
#define RTAO_FLAG_VIEW_MODEL 2

cbuffer CaptureParams : register(b0)
{
    uint Source;
    uint InstanceCount;
    uint DstSlot;
    uint SlotMask; // ring capacity - 1 (power of two)
    uint InstanceOffsetCB; // byte offset of cb_instancebufferoffset in the game's PerInstanceCB, or RTAO_NO_OFFSET
    uint RowXCB; // byte offsets of cb_modelmatrixx/y/z
    uint RowYCB;
    uint RowZCB;
    uint DepthHackCB; // byte offset of cb_modeldepthhack, or RTAO_NO_OFFSET
    uint RecordRow0; // byte offset of the x row inside an instance record (48 direct, 0 indexed)
    uint AlphaRefCB; // Cutouts: byte offset of the alpha test reference in the game's material cbuffer, or RTAO_NO_OFFSET (AlphaRefConst)
    uint UvScaleBiasCB; // Cutouts: byte offset of cb_layer0_scalebias in the game's material cbuffer, or RTAO_NO_OFFSET (none)
    uint4 MvpCB; // byte offsets of cb_mvpmatrixx/y/z/w (clip.c = dot(float4(pos, 1), cb_mvpmatrix[c]))
    uint ViewProjCB; // byte offset of cb_viewprojectionmatrix in the game's PerViewCB (4 rows: clip = world_h x VP)
    float AlphaRefConst; // Cutouts: the hardcoded reference when AlphaRefCB is RTAO_NO_OFFSET
    uint RecordTag; // capture_instances_cs: written into bits 8..31 of each instance's flags (readback checks it)
    uint DebugKey; // skin_vertices_cs: the draw's mesh key (low bits), for the probe ring (FrameCounters)
    // skin_vertices_cs only
    uint SkinVertexCount;
    uint SkinDstVertex; // first vertex written in SkinOut
    uint SkinSoup; // 0: thread i skins vertex i of the draw, 1: the vertex of the draw's index i
    uint SkinSets; // 1: 4 bones, 2: 8 bones
    uint MtxOffsetCB; // byte offset of cb_mtxoffset in the game's PerInstanceCB
    uint SkinIndexStart; // Arena byte address of the draw's first index
    uint SkinIndex32;
    uint SkinShadow; // 1: a shadow map draw (FrameCounters)
    // Streams 0 position, 1 indices0, 2 weights0, 3 indices1, 4 weights1 (components of the two uint4)
    uint4 StreamBase[2]; // Arena byte address of vertex 0's element
    uint4 StreamStride[2];
    uint4 StreamFormat[2]; // RTAO::VertexFormatCode
};

// The game's PerInstanceCB, bound as is (read past its end returns 0)
cbuffer GameInstanceCB : register(b1)
{
    uint4 GameCB[256];
};

// The game's PerViewCB, as bound for the draw (MVP source only)
cbuffer GameViewCB : register(b2)
{
    uint4 GameViewCB[64];
};

// Cutouts: the pixel shader's cbuffer holding the alpha test reference and the uv scale/bias (its PerInstanceCB)
cbuffer GameMaterialCB : register(b3)
{
    uint4 GameMaterialCB[256];
};

float4x4 Inverse4x4(float4x4 m)
{
    float4x4 a = m;
    float b00 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    float b01 = a[0][0] * a[1][2] - a[0][2] * a[1][0];
    float b02 = a[0][0] * a[1][3] - a[0][3] * a[1][0];
    float b03 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    float b04 = a[0][1] * a[1][3] - a[0][3] * a[1][1];
    float b05 = a[0][2] * a[1][3] - a[0][3] * a[1][2];
    float b06 = a[2][0] * a[3][1] - a[2][1] * a[3][0];
    float b07 = a[2][0] * a[3][2] - a[2][2] * a[3][0];
    float b08 = a[2][0] * a[3][3] - a[2][3] * a[3][0];
    float b09 = a[2][1] * a[3][2] - a[2][2] * a[3][1];
    float b10 = a[2][1] * a[3][3] - a[2][3] * a[3][1];
    float b11 = a[2][2] * a[3][3] - a[2][3] * a[3][2];
    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    float inv_det = abs(det) > 1e-30 ? 1.0 / det : 0.0;
    float4x4 r;
    r[0][0] = (a[1][1] * b11 - a[1][2] * b10 + a[1][3] * b09) * inv_det;
    r[0][1] = (a[0][2] * b10 - a[0][1] * b11 - a[0][3] * b09) * inv_det;
    r[0][2] = (a[3][1] * b05 - a[3][2] * b04 + a[3][3] * b03) * inv_det;
    r[0][3] = (a[2][2] * b04 - a[2][1] * b05 - a[2][3] * b03) * inv_det;
    r[1][0] = (a[1][2] * b08 - a[1][0] * b11 - a[1][3] * b07) * inv_det;
    r[1][1] = (a[0][0] * b11 - a[0][2] * b08 + a[0][3] * b07) * inv_det;
    r[1][2] = (a[3][2] * b02 - a[3][0] * b05 - a[3][3] * b01) * inv_det;
    r[1][3] = (a[2][0] * b05 - a[2][2] * b02 + a[2][3] * b01) * inv_det;
    r[2][0] = (a[1][0] * b10 - a[1][1] * b08 + a[1][3] * b06) * inv_det;
    r[2][1] = (a[0][1] * b08 - a[0][0] * b10 - a[0][3] * b06) * inv_det;
    r[2][2] = (a[3][0] * b04 - a[3][1] * b02 + a[3][3] * b00) * inv_det;
    r[2][3] = (a[2][1] * b02 - a[2][0] * b04 - a[2][3] * b00) * inv_det;
    r[3][0] = (a[1][1] * b07 - a[1][0] * b09 - a[1][2] * b06) * inv_det;
    r[3][1] = (a[0][0] * b09 - a[0][1] * b07 + a[0][2] * b06) * inv_det;
    r[3][2] = (a[3][1] * b01 - a[3][0] * b03 - a[3][2] * b00) * inv_det;
    r[3][3] = (a[2][0] * b03 - a[2][1] * b01 + a[2][2] * b00) * inv_det;
    return r;
}

struct InstanceRecord48 { float4 rows[3]; };
struct InstanceRecord96 { float4 rows[6]; };

Buffer<uint> InstanceIndex : register(t0);
StructuredBuffer<InstanceRecord48> Instances48 : register(t1);
StructuredBuffer<InstanceRecord96> Instances96 : register(t2);
RWByteAddressBuffer CaptureRing : register(u0);

uint LoadGameCB(uint byte_offset)
{
    uint4 v = GameCB[byte_offset >> 4];
    uint c = (byte_offset >> 2) & 3;
    return c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
}

float4 LoadGameCB4(uint byte_offset)
{
    return asfloat(GameCB[byte_offset >> 4]);
}

// The object to world 3x4 of instance "i" of the draw (rows: world.c = dot(rows[c], float4(pos, 1))), like the game's VS
uint ResolveInstance(uint i, out float4 rows[3])
{
    uint instance_offset = InstanceOffsetCB != RTAO_NO_OFFSET ? LoadGameCB(InstanceOffsetCB) : 0;
    rows[0] = float4(0, 0, 0, 0);
    rows[1] = float4(0, 0, 0, 0);
    rows[2] = float4(0, 0, 0, 0);
    uint flags = RTAO_FLAG_VALID;
    if (Source == RTAO_SOURCE_DIRECT_INSTANCE_BUFFER)
    {
        InstanceRecord96 r = Instances96[i + instance_offset];
        uint row0 = RecordRow0 / 16;
        rows[0] = r.rows[row0];
        rows[1] = r.rows[row0 + 1];
        rows[2] = r.rows[row0 + 2];
    }
    else if (Source == RTAO_SOURCE_INDEXED_INSTANCE_BUFFER)
    {
        uint index = InstanceIndex[i + instance_offset] & 0x00FFFFFF;
        InstanceRecord48 r = Instances48[index];
        uint row0 = RecordRow0 / 16;
        rows[0] = r.rows[row0];
        rows[1] = r.rows[row0 + 1];
        rows[2] = r.rows[row0 + 2];
    }
    else if (Source == RTAO_SOURCE_CONSTANT_BUFFER)
    {
        rows[0] = LoadGameCB4(RowXCB);
        rows[1] = LoadGameCB4(RowYCB);
        rows[2] = LoadGameCB4(RowZCB);
    }
    else if (Source == RTAO_SOURCE_MVP_CONSTANT_BUFFER)
    {
        // clip = world_h x VP (rows as stored) and clip.c = dot(pos_h, mvp[c]), so world.k = dot(pos_h, sum_c inv(VP)[c][k] * mvp[c])
        float4 mvp[4] = { LoadGameCB4(MvpCB.x), LoadGameCB4(MvpCB.y), LoadGameCB4(MvpCB.z), LoadGameCB4(MvpCB.w) };
        uint vp = ViewProjCB >> 4;
        float4x4 view_proj = float4x4(asfloat(GameViewCB[vp]), asfloat(GameViewCB[vp + 1]), asfloat(GameViewCB[vp + 2]), asfloat(GameViewCB[vp + 3]));
        float4x4 inv = Inverse4x4(view_proj);
        [unroll] for (uint k = 0; k < 3; k++)
            rows[k] = inv[0][k] * mvp[0] + inv[1][k] * mvp[1] + inv[2][k] * mvp[2] + inv[3][k] * mvp[3];
        // The w row must come out as (0, 0, 0, 1) for an affine object to world: normalise small drift, reject the rest
        float4 w_row = inv[0][3] * mvp[0] + inv[1][3] * mvp[1] + inv[2][3] * mvp[2] + inv[3][3] * mvp[3];
        if (abs(w_row.w) < 1e-6 || any(abs(w_row.xyz) > 1e-2 * abs(w_row.w)))
            flags = 0;
        else
        {
            rows[0] /= w_row.w;
            rows[1] /= w_row.w;
            rows[2] /= w_row.w;
            // A rigid transform (rotation x scale): perpendicular model axes of sane lengths. MVP x inverse(another view's
            // VP) (a shadow map draw whose PerViewCB isn't its own view) leaves a skew: rejected (as RTAO::IsRigidTransform)
            float3 ax = float3(rows[0].x, rows[1].x, rows[2].x), ay = float3(rows[0].y, rows[1].y, rows[2].y), az = float3(rows[0].z, rows[1].z, rows[2].z);
            float lx = length(ax), ly = length(ay), lz = length(az);
            float shortest = min(lx, min(ly, lz)), longest = max(lx, max(ly, lz));
            if (!(shortest > 1e-4) || !(longest < 1e4) || longest > shortest * 1000.0 ||
                abs(dot(ax, ay)) > 0.02 * lx * ly || abs(dot(ax, az)) > 0.02 * lx * lz || abs(dot(ay, az)) > 0.02 * ly * lz)
                flags = 0;
        }
    }
    else
    {
        flags = 0;
    }
    if (DepthHackCB != RTAO_NO_OFFSET && asfloat(LoadGameCB(DepthHackCB)) != 1.0)
        flags |= RTAO_FLAG_VIEW_MODEL; // First person view model (depth squeezed): not world geometry
    return flags;
}

[numthreads(64, 1, 1)]
void capture_instances_cs(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= InstanceCount)
        return;

    float4 rows[3];
    uint flags = ResolveInstance(i, rows);

    // Cutouts (texture exact): the draw's uv scale/bias (halves) and alpha test reference, the same for all its instances
    float4 scalebias = UvScaleBiasCB != RTAO_NO_OFFSET ? asfloat(GameMaterialCB[UvScaleBiasCB >> 4]) : float4(1, 1, 0, 0);
    float alpha_ref = AlphaRefConst;
    if (AlphaRefCB != RTAO_NO_OFFSET)
    {
        uint4 v = GameMaterialCB[AlphaRefCB >> 4];
        uint c = (AlphaRefCB >> 2) & 3;
        alpha_ref = asfloat(c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w)));
    }

    uint address = ((DstSlot + i) & SlotMask) * 64;
    CaptureRing.Store4(address, asuint(rows[0]));
    CaptureRing.Store4(address + 16, asuint(rows[1]));
    CaptureRing.Store4(address + 32, asuint(rows[2]));
    CaptureRing.Store4(address + 48, uint4(flags | ((RecordTag & 0xFFFFFFu) << 8), f32tof16(scalebias.x) | (f32tof16(scalebias.y) << 16), f32tof16(scalebias.z) | (f32tof16(scalebias.w) << 16), asuint(alpha_ref)));
}

// ---- Per frame vertex capture (skinned meshes, DYNAMIC vertex buffers). Must match RTAO::GeometryCapture::CaptureFrameDraw
// (rtao_capture.cpp). Dispatched next to a captured draw (single instance). SkinSets 0: no skinning (dynamic meshes). Replays the game's vertex shader position math: the
// bone palette blend (ro_matrices, 3 float4 rows per bone, row = (index + cb_mtxoffset.x) * 3, weighted, then
// model.c = dot(float4(pos, 1), blended row c)), then the instance's object to world (ResolveInstance), and writes
// world space float3s for the BLAS.

Buffer<float4> Bones : register(t3); // The draw's ro_matrices
ByteAddressBuffer Arena : register(t4); // The geometry arena (the game's vertex/index buffers copied whole), one view of it
ByteAddressBuffer DynamicVertices : register(t5); // DYNAMIC vertex buffers, copied at the draw (stream format flag RTAO_FORMAT_DYNAMIC)
RWByteAddressBuffer SkinOut : register(u1); // float3 per vertex
// Diagnostics (uint counters): [0] shadow map per frame meshes whose matrix was rejected, [1] shadow map per frame meshes,
// [2] main view per frame meshes rejected (not view models), [3] main view per frame meshes, [4] probes written; from byte 32,
// a ring of 64 probes of 32 bytes: mesh key, flags (1 shadow map, 2 skinned), world position of the draw's vertex 0.
// See GeometryCapture::PollCounters.
RWByteAddressBuffer FrameCounters : register(u2);

// Vertex element formats (must match RTAO::VertexFormatCode): kind | components << 4 | bits per component << 8
#define RTAO_KIND_FLOAT 1
#define RTAO_KIND_UNORM 2
#define RTAO_KIND_SNORM 3
#define RTAO_KIND_UINT 4
#define RTAO_KIND_SINT 5
#define RTAO_KIND_HALF 6
#define RTAO_KIND_RGB10A2_UNORM 7
#define RTAO_KIND_RGB10A2_UINT 8
#define RTAO_KIND_BGRA8_UNORM 9
#define RTAO_FORMAT_DYNAMIC (1u << 16) // RTAO::kVertexFormatDynamicBuffer

uint LoadData(uint address, bool dynamic) // Any byte alignment
{
    uint aligned = address & ~3u;
    uint shift = (address & 3u) * 8u;
    uint lo = dynamic ? DynamicVertices.Load(aligned) : Arena.Load(aligned);
    if (shift == 0)
        return lo;
    uint hi = dynamic ? DynamicVertices.Load(aligned + 4) : Arena.Load(aligned + 4);
    return (lo >> shift) | (hi << (32u - shift));
}

uint LoadArena(uint address)
{
    return LoadData(address, false);
}

uint ElementRaw(uint4 dwords, uint component, uint bits)
{
    if (bits == 32)
        return dwords[component];
    if (bits == 16)
        return (dwords[component >> 1] >> ((component & 1u) * 16u)) & 0xFFFFu;
    return (dwords.x >> (component * 8u)) & 0xFFu;
}

int SignExtend(uint value, uint bits)
{
    uint shift = 32u - bits;
    return int(value << shift) >> int(shift);
}

// Like the input assembler: missing components read as (0, 0, 0, 1). "as_uint": the shader input is uint (bit pattern
// of integers), otherwise the bit pattern of floats.
uint4 DecodeElement(uint address, uint code, bool as_uint)
{
    uint kind = code & 15u, count = (code >> 4) & 15u, bits = (code >> 8) & 0xFFu;
    bool dynamic = (code & RTAO_FORMAT_DYNAMIC) != 0;
    uint4 dwords = uint4(LoadData(address, dynamic), 0, 0, 0);
    if (count * bits > 32)
    {
        dwords.y = LoadData(address + 4, dynamic);
        dwords.z = LoadData(address + 8, dynamic);
        dwords.w = LoadData(address + 12, dynamic);
    }
    if (kind == RTAO_KIND_RGB10A2_UNORM || kind == RTAO_KIND_RGB10A2_UINT)
    {
        uint4 raw = uint4(dwords.x & 1023u, (dwords.x >> 10) & 1023u, (dwords.x >> 20) & 1023u, dwords.x >> 30);
        return kind == RTAO_KIND_RGB10A2_UINT ? raw : asuint(float4(raw) / float4(1023, 1023, 1023, 3));
    }
    uint4 result = as_uint ? uint4(0, 0, 0, 1) : asuint(float4(0, 0, 0, 1));
    [unroll] for (uint c = 0; c < 4; c++)
    {
        if (c >= count)
            break;
        uint raw = ElementRaw(dwords, c, bits);
        uint value;
        if (kind == RTAO_KIND_FLOAT)
            value = raw;
        else if (kind == RTAO_KIND_HALF)
            value = asuint(f16tof32(raw));
        else if (kind == RTAO_KIND_UNORM || kind == RTAO_KIND_BGRA8_UNORM)
            value = asuint(float(raw) / float((1u << bits) - 1u));
        else if (kind == RTAO_KIND_SNORM)
            value = asuint(max(float(SignExtend(raw, bits)) / float((1u << (bits - 1u)) - 1u), -1.0));
        else if (kind == RTAO_KIND_SINT)
            value = uint(SignExtend(raw, bits));
        else
            value = raw; // UINT
        result[c] = value;
    }
    if (kind == RTAO_KIND_BGRA8_UNORM)
        result.xz = result.zx;
    return result;
}

uint StreamComponent(uint4 values[2], uint stream)
{
    uint4 v = values[stream >> 2];
    uint c = stream & 3u;
    return c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
}

uint StreamAddress(uint stream, uint vertex)
{
    uint4 bases[2] = { StreamBase[0], StreamBase[1] };
    uint4 strides[2] = { StreamStride[0], StreamStride[1] };
    return StreamComponent(bases, stream) + vertex * StreamComponent(strides, stream);
}

uint StreamCode(uint stream)
{
    uint4 formats[2] = { StreamFormat[0], StreamFormat[1] };
    return StreamComponent(formats, stream);
}

[numthreads(64, 1, 1)]
void skin_vertices_cs(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= SkinVertexCount)
        return;

    uint vertex = i;
    if (SkinSoup != 0)
        vertex = SkinIndex32 != 0 ? LoadArena(SkinIndexStart + i * 4) : (LoadArena(SkinIndexStart + i * 2) & 0xFFFFu);

    float4 rows[3];
    uint flags = ResolveInstance(0, rows);

    if (i == 0)
    {
        uint base = SkinShadow != 0 ? 0 : 8;
        FrameCounters.InterlockedAdd(base + 4, 1);
        if ((flags & RTAO_FLAG_VALID) == 0)
            FrameCounters.InterlockedAdd(base, 1);
    }
    float3 world = asfloat(0x7FC00000u).xxx; // NaN: DXR treats the triangle as inactive (view model, unresolved matrix)
    if (flags == RTAO_FLAG_VALID)
    {
        float4 position = float4(asfloat(DecodeElement(StreamAddress(0, vertex), StreamCode(0), false).xyz), 1.0);
        uint mtx_offset = SkinSets != 0 ? LoadGameCB(MtxOffsetCB) : 0;
        float4 bone0 = float4(0, 0, 0, 0), bone1 = float4(0, 0, 0, 0), bone2 = float4(0, 0, 0, 0);
        [unroll] for (uint set = 0; set < 2; set++)
        {
            if (set >= SkinSets)
                break;
            uint4 indices = DecodeElement(StreamAddress(1 + set * 2, vertex), StreamCode(1 + set * 2), true);
            float4 weights = asfloat(DecodeElement(StreamAddress(2 + set * 2, vertex), StreamCode(2 + set * 2), false));
            [unroll] for (uint b = 0; b < 4; b++)
            {
                uint row = (indices[b] + mtx_offset) * 3;
                bone0 += Bones[row] * weights[b];
                bone1 += Bones[row + 1] * weights[b];
                bone2 += Bones[row + 2] * weights[b];
            }
        }
        float4 model = SkinSets != 0 ? float4(dot(position, bone0), dot(position, bone1), dot(position, bone2), 1.0) : position; // No bones: a dynamic mesh
        world = float3(dot(model, rows[0]), dot(model, rows[1]), dot(model, rows[2]));
    }
    SkinOut.Store3((SkinDstVertex + i) * 12, asuint(world));
    if (i == 0 && flags == RTAO_FLAG_VALID)
    {
        uint slot;
        FrameCounters.InterlockedAdd(16, 1, slot);
        const uint address = 32 + (slot % 64) * 32;
        FrameCounters.Store4(address, uint4(DebugKey, (SkinShadow != 0 ? 1u : 0u) | (SkinSets != 0 ? 2u : 0u), asuint(world.x), asuint(world.y)));
        FrameCounters.Store(address + 16, asuint(world.z));
    }
}

// ---- Texture exact cutouts: one channel of a material's alpha texture into a 256x256 R8 tile. Must match
// RTAO::CutoutMaterials::RecordStrip (rtao_alpha.cpp). Mip aware: sampled (trilinear) at the source mip whose size is
// closest to the tile; past the last mip the view has, a grid of taps averages the texels a tile texel covers.

cbuffer AlphaStripParams : register(b0)
{
    uint AlphaChannel; // 0..3: that channel. Textured windows (main.cpp RTSunWindowPattern), the opacity: 4 the lead of a
                       // BC5 SNORM normal map (raised = lead), 5 one minus channel x, 6 channel x, 7 one minus the luminance
    uint AlphaSourceMips; // Mips in the view
    float2 AlphaSourceSize; // Size of the view's most detailed mip
};
Texture2D<float4> AlphaSource : register(t0);
SamplerState AlphaSampler : register(s0); // Linear, wrap
RWTexture2D<unorm float> AlphaTile : register(u0);

[numthreads(8, 8, 1)]
void alpha_strip_cs(uint3 id : SV_DispatchThreadID)
{
    const float tile = 256.0;
    float lod = max(log2(max(AlphaSourceSize.x, AlphaSourceSize.y) / tile), 0.0);
    float last = float(max(AlphaSourceMips, 1u) - 1u);
    float sample_lod = min(lod, last);
    // Taps per axis to cover what the missing mips would have averaged (1 when the view has the mip)
    uint taps = (uint)clamp(exp2(ceil(lod - sample_lod)), 1.0, 4.0);
    float value = 0.0;
    [loop] for (uint y = 0; y < taps; y++)
    {
        [loop] for (uint x = 0; x < taps; x++)
        {
            float2 uv = (float2(id.xy) + (float2(x, y) + 0.5) / float(taps)) / tile;
            float4 texel = AlphaSource.SampleLevel(AlphaSampler, uv, sample_lod);
            float v;
            if (AlphaChannel == 4)
                v = saturate(length(texel.xy) * 2.0);
            else if (AlphaChannel == 5)
                v = 1.0 - texel.x;
            else if (AlphaChannel == 6)
                v = texel.x;
            else if (AlphaChannel == 7)
                v = 1.0 - dot(saturate(texel.rgb), float3(0.299, 0.587, 0.114));
            else
                v = AlphaChannel == 0 ? texel.x : (AlphaChannel == 1 ? texel.y : (AlphaChannel == 2 ? texel.z : texel.w));
            value += v;
        }
    }
    AlphaTile[id.xy] = value / float(taps * taps);
}
