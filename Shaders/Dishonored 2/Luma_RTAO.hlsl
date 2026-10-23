// RTAO D3D11 side passes. The D3D12 side lives in "Source/Games/Dishonored 2/rtao_*.cpp" and "Source/Games/Dishonored 2/Shaders".
//
// depth_copy_ps / depth_pack_ps: copy the SSAO pass depth (PS t0 of 0x94445D2D, device depth) into the texture shared with D3D12.
//    Drawn with "Luma_Copy_VS" at the depth texture's size. The "pack" version is used when the driver can't share
//    R32_FLOAT: it stores the 32 bits of depth in the 4 channels of an R8G8B8A8_UNORM target (bit-exact).
// composite_ps: replaces the game's SSAO pixel shader. Same output layout XeGTAO uses: x = visibility, y = 1.
// capture_instances_cs: dispatched next to a captured game draw. Resolves each instance's object to world matrix the
//    same way the game's vertex shader does (see RTAO_NOTES.md) and writes it to the capture ring shared with D3D12.

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
    float3 AOViewPad;
};
SamplerState ao_view_sampler : register(s0);

float4 ao_view_ps(float4 pos : SV_Position) : SV_Target
{
    float2 uv = pos.xy * InvTargetSize * UVScale;
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
    uint CapturePad0;
    uint CapturePad1;
    uint4 MvpCB; // byte offsets of cb_mvpmatrixx/y/z/w (clip.c = dot(float4(pos, 1), cb_mvpmatrix[c]))
    uint ViewProjCB; // byte offset of cb_viewprojectionmatrix in the game's PerViewCB (4 rows: clip = world_h x VP)
    uint3 CapturePad2;
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

[numthreads(64, 1, 1)]
void capture_instances_cs(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= InstanceCount)
        return;

    uint instance_offset = InstanceOffsetCB != RTAO_NO_OFFSET ? LoadGameCB(InstanceOffsetCB) : 0;
    float4 rows[3] = { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };
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
        }
    }
    else
    {
        flags = 0;
    }
    if (DepthHackCB != RTAO_NO_OFFSET && asfloat(LoadGameCB(DepthHackCB)) != 1.0)
        flags |= RTAO_FLAG_VIEW_MODEL; // First person view model (depth squeezed): not world geometry

    uint address = ((DstSlot + i) & SlotMask) * 64;
    CaptureRing.Store4(address, asuint(rows[0]));
    CaptureRing.Store4(address + 16, asuint(rows[1]));
    CaptureRing.Store4(address + 32, asuint(rows[2]));
    CaptureRing.Store4(address + 48, uint4(flags, 0, 0, 0));
}
