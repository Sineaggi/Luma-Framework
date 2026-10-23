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
};

// The game's PerInstanceCB, bound as is (read past its end returns 0)
cbuffer GameInstanceCB : register(b1)
{
    uint4 GameCB[256];
};

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
