// RTAO (milestone M0) D3D11 side passes. The D3D12 side lives in "Source/Games/Dishonored 2/rtao_interop.cpp".
//
// depth_copy_ps / depth_pack_ps: copy the SSAO pass depth (PS t0 of 0x94445D2D, device depth) into the texture shared with D3D12.
//    Drawn with "Luma_Copy_VS" at the depth texture's size. The "pack" version is used when the driver can't share
//    R32_FLOAT: it stores the 32 bits of depth in the 4 channels of an R8G8B8A8_UNORM target (bit-exact).
// composite_ps: replaces the game's SSAO pixel shader. Same output layout XeGTAO uses: x = visibility, y = 1.

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
