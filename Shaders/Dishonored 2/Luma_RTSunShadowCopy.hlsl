// RT sun shadows, shadow map check (see "Shadow map check" in Source/Games/Dishonored 2/RTAO_NOTES.md): at the game's
// screen shadows draw, copies its sun cascades (PS t0 "ro_lightdepthcomparemap", slices 0..cascades-1) into one texture
// shared with D3D12, stacked vertically, plus a last row with each cascade's light matrix (the draw's own PerInstanceCB,
// still bound at b2), so the D3D12 trace reads a shadow map and matrices of the same frame. After the matrices (x 48..55):
// cb_parallellightslicefars and the view matrix's z column (PerViewCB b1), with which the game picks a pixel's cascade.
// shadow_copy_ps writes R32_FLOAT, shadow_pack_ps the same 32 bits packed into R8G8B8A8_UNORM (when the driver can't
// share a float texture, as the RTAO depth copy). Read back by Shaders/rtsun_trace.hlsl (LoadShadowValue, ShadowMatrixRow).

Texture2DArray<float> shadow_cascades : register(t0);

cbuffer PerInstanceCB : register(b2)
{
  float4 cb_parallellightslicefars : packoffset(c0);
  float4 cb_shadowprojections[20] : packoffset(c1); // Cascade c's rows: [c * 4 + 0..3]
}

cbuffer PerViewCB : register(b1)
{
  float4 cb_viewmatrix[4] : packoffset(c6);
}

static const uint kCascades = 3; // Must match RTSun::kShadowCascades

float ShadowValue(uint2 p)
{
  uint w, h, slices;
  shadow_cascades.GetDimensions(w, h, slices);
  uint cascade = p.y / h;
  if (cascade < kCascades)
    return cascade < slices ? shadow_cascades.Load(int4(p.x, p.y - cascade * h, cascade, 0)) : 1.0;
  // Matrix row: 16 values per cascade, then the slice fars and the view z column
  uint c = p.x / 16, i = p.x % 16;
  if (c < kCascades)
    return cb_shadowprojections[c * 4 + i / 4][i % 4];
  uint k = p.x - kCascades * 16;
  if (k < 4)
    return cb_parallellightslicefars[k];
  if (k < 8)
    return cb_viewmatrix[k - 4].z;
  return 0.0;
}

float4 shadow_copy_ps(float4 pos : SV_Position) : SV_Target
{
  return float4(ShadowValue(uint2(pos.xy)), 0.0, 0.0, 0.0);
}

float4 shadow_pack_ps(float4 pos : SV_Position) : SV_Target
{
  uint bits = asuint(ShadowValue(uint2(pos.xy)));
  return float4(bits & 0xFF, (bits >> 8) & 0xFF, (bits >> 16) & 0xFF, bits >> 24) / 255.0;
}
