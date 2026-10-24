// Diagnostics (RT diagnostic capture): draws a texture the game bound (any format, BC included) into a 512x512 RGBA8
// target, read back and saved as PNG, to see what a material's textures hold (e.g. the window panes' textures).

Texture2D<float4> tex : register(t0);
SamplerState smp : register(s0);

float4 tex_dump_ps(float4 pos : SV_Position) : SV_Target
{
  return saturate(tex.SampleLevel(smp, pos.xy / 512.0, 0));
}
