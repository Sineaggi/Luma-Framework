// The game's screen shadows pass ("screenShadows.material": the shadow lookup is moved toward the camera by a 4x4
// ordered dither of cb_screenshadows_stepsize), with Luma's RT sun shadows. See Includes/ScreenShadows.hlsl.
#include "Includes/ScreenShadows.hlsl"

void main(
  float4 v0 : INTERP0,
  float4 v1 : SV_POSITION0,
  out float4 o0 : SV_TARGET0)
{
  // Dither index 1..63 from the pixel position (bits as the game's bfi/ubfe sequence)
  uint2 p = uint2(v1.xy);
  uint z = ((p.y & 1u) << 5) | ((p.x & 1u) << 4);
  z = (z + 16u) & 48u;
  uint a = ((p.y >> 2) & 1u) * 2u + ((p.x >> 2) & 1u) + 1u;
  uint b = ((p.y >> 1) & 1u) * 2u + ((p.x >> 1) & 1u) + 1u;
  uint y = (z & ~12u) | ((b & 3u) << 2);
  uint x = (y & ~3u) | (a & 3u);
  float step = (float)x * cb_screenshadows_stepsize * 0.015625;

  float3 world = ScreenShadowsWorldPos(v0.xy);
  world += normalize(cb_wscampos - world) * step;
  o0 = ScreenShadows(v0.xy, world);
}
