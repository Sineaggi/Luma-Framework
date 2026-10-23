// The game's screen shadows pass ("screenShadowsNoJitter.material"), with Luma's RT sun shadows. See Includes/ScreenShadows.hlsl.
#include "Includes/ScreenShadows.hlsl"

void main(
  float4 v0 : INTERP0,
  out float4 o0 : SV_TARGET0)
{
  o0 = ScreenShadows(v0.xy, ScreenShadowsWorldPos(v0.xy));
}
