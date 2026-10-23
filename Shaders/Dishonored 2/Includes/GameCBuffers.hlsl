#ifndef LUMA_GAME_CB_STRUCTS
#define LUMA_GAME_CB_STRUCTS

#ifdef __cplusplus
#include "../../../Source/Core/includes/shader_types.h"
#endif

namespace CB
{
struct LumaGameSettings
{
   float BloomStrength;    // 1 = vanilla, 0 = disabled.
   float LensDirtStrength; // 1 = vanilla, 0 = disabled.
   float LensFlareStrength;// 1 = vanilla, 0 = disabled.
   float VignetteStrength; // 1 = vanilla, 0 = disabled.
   // RT sun shadows, read by the screen shadows pass (Includes/ScreenShadows.hlsl): 0 = the game's shadow maps,
   // 1 = ray traced, 2 = ray traced x the game's. Only applies when Luma binds the traced visibility that frame.
   uint RTSunMode;
   float3 RTSunPad;
};

struct LumaGameData
{
   float Dummy;
};
} // namespace CB

#endif // LUMA_GAME_CB_STRUCTS