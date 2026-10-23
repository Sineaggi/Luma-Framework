# RTAO notes: geometry capture in Dishonored 2 (Void Engine)

Offline analysis, 2026-10-23. Source: the 6185 vertex shaders and 48 compute shaders Luma dumped while the game
ran (`<game>\Luma\Dishonored 2\Dump\0x*.vs_5_0.cso`). That is the set the game really created, so the game's
archives were not needed. RDEF/ISGN were parsed for every shader; representative variants were disassembled
with `fxc /dumpbin`. Nothing here was verified in game yet.

## Vertex input

- 47 distinct input signatures. Every mesh shader reads `VERTEX_POSITION0` as 3 components (`xyz`, float as
  the shader sees it). The storage format in the vertex buffer comes from the input layout, not the shader
  (see "What the addon can see").
- Other semantics: `VERTEX_NORMAL0` (4), `VERTEX_TANGENT0` (4), `VERTEX_TEXCOORD0/1` (2), `VERTEX_TEXCOORD4/5`
  (4), `VERTEX_COLOR0` (4), and for skinned meshes `VERTEX_SKININDICES0/1` (uint4) +
  `VERTEX_SKINWEIGHTS0/1` (4), i.e. 4 or 8 bones.
- Instanced shaders also take `SV_InstanceID`.

## Where the object-to-world matrix comes from (3 families)

All mesh VS bind `PerViewCB` at b1 (the same struct as `CBPerViewGlobals` in main.cpp) and `PerInstanceCB`
at b2. Clip position is always `world.x * VP[0] + world.y * VP[1] + world.z * VP[2] + VP[3]` with `VP` =
`cb_viewprojectionmatrix` (rows c2..c5), then `clip.z *= cb_modeldepthhack` (first-person view model).

1. Direct instance buffer: `StructuredBuffer ro_directinstancebuffer`, stride 96. Index =
   `SV_InstanceID + PerInstanceCB.cb_instancebufferoffset`. World 3x4, row major: x row at byte 48, y at 64,
   z at 80 (`world = dot(row, float4(pos, 1))`). Bytes 0..47 are very likely the previous frame's 3x4
   (motion vectors), unconfirmed.
2. Indexed instance buffer: `Buffer<uint> ro_instancebufferindex` +
   `StructuredBuffer ro_instancebuffer`, stride 48. Index =
   `ro_instancebufferindex[SV_InstanceID + cb_instancebufferoffset] & 0x00FFFFFF`; world 3x4 rows at bytes 0,
   16, 32.
3. Constant buffer: `PerInstanceCB.cb_modelmatrixx/y/z` (float4 rows, object to world) and
   `cb_mvpmatrixx/y/z/w`. Some variants have only the MVP (world = inverse(VP) * MVP * pos), or
   `cb_modelviewmatrix`. Offsets differ per shader (the cbuffer also holds material params; 4600+ distinct
   PerInstanceCB layouts), so offsets must come from each shader's reflection (RDEF), per pipeline.

Usage of used-variable sets in PerInstanceCB: 2750 shaders use only `cb_instancebufferoffset`, 1288
`cb_instancebufferoffset + cb_mtxoffset`, 1159 model+MVP rows, 468 model+MVP+`cb_mtxoffset`, the rest MVP
only / previous MVP / model-view.

Slots of the instance buffers vary per shader (t0..t9), so they also come from reflection.

## Skinning

Done in the vertex shader. `Buffer<float4> ro_matrices`, 3 float4 rows per bone, row index =
`(VERTEX_SKININDICES[i] + PerInstanceCB.cb_mtxoffset.x) * 3 + {0,1,2}`, blended by `VERTEX_SKINWEIGHTS`. The
skinned result is in model space, then goes through the family 1/2/3 transform. No compute skinning (no CS
binds bones). For RT that means either replaying the skinning in our own compute pass (inputs: the VB, the
bone buffer SRV, cb_mtxoffset) or capturing post-VS positions with stream output.

Checked on all 1930 skinned VS of the dump (disassembly scan): every one uses `cb_mtxoffset.x` of PerInstanceCB
(cb2) added to the skin indices, 3 rows per bone, model space result (`dot(float4(pos, 1), blended row)`), then
the family transform. Split: 4 bones (SKININDICES0/WEIGHTS0) 360, 8 bones (0 and 1) 1570; direct instance buffer
639, indexed 649, cb_modelmatrix 477, MVP only 165. `ro_matrices` is `Buffer<float4>`, slot t0 or t2 (reflected).

Implemented (rtao-skin): `skin_vertices_cs` replays that per skinned draw from the arena copy of the VBs and
writes world space vertices; one BLAS per draw, refit every frame, current-frame-only TLAS instance (see
rtao_capture.h). Meshes with DYNAMIC vertex buffers (doors, windows) take the same path without bones.

Raw SRVs on the geometry arena (D3D11, RTX 4090): on a buffer of 512 MB or more, only views of exactly 2^27
elements (512 MB) at multiples of 512 MB are accepted (anything else E_INVALIDARG); smaller buffers take any view.

## GPU culling (risk)

CS `0x2537D38B` ("arkcullgeometry") reads `ro_arkcullgeometry_vertexdata/indexdata/instancedata` plus the
instance buffers and writes `rw_arkcullgeometry_indirectargs` and `rw_arkcullgeometry_filteredindices`. So
some geometry is drawn with `DrawIndexedInstancedIndirect` and a view-culled index buffer: the CPU does not
know the instance/index counts, and the index buffer bound at draw time only holds visible triangles. For
those, the unculled source is the cull CS inputs (vertex/index data SRVs), not the draw's IB.

## Variant counts

| family | static | skinned |
| --- | --- | --- |
| cbuffer matrix | 1465 | 632 |
| `ro_instancebuffer` (indexed) | 1367 | 641 |
| `ro_directinstancebuffer` | 1358 | 631 |

Plus a few special ones: wind (`ro_windbuffer`, `ro_splineinfos`), ocean/FFT, particles
(`ro_particleparametersbuffer`), fullscreen/post.

## Depth-only vertex shaders (capture candidates)

Output `SV_POSITION` only (opaque) or `SV_POSITION + INTERP0` (uv, alpha tested): 28/18/15 static
position-only (cbuffer/direct/indexed), 26 skinned position-only, ~200+ with one interpolator. They do the
exact same position math as the G-buffer shaders, so capture could hook either. Shadow map draws very likely
use these same shaders, so capture must filter by view (the PerViewCB check main.cpp already does in
`UpdateGlobalCB`, or the bound depth target).

Recommendation: capture from the main-view depth prepass if it covers all opaque geometry (cheapest shaders,
fewest bindings, alpha-tested ones carry the uv). Otherwise the G-buffer pass. Confirm in game which draws
exist in the main view (DEVELOPMENT build, Luma's draw call list).

## What the addon can see

- ReShade reports D3D11 `CreateInputLayout` as `create_pipeline` / `init_pipeline` with an `input_layout`
  subobject (`reshade::api::input_element`: semantic, index, format, buffer slot, offset, step rate). Luma's
  core already records these into `device_data.input_layouts_descs`, but only in DEVELOPMENT builds
  (core.hpp ~3897). RTAO capture should register its own `init_pipeline` handler (or enable that path) to read
  the position format and offset at draw time.
- Per draw: `IAGetVertexBuffers` / `IAGetIndexBuffer` / `IAGetInputLayout`, `VSGetConstantBuffers(2)` (the
  PerInstanceCB contents need a map hook or a copy), `VSGetShaderResources` for the instance buffers and
  `ro_matrices`.
- The game's VBs and IBs are not shareable with D3D12, so they are copied (D3D11 `CopySubresourceRegion`) into
  a buffer both APIs can see. Measured on the RTX 4090 (probe + `Tests/rtao_interop_test.cpp`):
  - a D3D12 buffer with `D3D12_HEAP_FLAG_SHARED` does not open in D3D11 (`OpenSharedResource1`: E_INVALIDARG,
    with any resource flags);
  - a D3D11 buffer refuses `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` (E_INVALIDARG);
  - a D3D11 buffer with the legacy `D3D11_RESOURCE_MISC_SHARED` + `IDXGIResource::GetSharedHandle`, opened with
    `ID3D12Device::OpenSharedHandle`, works. The test copies data through it and builds a BLAS straight from it
    (`Interop::CreateSharedBuffer`, `BlasCache::BuildFromGpu`).

## Wind (vertex animated vegetation)

66 of the dumped vertex shaders animate vertices (none of them skinned), in two algorithms. Both displace the vertex in model
space, then transform it with an ordinary family 1/2/3 matrix, so capture treats them like their base family and
ignores the displacement: the BLAS is the rest pose (`VertexShaderInfo::wind`, UI "RTAO: wind meshes").

1. Vegetation bending (CryEngine style), 59 shaders with `cb_vegetanim_*` in PerInstanceCB: 11 read the wind from
   `StructuredBuffer ro_windbuffer` (stride 32: `previousWind` at 0, `wind` at 16; index `SV_InstanceID +
   cb_windbufferoffset`, so per instance), the other 48 a constant wind (e.g. 0x09B3E8ED: `(0, 0.01)` in xy, strength 0). From
   0x65471A10 (indexed family), with `c` = VERTEX_COLOR, `n` = VERTEX_NORMAL.xy, `p` = VERTEX_POSITION,
   `T` = cb_localtime.x, model z is up:
   - wind into model space: `w = (dot(normalize(row0.xyz), wind.xyz), dot(normalize(row1.xyz), wind.xyz)) * wind.w`
     (rows of the instance's world matrix), strength `s = wind.w`.
   - detail bending: `phase = dot(p, c.y + 1)`; `v = ((phase, phase, c.y, c.y) + T) * (1.975, 0.793, 0.375, 0.193)`;
     `u = frac((frac(v) * 2 - 1) * leaffreq + 0.5) * 2 - 1` (triangle wave), `t = u*u*(3 - 2*abs(u))` (smoothed);
     `leaf = t.x + t.y - 1`, `branch = t.z + t.w`. `p.x += leaf * n.x * A`, `p.z += leaf * n.y * A`,
     `p.y += branch * c.x * branchamplitude`, with `A = max(s, leafminwindstrength) * leafamplitude * c.z`.
   - main bending: `b = sin(mainbendingfreq * T) * mainbendingvar + mainbendingamplitude`; `f = p.z * b + 1;
     f = f^4 - f^2`; `p.xy += w * f`; then `p` is rescaled to its pre-bend length (no stretching).
   Per instance (own wind vector and rotation) and per frame (time), amplitude scaled by the vertex colors.
2. Spline wind, 7 shaders with `ro_splineinfos` (stride 24: `m_position`, `m_direction`) + `ro_windbuffer` at
   `cb_windbufferoffset` (per draw): the vertex is rotated around a pivot on a spline segment
   (`m_position + m_direction * TEXCOORD1.x`), about the horizontal axis perpendicular to the wind, by an angle
   `wind.w * (TEXCOORD1.y * pi/800 - color.x * pi/1600 * (0.5 + 0.5 * cos(10 * color.z * uv.y + 0.05 * (20 *
   color.w - 10) * wind.w * T)))`: small swings of hanging things (cloth, ropes, signs).

Replaying it (not implemented): the displacement differs per instance and per frame, so every animated instance
would need its own BLAS, refit every frame (`ALLOW_UPDATE` + `PERFORM_UPDATE`), from a compute pass that applies
the formula above to the arena vertices (it also needs VERTEX_COLOR/NORMAL/TEXCOORD offsets from the input layout,
the wind buffer entry, per-draw cb values and the instance rows: the capture ring has room for the 16 bytes of
`wind` next to each matrix). D3D11 side at the draw (the wind buffer is a game resource) or D3D12 side from
copies; the matrices arrive 3 frames late, so a per-instance BLAS has to be keyed by the instance identity, budgeted
to the nearest N instances (forests are thousands of instances). Leaf/branch sway is a few cm, main bending grows
with height^4; for AO with a 1.5 m radius and a temporal denoiser, the rest pose is a close match except at tree
tops in strong wind, where a ray from the swaying (depth) surface can start inside or behind the rest pose leaves.
Leaves are cutouts (below), which halves that self occlusion at the default opacity.

## Alpha tested geometry (cutouts)

D3D11 has no alpha test state: the game's pixel shaders `discard`. 5171 of the 8367 dumped pixel shaders have a
`discard` (3043 of them are FX shaders, `cb_fx_clipplane`), for different reasons:
- alpha test: the diffuse map's alpha against `cb_defaultalphatest` or `cb_hairalphatest`, or against a hardcoded
  0.5 in the 60 vegetation shaders (no alpha test constant at all, so a constant name check misses foliage);
- LOD cross fade dithering (`cb_ditheringdistance`, from the pixel position), the dissolve and immolation effects
  (noise textures against a threshold): opaque meshes that must stay opaque.

So `ReflectPixelShader` (rtao_capture.cpp, at init_pipeline) scans the SHEX tokens for `discard` and, if present,
runs a small forward data flow pass over the bytecode: which temp register components derive from a read (`sample*`,
`ld`, `gather4*`) of a texture whose name contains "diffuse", "opacity" or "alpha", and whether a discard's
condition (or an enclosing `if`) does. Over the dump: 1683 non FX shaders alpha tested, 445 discard for other
reasons only, all 60 vegetation shaders with a discard alpha tested. (The operand decoder parses every instruction
of the 8367 shaders to its exact length.) Main view draws whose bound pixel shader is alpha tested are captured as
cutouts (UI "RTAO: alpha tested cutouts").

Cutouts in the trace: their TLAS instances get `kInstanceMaskCutout`, `INSTANCE_FLAG_FORCE_NON_OPAQUE` and
`kInstanceIdCutoutBit` in their InstanceID; every other instance stays `FORCE_OPAQUE` (the BLAS is shared with
opaque draws of the same mesh, built OPAQUE; the instance flag overrides it). The AO ray accepts a candidate cutout
triangle with probability "RTAO Cutout Opacity" (default 0.5), from a hash of (pixel, ray, frame, instance,
primitive): a stochastic stand-in for the texture's coverage that the temporal denoiser averages like the rest of
the ray noise. Opacity 1 traces with `RAY_FLAG_FORCE_OPAQUE` (no candidates: the plain opaque trace), opacity 0 masks
cutouts out. Standalone test (320x180, RTX 4090): ground AO next to a cutout box 0.987 / 0.801 / 0.667 at opacity
0 / 0.5 / 1, RT pass 0.144 / 0.152 / 0.144 ms; opaque geometry never produces candidates, so its cost is unchanged.

Texture exact path (not implemented): the coverage the game really uses is `diffuse.a` (times
`lerp(1, VERTEX_COLOR.a, cb)` in some shaders) at `uv * cb_layer0_scalebias.xy + .zw`, against the reference.
- Masks: per alpha tested material (key: the bound diffuse texture), a D3D11 pass renders its alpha (with the
  shader's uv scale/bias and reference baked in as 0/1, or the raw alpha + reference per instance) into a shared
  R8_UNORM strip (shared textures from D3D12 open in D3D11 only as render targets, R8/R16/RGBA8/RGB10A2 UNORM),
  and D3D12 copies the strip into a slice of a private Texture2DArray (f4-mods rtcore/alpha_atlas.h: N strips per
  frame, slices keyed by texture pointer, a counted cap). BC textures can't cross directly, hence the render pass.
- UVs: VERTEX_TEXCOORD0 offset/format from the input layout, read from the vertex arena (already shared); per
  geometry a small record (arena offsets, strides, index format, slice, uv scale/bias, reference) in a structured
  buffer indexed by InstanceID (the low 23 bits are free).
- Trace: on a cutout candidate, fetch the 3 indices and UVs, interpolate with `CandidateTriangleBarycentrics()`,
  sample the slice (point/bilinear, LOD 0 or a ray cone LOD), `CommitNonOpaqueTriangleHit()` if alpha >= reference.
  Cost: a few dependent loads per candidate; the stochastic version stays as the fallback for materials without a
  slice.

## RT sun shadows

Offline analysis of the same dump (8367 PS, 48 CS, RDEF of every shader, `fxc /dumpbin` of the key ones) plus the
strings of Dishonored2.exe (Steam 1.77.9.0). Not verified in game yet.

### How the game shades and shadows the sun/moon

The world is lit by a tiled FORWARD renderer ("tlf" = tiled light forward: CS `0x32EF950D`/`0x7576F9BC` cull
`ro_tlf_lights_sb` into per tile linked lists `ro_tlf_ll_heads`/`ro_tlf_ll_pool`), not a deferred lighting pass. The
directional light ("parallel light": sun or moon) is shaded in every forward pixel shader (3731 of them use
`cb_parallellightdir`). Its shadow is NOT computed there for opaque world geometry: it comes from a screen space
shadow mask computed before the forward pass.

1. Screen shadows pass, fullscreen PS `0xF324B47B` ("screenShadowsNoJitter.material", inputs: INTERP0 uv only) or
   `0x6C8C2397` ("screenShadows.material": same, plus SV_POSITION for a 4x4 Bayer dither that moves the shadow lookup
   toward the camera by `cb_screenshadows_stepsize * bayer / 64`). Bindings: t0 `ro_lightdepthcomparemap`
   (Texture2DArray, the cascaded shadow map, one slice per cascade), t1 `ro_env_atm_sky_cloudshadow`, t2
   `ro_screenshadows_depthbuffer` (device depth, sampled at `uv * cb_resolutionscale`, point clamp), s0
   `smp_shadowhardwarepcfsampler` (comparison), b1 PerViewCB, b2 PerInstanceCB (400 bytes):
   `cb_parallellightslicefars` @0, `cb_shadowprojections[5]` (float4x4) @16, `cb_atm_sky_sundirection` @336,
   `cb_env_atm_sky_cloudaltitude` @348, `cb_shadow_bias` @352, `cb_screenshadows_stepsize` @356,
   `cb_modeldepthhack` @360, cloud tiling/speed/rotation/density @364..384.
   - world position from depth: `cb_inverseviewprojectionmatrix` applied to (ndc.xy, depth * cb_modeldepthhack, 1).
   - cascade = number of `cb_parallellightslicefars` components <= view depth (view depth from `cb_viewmatrix`
     column z).
   - light space = `x * M[0] + y * M[1] + z * M[2] + M[3]` with `M = cb_shadowprojections[cascade]` (row vector
     convention, rows at cb2[1 + 4c .. 4 + 4c]); uv = `ls.xy * cb_parallelshadowmapslicedimensions.xy + 0.5`,
     reference = `ls.z - cb_shadow_bias`.
   - PCF: 4 bilinear `SampleCmpLevelZero` taps with tent weights, `/ 16` (a 3x3 tent over 4x4 texels).
   - cloud shadow: a ray from the point along `cb_atm_sky_sundirection` (world space, toward the light, z up) to a
     sphere at the cloud altitude, sample `ro_env_atm_sky_cloudshadow` there, `min(exp(-density * c), 1)`.
   - output: `cloud * pcf` into all 4 channels.
2. Separable bilateral blur, PS `0x9A8C5E06` / `0x9B5682FF` ("screenShadows/bilateralBlur/blurX/Y"): reads `.y` of
   `ro_screenshadows_bilateralblur_sourceimage` with linearized depth weights, writes all channels.
3. The result is `ro_texscreenshadows` (string "texScreenshadows"). The exe also has "ssaoScreenShadows" and
   "_SSAO_ScreenShadows_%p": the game's SSAO and the screen shadows share one render target, x = AO (the SSAO pass and
   the DWAO blur `0x4CD2BE39`, i.e. the target RTAO's composite writes `(ao, 1, 0, 0)` into), y = sun shadow. The
   screen shadows and its blur must then write y only (blend/write mask), which matches the forward shaders' use of
   x and y. To confirm in game (the "View AO only" path reads x; the y channel after the blur is the sun term).
4. Forward pass, opaque world shaders (1726 variants with `ro_texscreenshadows` and no cascade code; and 997 that
   also have the cascades): `ld ro_texscreenshadows[SV_POSITION.xy]`; `.y` > 0 gates the whole sun term
   (`cb_parallellightcolor * y * BRDF(N, L = cb_parallellightdir.xyz, V)`), `.x` (AO) is min'ed into the specular
   occlusion. `cb_parallellightdir` is world space, pointing TOWARD the light (it is used as L in N.L and in the half
   vector), in PerInstanceCB at a per shader offset (64 in `0x0009E597`, 736 in `0x0002C598`).
5. Shaders that compute the cascades themselves (same PCF code inline, with `ro_lightdepthcomparemap`): skin (SSS,
   `ro_translucency_mask`, ~900 variants, which also need the shadow map depth for translucency), hair (~450),
   translucent/alpha lit and FX (~600), water/ocean (~30). They read `ro_texscreenshadows.x` (AO) at most, so a
   change to the screen shadow mask does not reach them: characters' skin/hair and transparent surfaces keep the
   game's shadow maps.

Frame order, from the render pass enum strings in the exe (`ARK_RENDERPASS_*`, in enum order): DEPTHPASS,
DEPTHMINMAX, SSAO, SHADOWS_WORLD, SHADOWS_DYNAMIC, INDIRECT_LIGHTING_*, SCREENSHADOWS, TLFLIGHTCULL, FORWARD_WORLD,
FORWARD_DYNAMIC, ... So the SSAO draw (where RTAO already runs its D3D12 submission) comes before the shadow maps
and the screen shadows pass. The game records passes on several deferred contexts, but executes them in this order
(the enum order is an inference, check it in game: the RT sun stats show how old the visibility was).

### Injection

The screen shadows pass is the single point where the opaque world's sun visibility is made, as a fullscreen draw,
with the shadow map term isolated (`pcf`) from the clouds (`cloud`). Replacing its two pixel shaders gives full
control without touching any of the ~3700 forward shader permutations:
- `Shaders/Dishonored 2/ScreenShadows_0xF324B47B_0x6C8C2397.ps_5_0.hlsl`: an HLSL port of both (the dither variant
  under `SV_POSITION`), which reads the traced visibility at t3 (a free slot) and combines it according to the mode
  in LumaGameSettings: Off = the game's `cloud * pcf` (bit for bit the same math), RT = `cloud * rt`, RT x game =
  `cloud * pcf * rt`. Without a bound visibility texture (RT not ready this frame, other AO modes) it is the game's.
- the game's bilateral blur still runs over the result (a little extra softening, also over the traced penumbra).
- the trace runs in the same D3D12 submission as RTAO, right after the AO, from the same depth and TLAS, into a
  second texture shared with D3D11 (owned by the sun pass, copied inside the recorder: Interop is unchanged).

### Light direction

`cb_parallellightdir` lives in the forward shaders (per shader offsets), after the trace; `cb_atm_sky_sundirection`
is the atmosphere's sun (it may stay the sun at night while the shadow casting light is the moon). The direction
that matches the shadow maps exactly is the cascades' depth axis: the gradient of light space z,
`(M[0].z, M[1].z, M[2].z)` of `cb_shadowprojections[0]` (orthographic, so it is parallel to the light direction).
Its sign (toward or away from the light) depends on the shadow map's depth convention, so the light is taken as the
side above the horizon (world z up; a light below the horizon lights nothing), and `cb_atm_sky_sundirection` is
reported next to it. Both come from the screen shadows draw's PerInstanceCB, copied (D3D11 CopySubresourceRegion,
honouring `PSSetConstantBuffers1` offsets) into a staging ring and read back a few frames later without stalling:
the trace uses the direction from a few frames ago, which is fine for a light that moves slowly if at all.

### Local lights (not implemented)

Point/spot/area lights are in `tlf_light_tCB.co_tlf_lights[]` (float3 mWorldPosition, mRadiusFalloff, mColor,
mRadiusCutoff, mSpotDir, mDecay, ..., mAreaLightX, mShadowSetup, mFlags; 112 bytes, also `ro_tlf_lights_sb` for the
culling CS), with their shadows from `tlf_shadow_tCB.co_tlf_shadows[]` (mWorldToShadowMat, mShadowToWorldMat, mInfo)
sampling `ro_shadowmapatlas` (Texture2DArray atlas) inside every forward shader's per tile light loop. A ray traced
version would need: the light list on the D3D12 side (copy `ro_tlf_lights_sb` into a shared buffer at the cull CS,
or the cbuffer at the forward draws), a per pixel per light shadow ray (bounded by a per tile light count), and a
place to inject a per light visibility. There is no screen space mask per light: injection would mean patching the
atlas lookup in every forward shader (thousands of permutations: a generic DXBC patch of the `ro_shadowmapatlas`
sample in the style of the F4 RtShadows light_inject.h, keyed on the light index register), or a visibility buffer
of N packed lights per pixel read by that patch. Feasible but a project of its own; the sun mask is the cheap case.
