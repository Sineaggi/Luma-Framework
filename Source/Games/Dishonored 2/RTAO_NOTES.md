# RTAO: ray traced ambient occlusion for Dishonored 2

## Summary

Luma's "Ambient Occlusion" setting picks one of: Off, HBAO+ (the game's own, through its r_hbao cvar), XeGTAO, or
RTAO. RTAO replaces the game's SSAO pass with AO ray traced on a side D3D12 device (DXR 1.1 RayQuery, same GPU,
shared textures + a shared fence), against geometry captured from the game's own draws:

- **Scene:** static meshes (all matrix families, including MVP-only props), skinned characters and bone-animated
  props, and meshes with CPU-written (DYNAMIC) vertex buffers, both rebuilt and refit every frame. Wind vegetation
  is in its rest pose. Alpha-tested cutouts are partly transparent. Off-screen static geometry is retained for 180
  frames once seen in 4. Geometry drawn after the AO pass, the UI, particles, water and the first-person view model
  are left out.
- **Pass:** cosine-distributed rays from the depth-reconstructed position, then a cross-bilateral + temporal
  denoiser (ported from the user's Fallout 4 RtShadows), then a power curve (Strength).
- **Cost (RTX 4090, 3440x1440):** about 0.8 ms indoors and 2.3 ms on the Karnaca docks before the performance work.
  The quality modes / dedup aim at about 1 ms there (not yet measured in game).

Panel (Luma settings, "Ambient Occlusion" section) and config keys (ReShade.ini, `[Luma]`):

| UI | Key | Default | Notes |
| --- | --- | --- | --- |
| Ambient Occlusion | AOMode | 1 (XeGTAO) | 0 Off, 1 XeGTAO, 2 RTAO, 3 HBAO+; the old XeGTAOEnable is migrated |
| View AO only (debug) | (not saved) | Off | Final AO (after the game's blur, any mode), or RTAO output |
| RTAO Output | RTAOTestOutput | 0 | Ray traced AO, test pattern, depth test, BLAS coverage, no denoiser |
| RTAO Strength | RTAOStrength | 2 | Power curve (>1 darker) |
| RTAO Radius / Rays | RTAORadius / RTAORays | 1.5 / 4 | World units / rays per traced pixel |
| RTAO Quality | RTAOQuality | 0 Low | Low: 1 of 4 pixels traced per frame + upsample; Medium: 2 rays; High |
| RTAO Capture | RTAOCapturePass | 1 | Depth-only draws, or all depth-writing draws |
| RTAO: skinned characters / dynamic meshes | RTAOSkinned / RTAODynamic | on | Per-frame BLAS |
| RTAO: wind meshes / alpha tested cutouts | RTAOWindMeshes / RTAOCutouts | on | |
| RTAO Cutout Opacity / fast cutouts | RTAOCutoutOpacity / RTAOFastCutouts | 0.5 / off | |
| RTAO: skip repeated draws | RTAODedup | on | One capture per object per frame |

Known limits: instanced skinned draws (crowds) and GPU-culled indirect draws are not captured (0 seen so far).
Wind sway isn't replayed. Cutouts are statistically, not texture, exact (work in progress). A moving object only
occludes from the frames it is drawn in (no off-screen retention). Needs a DXR 1.1 GPU; it falls back to XeGTAO
otherwise. Tested on an RTX 4090 only. The standalone GPU test (`Tests/rtao_interop_test.cpp`) covers the interop,
capture, BLAS/TLAS, trace and denoiser offline.

# Geometry capture in Dishonored 2 (Void Engine)

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

## Performance (rtao-perf)

In game (Karnaca docks, 3440x1440, RTX 4090) the RT pass took 2.29 ms (indoors ~0.8). The standalone benchmark
(`Tests/rtao_interop_test.exe bench`: a street of 2821 instances, 652 of them foliage cutouts of 250 to 1200 overlapping
leaf cards, 1M triangles, depth rasterized by D3D11, medians over 60 frames) splits the old pass (High, 4 rays) into
TLAS build 0.17 ms, trace 0.86 ms (0.69 with cutouts forced opaque, 0.50 with them masked out: the per crossing
candidate loop costs ~0.2 to 0.4 ms there), denoise 0.18 ms.
- TLAS build: ~0.03 ms fixed + ~0.05 ms per 1000 instances; neither FAST_TRACE nor a default heap copy of the instance
  descs helps. Not touched (distance culling would need instance bounds the CPU doesn't have).
- Denoise: the 3x3 neighbourhood (raw value + world position from depth) is loaded once per 8x8 group into groupshared
  memory instead of 9 times per pixel: 0.18 -> 0.14 ms, same output.
- "RTAO Quality" (config `RTAOQuality`, default Low): Low traces "RTAO Rays" at one pixel per 2x2 block (position
  cycling over 4 frames, `kTraceJitter`), and the denoiser's bilateral (plus a 1.5 px gaussian) upsamples it; Medium
  traces every pixel with half the rays; High is the old pass. Bench totals: High 1.22 ms, Medium 0.73, Low 0.53. Mean
  abs error of the denoised AO against a 128 ray reference: High 0.0203 static / 0.0251 after a camera move, Medium
  0.0224 / 0.0272, Low 0.0205 / 0.0250 (and the least frame to frame flicker): the denoiser hides the lower ray count,
  Low only looks a bit softer on fine detail. Debug outputs (raw, coverage) always trace at full resolution.
- "Fast cutouts" (config `RTAOFastCutouts`, default off): one coin per ray instead of one per crossed cutout triangle
  (the ray treats all cutouts as solid with probability 1 - (1 - opacity)^2, else masks them out), so traversal never
  returns candidates: trace 0.86 -> 0.64 ms at High, 0.22 -> 0.17 at Low. Error 0.0203 -> 0.0250: thin single layers
  occlude a bit more, dense canopies a bit less than per crossing.
- GPU time split: GpuTimer::Mark; the UI ("RTAO stats") and the periodic log show capture / TLAS / trace / denoise.

### Duplicate draws (dedup, config `RTAODedup`, default on)

With "All depth writing draws", an object drawn by the depth prepass and again by a depth writing G-buffer pass was
captured twice: two matrix dispatches on the game's thread, two readback entries (merged by the TLAS key anyway) and,
for skinned/dynamic meshes, two skinning dispatches and two per frame BLASes at the same place. A draw is now skipped if
one with the same identity was captured earlier in the frame: mesh key + bound PerInstanceCB (object, first/num
constants) + instance buffer and index SRVs + PerViewCB (MVP family) + bone palette SRV + instance count + cutout flag +
a binding version. The version guards against a buffer the game rewrites between draws (a small per draw cbuffer
updated with Map DISCARD would otherwise make every draw of the mesh look the same): main.cpp's map hook gives each
Map(WRITE_DISCARD) a unique id, kept per recording thread, and the draw mixes in the ids of its vertex buffer, cbuffers
and the resources behind its SRVs. Not covered: buffers rewritten with UpdateSubresource/copies, or by another thread
between the two draws (a ring wrapping mid frame); the worst case is an object missing for that frame (retention hides
it for static objects). Counted in "RTAO stats" and the log ("per frame duplicates skipped").
