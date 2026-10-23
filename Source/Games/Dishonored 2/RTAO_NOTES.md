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
