# StableGen–Isaac Sim Bridge Document

> Scope: an evidence-backed walkthrough of the current StableGen Blender add-on, and a phased design for
> a **NVIDIA Isaac Sim 6.0** bridge that re-textures existing Isaac assets by driving StableGen's proven
> projection/bake machinery.
> Target StableGen: `0.3.1` (Blender 4.2+ / 5.1+). Target sim: **Isaac Sim 6.0 / 6.0.1** (GTC'26).
> All repo claims below cite exact files/lines; all Isaac claims cite docs in [§6](#6-references).

**The bridge in one sentence:** inside Isaac Sim, select prims and an ordered list of USD cameras, capture
RGB + depth + normals through an RTX-camera adapter, run StableGen's existing ComfyUI SDXL+depth workflow,
then hand the captures to a **headless, CPU-only Blender worker** that reuses StableGen's projection and
**bakes onto the asset's existing UVs**, returning texture files that update the original USD materials
through an override layer — with **single-GPU VRAM admission control** treated as a hard safety boundary.

Design commitments (from the reviewed plan):
- **Texturing subset only** for v1: SDXL + **depth** ControlNet first; selected-mesh descendants; explicit
  ordered camera list; bake onto existing UVs; single-GPU memory admission is first-class.
- **Isaac Sim 6.0 only** as the primary target (USD Asset Structure 3.0 / RTX sensor era); no legacy fallback.
- **Same repository**, but the Isaac-side extension and the Blender worker live in their own top-level dirs,
  keeping a clean boundary from the GPL add-on.
- **Non-goals:** robot articulation/URDF authoring, Blender camera auto-placement, and the TRELLIS.2 /
  FLUX / Qwen / Klein paths (adapters designed, implementation deferred).

---

## 1. Current StableGen architecture

### 1.1 Repository map (texturing-relevant)

| Concern | File | Key symbols |
|---|---|---|
| Panel / UI entry | `stablegen/ui/panel.py` | `StableGenPanel` |
| Texturing orchestrator | `stablegen/texturing/generator.py` | `ComfyUIGenerate` (`object.test_stable`), `Regenerate`, `Reproject` |
| ControlNet renders | `stablegen/texturing/generator.py`, `stablegen/texturing/rendering.py` | `export_depthmap`, `export_normal`, `export_canny`, `export_emit_image`, `export_visibility` |
| ComfyUI transport | `stablegen/workflows.py` | `_WorkflowBase` (`_queue_prompt`, `_execute_prompt_and_get_images`, `_flush_comfyui_vram`) |
| Workflow JSON assembly | `stablegen/texturing/workflows.py`, `stablegen/util/workflow_templates.py` | `_TexturingWorkflowMixin`, `_build_controlnet_chain_extended` |
| Image upload | `stablegen/_generator_utils.py` | `upload_image_to_comfyui` |
| Projection + visibility | `stablegen/texturing/projection.py` | `project_image`, `build_mix_tree`, `_bake_visibility_weights`, `create_native_raycast_visibility` |
| Bake | `stablegen/texturing/rendering.py` | `BakeTextures` (`object.bake_textures`), `bake_texture`, `bake_pbr_channel` |
| Paths / revision dir | `stablegen/utils.py` | `get_generation_dirs`, `get_file_path`, `get_dir_path` |
| Server health / model lists | `stablegen/core/server_api.py` | `check_server_availability`, `fetch_from_comfyui_api` |

### 1.2 End-to-end flow (what actually happens)

1. **Target & camera resolution** (`generator.py`). `ComfyUIGenerate.execute` sets the revision timestamp
   (`context.scene.output_timestamp`, generator.py ~L458–460) which drives the output tree in
   `utils.get_generation_dirs` (utils.py ~L268–276). Cameras are collected as scene objects of type
   `CAMERA`, **sorted by name**, then optionally reordered by a custom generation order
   (generator.py ~L530–555). Targets are either selected meshes or all visible meshes
   (`texture_objects`, generator.py ~L602–615).

2. **Capture / ControlNet inputs** are rendered **per camera**, each with its own resolution
   (`cameras/geometry.py` `_get_camera_resolution`, `sg_res_x/sg_res_y`):
   - **Depth** — Workbench **Z pass**, normalized + inverted → `controlnet/depth/depth_map{cam}.png`
     (`export_depthmap`, generator.py ~L1689–1777). *(There is no `distance_to_image_plane` AOV; it's a
     normalized Z-depth.)*
   - **Normal** — EEVEE normal pass over neutral `(0.5,0.5,1.0)` → `controlnet/normal/normal_map{cam}.png`
     (`export_normal`, generator.py ~L1779–1851).
   - **Canny** — Workbench grey render → OpenCV Canny → `controlnet/canny/canny{cam}.png`
     (`export_canny`, rendering.py ~L1071–1099).
   - **Context/emit + visibility** — Cycles emission re-wire for Sequential mode
     (`export_emit_image`/`export_visibility`, rendering.py ~L603–790, L1182+).
   - **Selected-mode hiding:** unselected meshes get `hide_render = True` during captures
     (generator.py ~L650–657), restored on completion.

3. **ComfyUI generation** (`workflows.py` + `texturing/workflows.py`):
   - Upload captures via HTTP `POST /upload/image` (`_generator_utils.upload_image_to_comfyui`).
   - Build the SDXL workflow from a JSON template and append a **ControlNet chain**
     (`_build_controlnet_chain_extended`, texturing/workflows.py ~L2013–2110); depth union type set at
     ~L2098–2099.
   - Queue via `POST /prompt` with `{"prompt", "client_id"}` → `prompt_id` (`_queue_prompt`,
     workflows.py ~L398–412).
   - Stream over `ws://{server}/ws?clientId=...`: `executing` (filtered by `prompt_id`), `progress`,
     `execution_error`, and **binary image frames** on the `SaveImageWebsocket` node (skip 8-byte header)
     (`_execute_prompt_and_get_images`, workflows.py ~L285–396).
   - **HTTP fallback:** an extra `SaveImage` node is injected so corrupt WS payloads can be re-fetched via
     `GET /history/{id}` + `GET /view?...` (workflows.py ~L194–219, L254–283).
   - Per-view result saved to `generated/generated_image{cam}-{material_id}-0001.png`.

4. **Projection & multi-view blending** (`projection.py`): `project_image` adds a per-camera **UV Project**
   modifier (projector = camera, aspect from per-camera resolution), bakes the projected UV into a mesh
   attribute `ProjectionUV_{i}_{mat_id}` (projection.py ~L1236–1262), loads each generated image as a
   `TEX_IMAGE` node, and `build_mix_tree` blends views with a nested `MixRGB` tree whose factors come from
   **camera visibility weights**. Visibility is a `ShaderNodeRaycast`-based occlusion test (Blender ≥ 5.1)
   or OSL `raycast.osl` (< 5.1); projection is forced to **Cycles**.

5. **Bake to UV image textures** (`rendering.py` `BakeTextures`): modal state machine
   `unwrap → bake → bake_pbr → pack_orm → apply_material`. **Unwrap is optional** (`try_unwrap='none'`)
   and existing non-projection UVs are reused when present (rendering.py ~L2437–2447); `bake_texture`
   selects the first UV layer that is not `ProjectionUV*`/`_SG_ProjectionBuffer` (~L1728–1732). Cycles
   `DIFFUSE` for base color, `EMIT`/`NORMAL` for PBR channels
   (`_PBR_CHANNEL_SUFFIXES = BaseColor/Roughness/Metallic/Normal/Emission/Height/AO`, ~L1836–1844) →
   `{obj.name}_{Suffix}.png`. CPU is forced on Blender < 5.1; GPU is auto-selected ≥ 5.1.

### 1.3 Data-flow diagram

```mermaid
flowchart TD
    subgraph BL["Blender + StableGen (host today)"]
        A[Selected meshes + ordered cameras] --> B[Per-camera captures]
        B -->|Workbench Z| D[depth_map{cam}.png]
        B -->|EEVEE normal| N[normal_map{cam}.png]
        B -->|Workbench+Canny| C[canny{cam}.png]
        B -->|Cycles emit| CTX[ctx_render / visibility]
    end
    subgraph CU["ComfyUI server (GPU)"]
        D --> W[SDXL + ControlNet-depth workflow]
        N --> W
        C --> W
        W -->|/prompt + /ws| G[generated_image{cam}.png]
    end
    subgraph PR["Projection + bake (Cycles)"]
        G --> P[UV Project per cam -> ProjectionUV_i]
        P --> M[build_mix_tree: blend by visibility weights]
        M --> BK[BakeTextures onto existing/BakeUV]
        BK --> TEX[obj_BaseColor / _Roughness / _Metallic / _Normal ...]
    end
    A -.-> BL
    TEX --> OUT[(revision dir: generated/ controlnet/ baked/)]
```

### 1.4 High-level contracts the bridge must honour

- **Targets:** a concrete set of mesh objects (selected descendants) with valid materials and a non-projection UV set.
- **Cameras:** an **ordered** list with per-camera intrinsics (focal/aperture → resolution/aspect) and world transforms; order affects Sequential inpainting.
- **Inputs:** per-camera depth (normalized Z), normal, canny, and optional context/visibility renders, named deterministically under a timestamped revision dir.
- **Outputs:** `generated/generated_image{cam}-{mat}-0001.png` per view; baked `{obj}_{Suffix}.png` maps.
- **Engine assumptions:** Cycles for projection/emit/bake; OSL + CPU when Blender < 5.1; EEVEE/Workbench for normal/depth/canny.

### 1.5 Selected-object visibility edge case (call-out, not a requirement)

There is an **asymmetry** between capture and visibility weighting:

- During captures in selected mode, unselected meshes are hidden with **`hide_render = True`**
  (generator.py ~L650–657).
- But `_bake_visibility_weights()` builds its occlusion **BVH from all viewport-visible meshes**
  (`not obj.hide_get()`), explicitly ignoring `hide_render` (projection.py ~L94–118), to match the runtime
  `ShaderNodeRaycast` behaviour which tests all visible scene geometry.

Consequence: geometry excluded from the render can still occlude/weight the projection. This is **inconsistent
edge behaviour to be aware of**, not something the bridge needs to reproduce. The bridge sidesteps it by
isolating exactly the target set in a temporary stage/scene (both capture and projection see the same
geometry).

---

## 2. Chosen bridge architecture

### 2.1 Isaac Sim 6.0 Kit extension (host)

An `omni.ext`-based extension `stablegen.bridge` with an `omni.ui` dockable panel that mirrors the
**texturing subset** of StableGen's panel:

- **Targets:** resolve from the **current selection**; two modes — *all `UsdGeom.Mesh` descendants* under the
  selected prims, or *only the first mesh descendant*.
- **Cameras:** an **explicit, ordered list of USD cameras** (add/remove/reorder), read from the stage. No
  auto-placement in v1 (deferred; StableGen's placement strategies are a Phase 3 port).
- **Prompt controls:** prompt / negative prompt, seed, resolution, SDXL checkpoint, LoRA, ControlNet units
  (depth first), projection/bake controls, advanced sections. Server + model refresh reuse ComfyUI's
  `/object_info/*` listing (as `core/server_api.py` does today).
- **Run UI:** progress / phase / per-step bars, cancel (maps to ComfyUI `/interrupt` used at generator.py
  ~L933), and explicit error surfaces.
- Deferred: TRELLIS.2 and all non-texturing UI; FLUX/Qwen/Klein controls hidden behind the model adapter.

### 2.2 Target isolation via a temporary USD overlay

To capture and project **exactly** the chosen targets (and avoid §1.5's asymmetry), the extension composes a
**temporary anonymous overlay layer** on the stage that isolates the target meshes plus the required cameras
and lights, and restores the previous state in a `finally` block. Geometry is **not** round-tripped back into
the source stage; the overlay is purely for deterministic capture/visibility.

### 2.3 RTX-camera capture adapter

Capture **RGB**, **`distance_to_image_plane`** (depth), **normals**, and instance/depth-derived **masks**
through an **Isaac 6 RTX camera compatibility adapter**. Because the recommended 6.0 RTX sensor package is
still evolving, the adapter keeps **direct Replicator annotators as a fallback** behind a stable interface.
The adapter is responsible for converting Isaac captures into the exact tensors/paths StableGen expects:
- depth → the same **normalized, inverted** encoding StableGen's `export_depthmap` produces;
- normals → StableGen's neutral-background `(0.5,0.5,1.0)` convention;
- **canny is derived locally** from the RGB/context render (as StableGen does with OpenCV), not requested from
  Isaac.

Camera intrinsics (focal length, horizontal aperture, clipping) and transforms are exported alongside so the
Blender worker can reconstruct identical cameras.

### 2.4 ComfyUI adapter (no `bpy` in Kit)

Reuse StableGen's ComfyUI **wire and file contracts** without importing any `bpy`-dependent module into Kit.
Port the pure-Python pieces of `workflows.py` (`_queue_prompt`, `_execute_prompt_and_get_images`, WS/HTTP
handling, `_flush_comfyui_vram`) and the SDXL+depth JSON assembly into a Kit-safe module:
- `POST /upload/image` for captures; `POST /prompt` (`client_id`); stream `/ws`; `GET /history` + `/view`
  fallback; `POST /interrupt` for cancel; `POST /free` + `GET /system_stats` for VRAM (see [§4](#4-gpuvram-safety-design)).
- **Model adapters are designed now** (SDXL implemented) with stubs for FLUX.1 / Qwen / FLUX.2 Klein so the
  workflow builder can be swapped without touching the extension.

### 2.5 Versioned job manifest & shared workspace

Define a **versioned job manifest** written to a shared workspace directory (visible to both the Kit extension
and the Blender worker), containing:
- target **identities** and **topology/UV checksums** (to detect stage drift between capture and bake);
- ordered **camera transforms + intrinsics**;
- capture image paths, generated image paths, projection/bake settings;
- output texture paths and per-stage diagnostics/timings.

Draft schema:

```json
{
  "schema_version": "1.0",
  "isaac_sim": "6.0.1",
  "stage": "omniverse://.../scene.usd",
  "targets": [
    { "prim": "/World/Chair", "mesh": "/World/Chair/Geom/mesh",
      "topo_hash": "sha1:...", "uv_hash": "sha1:...", "uv_set": "st",
      "material": "/World/Chair/Looks/Chair_mat", "material_type": "UsdPreviewSurface" }
  ],
  "cameras": [
    { "index": 0, "prim": "/World/Cam_00",
      "xform": [[...4x4...]], "focal_length": 24.0, "horizontal_aperture": 20.955,
      "resolution": [1024, 1024], "clip": [0.01, 1000.0] }
  ],
  "generation": { "model": "sdxl", "checkpoint": "...", "prompt": "...", "negative": "...",
                  "seed": 0, "controlnets": [ { "type": "depth", "strength": 0.8 } ] },
  "workspace": "/shared/sg_jobs/2026-07-30T03-18-00/",
  "outputs": { "textures": [], "diagnostics": "diag.json" }
}
```

---

## 3. Blender projection and UV-preserving return path

### 3.1 Disposable USD snapshot (targets + cameras only)

Export a **throwaway USD snapshot** containing only the selected target meshes and the ordered cameras at a
**single frozen time sample**. The snapshot is consumed by the Blender worker and discarded; **Blender geometry
is never merged back into the source Isaac stage** (only texture paths are updated, §3.4).

### 3.2 What must be preserved (and what is rejected/deferred)

Preserve exactly, so the bake lands on the original parametrization:
- `primvars:st` (the UV set), **face-corner indexing**, topology, material subsets/`GeomSubset`s,
  transforms, camera focal/aperture, **stage units** (`metersPerUnit`) and **up axis** (`upAxis`).

Explicitly **reject or defer** until validated (surface a clear error rather than silently mis-bake):
- ambiguous **overlapping UVs**, arbitrary/custom shader graphs, **per-instance material overrides**, and
  fragile **UDIM / skinned** workflows.

Topology and UV **checksums from the manifest are re-validated** on import; a mismatch aborts the job (the
stage changed between capture and bake).

### 3.3 Headless, CPU-only Blender worker

Run a **pinned Blender version headlessly** (`blender -b -P bridge_worker.py`) that:
1. imports the USD snapshot and validates topology/UV hashes;
2. recreates the scene properties + deterministic file names StableGen expects (revision dir, per-camera
   resolutions, generated-image names);
3. invokes the **existing** camera-projection infrastructure (`project_image` / `build_mix_tree`) with the
   generated per-view images;
4. runs `BakeTextures` with **`try_unwrap='none'`** so it bakes onto the **imported original UV map** — no
   unwrap, no topology change (leveraging rendering.py ~L2437–2447 and the non-projection-UV selection at
   ~L1728–1732).

**CPU is forced for both projection and bake** on the single-GPU baseline (StableGen already forces CPU for
< 5.1; the worker forces `cycles.device='CPU'` unconditionally). This keeps the GPU reserved for Isaac +
ComfyUI (see [§4](#4-gpuvram-safety-design)).

Outputs are **immutable / versioned** base-color texture files plus a result manifest. PBR channels
(`_Roughness/_Metallic/_Normal/...`) are produced **only when explicitly requested** (generated or decomposed),
matching StableGen's optional PBR path.

### 3.4 USD material update via an override layer

Update **only texture asset paths** on the original Isaac materials, through a **dedicated USD override layer**
(never editing the base asset in place):
- **Adapters** for the supported material graphs: **UsdPreviewSurface** (set the `file` input on the
  `UsdUVTexture` feeding `diffuseColor`, with correct `sourceColorSpace` = sRGB for base color / raw for
  data maps) and **OmniPBR MDL** (`diffuse_texture`, `reflectionroughness_texture`, `metallic_texture`,
  `normalmap_texture`, etc.).
- Always write **new filenames** (versioned) to defeat Omniverse's texture cache, so the viewport reliably
  hot-reloads the new look.
- Materials/graphs that don't match an adapter are **refused** with a clear message (deferred to Phase 3).

---

## 4. GPU/VRAM safety design

Single-GPU is the baseline and **memory admission is a hard safety boundary**: the bridge refuses to start a
job it cannot prove will fit, rather than risking an OOM that takes down Isaac.

### 4.1 The explicit single-GPU sequence

1. Unload idle ComfyUI models up front.
2. **Pause Isaac** for state consistency during capture.
3. Capture all target views through the RTX adapter.
4. **Drain / detach / destroy extension-owned render products** (release annotator buffers).
5. **Verify free VRAM** via NVML + ComfyUI `/system_stats` before generation.
6. Run **one** ComfyUI job.
7. Call `/free` and **verify reclamation** (poll `/system_stats`).
8. Run the **CPU-only** Blender worker (projection + bake).
9. Apply textures via the override layer and restore Isaac state.

StableGen already implements steps 7's mechanics: `_flush_comfyui_vram` sends `POST /free`
(`unload_models`+`free_memory`), clears history/queue, sends `/free` again, and **polls
`GET /system_stats` for `vram_free`/`vram_total`** until VRAM rises or a threshold is met
(workflows.py ~L82–172, `_get_vram_stats` ~L68–80). The bridge reuses this verbatim.

### 4.2 Honest statement of what pausing does *not* free

Pausing Isaac **does not release** its renderer, stage, texture, physics, or CUDA allocations, and a paused
Isaac can keep rendering asynchronously. Tearing down render products **may reduce** usage but is **not a
guaranteed release**; **process exit is the only hard reclamation boundary**. The design states this plainly
and never assumes a pause frees VRAM.

### 4.3 NVML-based admission control & failure policy

- Maintain a **profiled table of per-model / per-resolution peak VRAM** plus a safety margin.
- Before generation, read free VRAM (NVML + `/system_stats`) and **refuse** the job if
  `free < profiled_peak + margin`. Offer concrete remedies: lower resolution / lighter model, or optional
  **multi-GPU / remote ComfyUI**.
- On OOM, permit **exactly one degraded retry** *after* a full cleanup; on ComfyUI unload timeout, **restart
  ComfyUI** as the reliable reclamation path.

### 4.4 Optional multi-GPU pinning

- Pin Isaac renderer/physics to one GPU and ComfyUI to another via **UUID-based `CUDA_VISIBLE_DEVICES`**.
- Isaac multi-GPU is **explicitly bounded** (not a free scaling knob); the Blender worker stays **CPU-only**.
- Note plainly: **GPU VRAM is not pooled** across devices — pinning partitions, it does not aggregate.

### 4.5 Empirical profiling matrix

Profile (don't guess) across: scene size, active annotators, model family, image resolution, bake resolution,
teardown stage, and repeated cycles, on **16 / 24 / 48 GB** hardware. Avoid claiming a universal minimum beyond
official Isaac Sim / model requirements; publish the measured table with the extension.

| Axis | Sample values |
|---|---|
| Scene size | small prop / room / large environment |
| Annotators | RGB only / +depth / +normal / +instance |
| Model | SDXL base / SDXL+LoRA |
| Image res | 768 / 1024 / 1536 |
| Bake res | 1k / 2k / 4k |
| Teardown | none / render-product drain / `/free` / ComfyUI restart |
| Cycles | 1 / 5 / 20 repeats (leak detection) |
| Hardware | 16 GB / 24 GB / 48 GB |

---

## 5. Phases and verification

### Phase 0 — Parity & isolation (no AI)
Prove camera/coordinate parity (Isaac camera → USD snapshot → Blender camera renders match), target isolation
via the temporary overlay, **UV/topology preservation** through the round-trip, and a **checkerboard bake**
onto existing UVs — all without ComfyUI. *Exit:* a known-good asset re-textured with a test pattern, pixels
landing on the correct UV islands.

### Phase 1 — SDXL + depth vertical slice
End-to-end for **static, non-instanced, single-UV** selected meshes with supported materials: capture depth,
run SDXL+depth via the ComfyUI adapter, project, **bake onto existing UVs (CPU)**, update material via override
layer. *Exit:* the recommended **first vertical-slice milestone** — one chair/prop re-textured in Isaac from a
prompt.

### Phase 2 — Robustness & PBR
Add normal + canny units, multiple cameras / multiple descendant meshes, **PBR outputs**, stronger material
adapters (more UsdPreviewSurface/OmniPBR graph shapes), NVML admission control, automated recovery, and the
profiling matrix.

### Phase 3 — Breadth
FLUX / Qwen / Klein model adapters; camera **auto-placement** ported from StableGen concepts; explicit
UDIM / skinning / instancing policies; optional **remote / multi-GPU** workers.

### 5.1 Test matrix
- **Pixel / reprojection parity:** re-projected captures match Isaac renders within tolerance.
- **UV correctness:** seam/orientation correctness; no unwrap or topology change during bake.
- **Visibility restoration:** stage/selection/visibility restored after `finally` on success **and** failure.
- **Hot reload:** repeated texture updates reload in-viewport (new filenames defeat cache).
- **Cancellation** at every phase (capture, ComfyUI `/interrupt`, worker, apply) leaves a clean state.
- **Failure injection:** Blender worker crash, ComfyUI disconnect, WS-corruption → HTTP fallback, VRAM
  admission refusal, OOM degraded-retry, unload-timeout ComfyUI restart.
- **Source invariance:** confirm **geometry, UVs, and physics opinions on the source stage are unchanged** —
  only texture asset paths differ, and only in the override layer.

### 5.2 Risks
- RTX sensor package churn in 6.0 → adapter + Replicator fallback.
- Material-graph diversity beyond the adapters → refuse-and-report, expand coverage in Phase 3.
- Coordinate/units mismatches (Isaac Z-up, meters vs Blender) → preserve `upAxis`/`metersPerUnit`, verify in Phase 0.
- VRAM estimation error → conservative margins + refuse-first policy + ComfyUI restart escape hatch.
- Stage drift between capture and bake → topology/UV checksum gate.

### 5.3 Explicit non-goals (v1)
Robot articulation / URDF joints; Blender camera auto-placement; TRELLIS.2 / FLUX / Qwen / Klein
implementation; editing the source asset in place; multi-GPU as a required dependency.

### 5.4 Open validation items
Exact 6.0 RTX depth/normal encodings vs StableGen's; UsdPreviewSurface ↔ OmniPBR color-space round-trip
fidelity; instanceable-asset override semantics; headless Blender USD-import fidelity for `GeomSubset`
material bindings; NVML peak-vs-margin calibration per hardware tier.

---

## 6. References

**Isaac Sim / Omniverse / OpenUSD**
- Isaac Sim 6.0 GTC'26 announcement (RTX sensors, USD Exchange SDK, multi-physics):
  https://forums.developer.nvidia.com/t/announcement-isaac-sim-6-0-early-developer-release-for-gtc26/363709
- Isaac Sim — Asset Structure (USD Asset Structure 3.0): https://docs.isaacsim.omniverse.nvidia.com/6.0.1/robot_setup/asset_structure.html
- Isaac Sim — Conventions (Z-up, meters, right-handed): https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
- Isaac Sim — Scene Setup Snippets (materials/textures: UsdPreviewSurface, OmniPbrMaterial): https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/environment_setup.html
- Replicator annotators (RGB / distance_to_image_plane / normals / instance): https://docs.omniverse.nvidia.com/py/replicator/1.11.0/source/extensions/omni.replicator.core/docs/API.html
- OpenUSD — UsdPreviewSurface spec: https://openusd.org/release/spec_usdpreviewsurface.html
- Kit — `omni.usd` material commands (CreatePreviewSurfaceMaterialPrim / BindMaterial): https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands.html
- Kit extension authoring (`omni.ext`, `omni.ui`): https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_basic.html

**In-repo grounding (evidence)**
- Orchestrator / camera order / target resolution / render hiding: `stablegen/texturing/generator.py` (`ComfyUIGenerate`, ~L458–460, L530–555, L602–615, L650–657).
- ControlNet renders: `stablegen/texturing/generator.py` (`export_depthmap` ~L1689–1777, `export_normal` ~L1779–1851); `stablegen/texturing/rendering.py` (`export_canny` ~L1071–1099, `export_emit_image` ~L603–790).
- ComfyUI transport + VRAM flush: `stablegen/workflows.py` (`_queue_prompt` ~L398–412, `_execute_prompt_and_get_images` ~L285–396, `_flush_comfyui_vram` ~L82–172, `_get_vram_stats` ~L68–80).
- Workflow JSON + ControlNet chain: `stablegen/texturing/workflows.py` (`_build_controlnet_chain_extended` ~L2013–2110); templates in `stablegen/util/workflow_templates.py`.
- Upload: `stablegen/_generator_utils.py` (`upload_image_to_comfyui`).
- Projection + visibility BVH edge case: `stablegen/texturing/projection.py` (`project_image` ~L792+, UV Project ~L1236–1262, `_bake_visibility_weights` BVH ~L94–118, native raycast ~L482–509).
- UV-preserving bake: `stablegen/texturing/rendering.py` (`BakeTextures` ~L2185+, optional unwrap ~L2437–2447, non-projection UV selection ~L1728–1732, `_PBR_CHANNEL_SUFFIXES` ~L1836–1844, CPU/GPU device ~L1533–1570).
- Paths / revision dir: `stablegen/utils.py` (`get_generation_dirs` ~L268–276, `get_file_path` ~L333–412).
- Server health / model lists: `stablegen/core/server_api.py`.
