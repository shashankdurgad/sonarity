# Simulator integration patch

This patch changes only sim/scene.py, tools/check_scene.py, adds tools/check_scene_runtime.py, and this note. It does not touch the simulator owner's session, survey, objects, sonar geometry, or infra files. Keep the shared world.json schema unchanged.

- displayColor is authored with constant interpolation (one value per entire primitive), including the 384-vertex tyre mesh.
- build_scene has the new keyword-only include_seabed: bool = True. Passing False omits the flat floor; the simulator owns its textured/labelled floor.
- DIMENSIONS_M provides exact local length/width/height. HEIGHTS_M is derived from it. The bomb is centred on its complete footprint: body centre x=0.05, fin centres x=-0.60, extents x=[-0.8,0.8], y=[-0.35,0.35], z=[0,0.5]. Geometry extents are authored explicitly.
- Training seeds other than 1 generate 4-6 objects, random classes (at least one mine-like and one other), centres in [2,30] metres and minimum 6 m centre-to-centre spacing. The margin keeps these object footprints within approximately [0,32].
- Seed 1 deliberately preserves runs/sample layout for compatibility; it is the exception to compact bounds and can require a longer survey. Use training seeds 10-19 or the new tagged training seeds for compact layouts.
- clear_sand and murky share identical object data; camera turbidity remains the sensor renderer's responsibility. half_buried keeps the same x/y/yaw/classes but changes burial/z.
- add_object retains its signature. IDs such as o7 validate. It creates only its requested object and calls the semantic_labeler on all geometry parts. It does not prove that a live OceanSim capture works without a runtime test.

## CPU check

python -m tools.check_scene

## Runtime check, owned by the existing configured simulator session

Inside an already initialised Isaac session, call:

    from tools.check_scene_runtime import check_runtime
    check_runtime(stage, existing_semantic_labeler, update=existing_update_function)

Replace the callback names with the real session's functions; no names are assumed here. The diagnostic uses /World/SceneIntegrationCheck plus a temporary /World/SurveySite/Objects/o7 and cleans them up. It refuses to replace an existing o7. Run only when no survey capture is in progress. It tests actual USD bounds and constant color plus adding o7 after scene construction with the supplied real callback. It cannot confirm sonar visibility without capturing a real frame. CPU checks alone are not a claim that item 5 passed in OceanSim.

## Data QA

Download two or three existing uxo-runs surveys into runs/ using the configured Modal workspace. Run tools/validate_run.py with --overlays to an output folder outside the run. Review sonar/ PNGs against overlay boxes. Report run ID, frame, affected object/label and whether a box misses a return, includes background, or is absent. Do not relabel data automatically from this patch's new generator; old surveys were rendered from their saved world.json.

## Integration order

Push to p3/scene-data. Do not merge main yet. Wait for the simulator-side commit on main and the simulator owner's confirmation, then rebase with conflict review and run the requested quick GPU survey before batch generation. Do not overwrite the other owner's files to resolve conflicts.

This integration note supersedes the earlier fixed-four-object training and 50-seed-check descriptions in prototype documentation.
