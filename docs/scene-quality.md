# Scene component: contract audit and quality checks

Scope: the repository proof-of-concept PLAN.md provided in this conversation, particularly sections 5, 7 and 11. This is not an audit of unseen changes in the private repository. The earlier 24-hour proposal is not the implementation contract for this component.

| Requirement | Status in this component |
| --- | --- |
| sim/scene.py reusable by survey and visit jobs | build_scene and add_object provided; call sites need simulator integration |
| Four object classes | All four present as simple static visual proxies |
| world.json seed/scenario/objects and exact object fields | Preserved; no new contract fields |
| Three scenario names | Preserved: clear_sand, half_buried, murky |
| Seeded generation | Reproducible locally; same seed preserves x/y/yaw across scenarios |
| Generated seabed | Simple flat generated primitive; not a procedural terrain model |
| Partial burial | z placement generated from nominal shape height and burial fraction; rendered appearance unverified |
| Murky camera imagery | Not implemented by scene metadata; renderer responsibility |
| Sonar/camera frame sizes, naming and aligned poses | Full validator provided; real captures not supplied by this component |
| Visible per-frame sonar YOLO labels | Outstanding: requires renderer mask or validated sonar-specific projection |
| Ground truth separate from agent observations | Data API separates world data; callers must enforce the boundary |
| Clean hypothesis scenes | add_object accepts independent hypotheses; caller must not use true object class |

## Local quality gate

```powershell
python -m tools.check_scene
python -m tools.validate_run runs/clear_sand_seed001 --scene-only
python -m tools.preview_scene runs/clear_sand_seed001/world.json
```

check_scene exercises generation across 50 seeds and three scenarios. It checks repeatability, the supplied sample anchors, consistent x/y/yaw between scenarios, burial/z agreement, initial object separation and several invalid inputs. It is not a performance benchmark or an Isaac test. The source change separates layout and burial random streams; an earlier prototype could change layout in half_buried for seeds other than 1 when consuming random burial draws. Regenerate scene-only outputs after updating. Do not overwrite world definitions paired with already captured sensor datasets.

The clear_sand seed-1 configuration intentionally matches the supplied sample, including its partly buried bomb and sphere. Do not interpret clear_sand as zero burial.

The preview is an optional development artifact. Its labels and class colors come from world.json, not a detector. It must not be supplied to the live agent as sensor evidence or presented as a result of classification. Technical implementation notes belong in documentation; the preview retains a compact ground-truth badge and a normal legend instead.

## Before claiming the scene component is complete

1. Verify metre-scale Z-up geometry and the agreed bottom-anchor convention in Isaac.
2. Capture one genuine sonar/camera pair and actual pose at the same instant.
3. Produce a sonar-aligned label and inspect its overlay.
4. Verify compound-object identity, occlusion, partial burial and frame boundaries.
5. Confirm class IDs with training configuration, then generate a pilot dataset.
6. Separate train/validation by scene seed or survey pass.

P3's full scene-and-data assignment is not complete until these integration checks and labelled frames exist. Locally passing JSON checks are a useful checkpoint that can be committed with these limitations documented.

## Public project organization

Keep sim/scene.py, tools/validate_run.py, tools/preview_scene.py and tools/check_scene.py named by purpose. Run IDs encode scenario and seed. Extra layout.svg files are optional development outputs outside the required JSON contract; no existing contract paths were renamed. Documentation is separate from observations. Temporary contributor branch names can remain personal. Coordinate changes to shared PLAN.md/README with the integration owner.
