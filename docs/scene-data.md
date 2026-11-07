# Scene data and geometry

The scene module creates reproducible object descriptions and static OpenUSD geometry for a simulated seabed survey. It does not render sonar or camera frames, create the vehicle, or generate YOLO annotations.

## Local development

From the repository root, standard-library Python is enough:

```powershell
python -m sim.scene --seed 1 --scenario clear_sand --out runs/clear_sand_seed001/world.json
python -m tools.validate_run runs/clear_sand_seed001 --scene-only
python -m tools.preview_scene runs/clear_sand_seed001/world.json
```

Open the generated layout.svg in a browser for a ground-truth plan view. This is a development/spectator artifact, not a synthetic sonar image and not an agent observation. Symbols are enlarged for readability.

Additional scene configurations:

```powershell
python -m sim.scene --seed 1 --scenario half_buried --out runs/half_buried_seed001/world.json
python -m sim.scene --seed 1 --scenario murky --out runs/murky_seed001/world.json
```

Output names encode scenario and seed. The default sample layout spans x=20..70 m and y=15..45 m. A roughly 10 m-range sensor must pass near the objects. Other seeds jitter anchors and orientation. The murky setting only records the scenario name here; the sensor renderer must implement the actual turbidity. The half_buried setting changes placement and burial. Scene generation refuses to overwrite world.json unless --force is supplied intentionally.

## Data contract

world.json retains the shared seed, scenario, and objects schema. Object classes are aerial_bomb, sphere_mine, rock, and tyre. Each object has id, cls, xyz, yaw, burial. Geometry dimensions remain internal to sim/scene.py. Existing runtime consumers should not receive additional JSON keys without agreement.

Coordinates are metre-scale, Z-up; yaw is radians about +Z. This implementation defines object z as its bottom-anchor offset from a flat seabed. The build_scene seabed_z argument converts that offset to stage z. Burial is already reflected in z and is not applied twice. Water-surface depth in poses.csv requires a separate conversion using the actual sensor pose and water surface.

## Simulator integration

Start Isaac's SimulationApp before using the geometry functions. Supply a metre-scale, Z-up stage with identity parent transforms:

```python
from sim.scene import make_world, save_world, build_scene
world = make_world(seed=1, scenario="clear_sand")
object_paths = build_scene(stage, world, seabed_z=0.0)
```

The default owned subtree is /World/SurveySite. Rebuilding replaces only that subtree; put sensors and cameras elsewhere. The custom geometry attributes survey:object_id and survey:class are ground-truth mapping hooks. They do not automatically create sonar segmentation or training labels. Consumers of earlier prototype paths/attributes need to use these names. Function names and world.json fields remain unchanged.

Inspection code may call add_object with hypothetical objects based on detector position/size estimates. It must not read true object classes to choose a hypothesis. The optional semantic_labeler(prim, class_name, object_id) callback must be implemented using the actual renderer-supported annotation mechanism.

## Annotation and validation

The first integration milestone is one real sonar frame with an inspected bounding-box overlay. Obtain a sonar-aligned visible mask or validate a sonar-specific projection and visibility calculation; RGB camera boxes are not sonar boxes. Agree whether shadows belong in boxes and confirm the training class IDs. The checker assumes 0=mine_like, 1=other. Empty frames have empty label files. A visible rock/tyre gets the other label.

Once sensor outputs exist, install Pillow in the local environment if needed and run:

```powershell
python -m pip install Pillow
python -m tools.validate_run runs/clear_sand_seed001 --overlays run_qa/clear_sand_seed001
```

The full validator checks file alignment, formats, JSON/CSV structure, and normalized boxes. It cannot prove acoustic realism or correct visual annotation; inspect the overlays. Scene-only mode checks world.json without Pillow and deliberately does not require images/labels/poses. Separate training and validation by scene seed or survey pass, not adjacent random frames.

## Current verification boundary

Local JSON generation, validation, and SVG preview can be checked without Isaac. OpenUSD geometry, sensor appearance, turbidity, semantic-mask support, and sonar-label projection require the simulator runtime. A generated manifest or a passing stub pipeline is not evidence that those integrations work.
