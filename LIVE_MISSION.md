# Live mission: the robot reacts while it runs

**Status:** proposed by P2 on Sat 7 Nov, late evening. The team works toward this after Gate B is saved. **The batch pipeline in PLAN.md stays as the baseline and safety net. Don't break it.**
**Deadline:** submit by 14:00 on Sunday 8 Nov, so we can work through the night and the morning.

## Why

The batch pipeline (survey → detect → imagine → planner, each run after the previous one finishes) can't react to anything. The robot mows fixed lanes, and every decision happens after the run. Three judging criteria need the robot to act *during* the run:

| Criterion | Weight | What the judges want to see |
|---|---|---|
| Perception-driven action | 25% | "Every decision traces to what it perceived, and it corrects mid-run" |
| Adaptation to change | 25% | "The scene changes mid-sequence and the agent visibly re-plans", with an unscripted change in one continuous take |
| Relevance: critical failure mode | 20% | "The most critical failure mode handled on screen" |

**Critical failure mode:** a survey robot passing close to a mine. Influence mines go off from a vessel's magnetic, acoustic or pressure signature, so it doesn't need to touch one. In the batch pipeline, a bomb exposed mid-survey would have the robot fly about 1.5 m over it and only flag it afterwards.

## The hero sequence (one continuous take)

1. The robot surveys lanes. The sonar, robot camera, chase view and site map are all on screen.
2. **A person triggers a change at a moment of their choosing**, e.g. "a bomb is uncovered" (scour exposes a buried bomb ahead in the current lane). The robot is never told; it only finds out through its sonar.
3. The sonar detects a `mine_like` contact ahead → the robot **stops** before reaching it.
4. It draws a **keep-out circle** around the contact and **re-plans its remaining path around the circle**. The map shows the old route greyed out with a "would have passed 1.6 m over a mine" warning, and the new route on top.
5. It imagines bomb vs rock from 4 angles, picks the clearest angle **outside the circle**, moves there and inspects.
6. It gives a verdict (UXO, priority, reason) and **resumes the survey** on the new route.
7. Optional second change: the water goes murky. The robot notices the camera has lost contrast, switches to sonar only, and marks later contacts `uncertain` with a revisit, instead of guessing.

The keep-out radius is **scaled to the 32 m test site (about 4 m)**. Say so on screen; real standoffs are far larger.

## How it runs

```
modal: infra/mission.py ── one Isaac session (SimSession) on an L40S, the whole run
  loop every step:
     sonar, cam = sim.observe(); chase = sim.chase()
     dets = detect_frame(sonar, pose)              # P1
     tracker.update(dets)                          # P1: contacts build up as the robot moves
     events = poll Modal Dict "uxo-mission-events" # P2: changes a person sends live
     state, path = mission_step(...)               # P2: SURVEY / HOLD / REPLAN / INSPECT / RESUME
       on a new mine_like contact near the path:
          view = imagine_contact(contact)          # P5: pick the angle (outside the keep-out circle)
          sim.move_to(inspection pose); sonar, cam = sim.observe()
          target = classify_contact(contact, inspection images)   # P4
     log one line to mission_log.jsonl; save frames
  at the end: contacts.json, viewpoints.json, targets.json, world.json (written at the END, so it includes injected objects)

laptop: python -P -m modal run infra/mission.py::trigger --event uncover_bomb   # sends a change while it runs
```

**Rule (as in PLAN.md):** the robot's decisions use only what `observe()` returns. Never read `world.json` or `sim.world` in decision code; those are ground truth, for labels, scoring and rendering.

## What each person needs to provide

Everything below is **new functions next to the existing batch code**. Keep the batch `run(run_dir)` entry points working.

### P2 (sim, mission loop)
- `infra/mission.py`: the Modal job with one Isaac session for the whole run, the `uxo-mission-events` Modal Dict, and a `trigger` entrypoint with events `uncover_bomb`, `drop_object`, `murk_on` and `murk_off`. Positions are picked from the robot's current lane with some randomness, so the change isn't pre-placed.
- `sim/mission.py`: the state machine (SURVEY → HOLD → REPLAN → INSPECT → RESUME), a path re-planner that detours around keep-out circles, and `mission_log.jsonl` (one JSON line per step: `t, pose, state, planned_path, contacts, keep_out, event, decision_reason`).
- Write `world.json` at the end of every run, including survey.py (bug: an object added mid-run is missing from it today).
- Stand-ins for the other functions so the loop runs before the real ones arrive.
- Stretch: a laptop viewer that shows the latest frames while the mission runs.

### P1 (detector)
- `perception/detect.py: detect_frame(sonar: np.ndarray (512×512 uint8), pose: dict) -> list[dict]`. Each dict: `{"xy": [x, y], "size_m": [l, w], "cls": "mine_like"|"other", "conf": float, "box": [x0, y0, x1, y1]}`. World xy via `sim.sonar_geometry.pixel_to_world`.
- `ContactTracker` (same file): `update(dets, frame_name) -> list[new_or_updated contacts]`, using the same merge logic as `merge_detections` but incremental. A contact counts as confirmed after ≥ 2 hits. Contacts follow the `contacts.json` format, so `id` is `c1`, `c2`, …
- **Decide where YOLO runs.** Inside Isaac's Python (check `ultralytics` installs without breaking Isaac's own torch), or as a separate Modal function called each frame (about 0.1–0.3 s is fine). Tell P2.

### P4 (planner, integration)
- `agent/planner.py: classify_contact(contact: dict, inspection_img: str | np.ndarray, viewpoint: dict | None) -> dict` returning one `targets.json` entry. Use the template fallback if Gemini fails or takes more than about 8 s, because the robot is holding position while it waits.
- **Autonomy (15%):** the judges score remote calls lowest ("keep it on the device"). The template or local rule must make the decision; Gemini can at most write the `reason` text. Say that in the technical summary.
- Mission metrics in `viz/metrics.py`: minimum distance from the robot to each mine (old path vs new path), time from change to re-plan, and contacts found.
- Keep owning `run_demo.py` and the batch baseline.

### P5 (imagination, video)
- `agent/imagine.py: imagine_contact(contact)` already works per contact. Add an optional `keep_out_m` argument so it only picks angles whose inspection pose is outside the circle, and have it return the chosen pose `(x, y, yaw)` as well as the direction.
- `viz/compose.py`: a **mission layout** read from `mission_log.jsonl` and the saved frames. It shows the chase view, sonar with detection boxes, the robot camera, and a **site map** (path driven so far, old route greyed out with a warning, new route, keep-out circles, contacts coloured by verdict). Add a state banner ("SURVEYING", "CONTACT AHEAD: HOLDING", "RE-PLANNING", "INSPECTING FROM NORTH", "RESUMING") and a caption when a change event arrives.

### P3 (scene, scenarios)
- Pick 1–2 **mission seeds** where the hero sequence reads clearly: room to detour, and the bomb event lands in an open stretch of a lane.
- `sim/scene.py: make_event_object(cls, xy, yaw, burial) -> dict`, a valid `world.json` object with the next free id, for P2's trigger.
- QA: `tools/validate_run.py` on mission outputs (same contract files plus `mission_log.jsonl`).

## Order of work (milestones)

| # | Milestone | Who | Done when |
|---|---|---|---|
| M0 | Gate B batch baseline saved | P4 + all | `demo.mp4` from a real OceanSim run committed |
| M1 ✅ | Loop runs with stand-ins | P2 | A mission run with a live-triggered `uncover_bomb`: the robot holds, detours and resumes, with `mission_log.jsonl` written. Stand-in detector and planner |
| M2 | Real detector in the loop | P1 + P2 | `detect_frame` + `ContactTracker` replace the stand-ins; the bomb is detected live |
| M3 | Real imagine + classify in the loop | P5 + P4 + P2 | Inspection from the chosen angle outside the circle; verdict from `classify_contact` |
| M4 | Mission video | P5 + P3 | The composed mission view from one continuous run, with a human-triggered change |
| M5 | Record final takes | all | 2–3 takes; pick the best |

M1 doesn't wait on anyone, and P1, P4 and P5 can write their functions against the formats above in parallel. Post in the group chat if you need to change any format here (same rule as PLAN.md §11).

## M1 status (done, Sun 8 Nov ~00:30)

```bash
python -P -m modal run infra/mission.py::main --seed 3 [--lanes 2] [--auto "uncover_bomb@40,murk_on@95"]   # -> runs/mission_<scenario>_s<seed>/mission.mp4
python -P -m modal run infra/mission.py::status                          # while it runs
python -P -m modal run infra/mission.py::trigger --event uncover_bomb    # or drop_object --cls rock, murk_on, murk_off
```

Tested on seeds 3, 5 and 7 (2–3 lanes, bomb uncovered plus murk, both triggered automatically; the live trigger from the laptop was also tested). In every run the uncovered bomb was detected by sonar and the robot held, re-planned and inspected. **No keep-out violations** (closest approach to a confirmed mine-like contact was 4.0 m or more).

**How it works / gotchas (P2):**
- `SimSession(world=..., reserve=...)`: the world and the hidden objects for events must be created **before** the sensors start. Objects added later sometimes never get sonar labels: they cast a shadow but give no echo.
- Events **reveal** a pre-created object hidden under the seabed. They're placed 11–13 m ahead on the route, 1–1.6 m off the centreline and angled across it. End-on bombs barely echo, and dead-ahead objects sit in OceanSim's bright centre streak. An event is skipped if less than about 13 m of route is left.
- Path planning is grid A* around the union of keep-out circles (`mission_logic.route` / `detour`). New contacts are acted on in any moving state. If the robot confirms a contact when it's already inside a keep-out zone, it backs off.
- **Stand-in detector limitation:** it masks the centre streak, so it can't see within about ±0.6 m of dead ahead. **P1: please check `detect_frame` for the same blind spot.** The stand-in also can't tell classes apart: everything is `mine_like`, inspected and given an `uncertain` verdict.
- Murk only degrades the camera. Nothing reacts to it yet; P4's "camera unusable, so mark contacts `uncertain`" logic should do that.

**Plug-in points in `sim/mission.py`:** `perception.detect.detect_frame` (used when `ultralytics` imports inside Isaac), `agent.planner.classify_contact(contact, inspection_png_path, view)` (wrapped in try/except, with the stand-in as fallback), and `ml.choose_view_standin`, which P5's `imagine_contact` replaces. Any inspection pose it picks must pass `ml.pose_clear(pose, circles)`.

## Measured timing (for the technical summary and the 15% edge-budget criterion)

Every mission writes `runs/<id>/timing.json`, and `mission_log.jsonl` has a `timing` field per step. Numbers below are from run `timing_clear_sand_s3` (seed 3, 2 lanes, bomb uncovered, real YOLO), 165 steps:

| Stage | Runs on | Median | Worst |
|---|---|---|---|
| Sonar + camera render | Isaac Sim / OceanSim on a Modal L40S (stands in for the real sensors) | 0.16 s | about 99 s on the first frame only (GPU kernel compile) |
| Detection (YOLOv8n) | `uxo-detector` Modal service, GPU (P1) | 0.09 s | 0.29 s |
| Imagine + A* re-plan | On the robot: NumPy, CPU (P5 + P2) | 1.0 s | 1.0 s |
| Classify | Gemini via REST, 8 s timeout, template fallback (P1 + P4). Run `postmerge_clear_sand_s1` | about 4.1 s | 4.1 s |

The robot holds position while it imagines and classifies, so none of this is on the critical path for safety. **Autonomy note for the summary:** the classify step calls a remote model (Gemini). If it times out or fails, the on-robot template decides, so the robot never depends on the link to stay safe. Holding, keep-out and re-planning are always local. Detection is: about 0.1 s per frame against a 0.75 m step.
