# Live mission: the robot reacts while it runs

## Status / handoff (P2, Sun 8 Nov 02:10, checked 03:10 and morning)

**Working on `main` (d102e5a):** the full live loop with real YOLO (P1), imagination (P5) and Gemini classify (P4/P1), on P3's merged scene. Mission events are human-triggered and safe to press at any moment.

**Full-length test missions, one per scenario** (bomb triggered mid-inspection; it waited, then fired ahead of the robot). All are on the `uxo-runs` Volume, ready for the video:

| Run (Volume `uxo-runs`) | Steps | Objects found | False contacts | Closest to uncovered bomb | Keep-out breaches |
|---|---|---|---|---|---|
| `full_clear_sand_s1` | 628 | 5/5 | 0 | 4.4 m | none |
| `full_half_buried_s2` | 295 | 5/5 | 0 | 4.4 m | none |
| `full_murky_s4` | 448 | 5/5 | 0 | 4.5 m | 2 steps at 3.0 m from a tyre (see P1 below), then backed off |

Fetch with `python -P -m modal volume get uxo-runs full_clear_sand_s1 runs/`. Each folder has `mission_log.jsonl`, `timing.json`, `sonar/`, `cam/`, `chase/`, `imagined/`, `inspections/` (including `*_cam.png` with the camera aimed at the contact) and the PLAN §11 JSON files.

**Recording a final take (whoever records):**
```bash
python -P -m modal run infra/mission.py::main --seed 1 --scenario clear_sand --tag final   # full survey, ~8-10 min
python -P -m modal run infra/mission.py::trigger --event uncover_bomb                      # a person presses this, any time
python -P -m modal run infra/mission.py::trigger --event murk_on                           # optional second change
```
If the robot is holding, inspecting or re-routing when you press it, the bomb waits and fires when it resumes surveying. It's placed 11–13 m ahead on the route, at least 9 m from the robot and clear of existing keep-out zones. Use `viz/mission_compose.py runs/final_clear_sand_s1` for the video.

**Open items for teammates (from the test runs):**
- **P1 (detector):** (1) Low objects are seen late: in `full_murky_s4` a tyre wasn't detected until 3.8 m, so the robot had to back off. (2) Aerial bombs are still sometimes classed `other`. The mission holds for any contact near the route, so this is safe, but it affects verdicts. (3) YOLO sometimes draws two overlapping boxes on one object. (4) Check for a blind spot in OceanSim's centre streak (dead ahead).
- **P4 (classify):** verdicts across the 3 runs were mixed. A buried bomb was called non-UXO, two rocks were called UXO, and an uncovered bomb was called uncertain twice. The "murk makes the camera unusable, so say `uncertain`" logic isn't implemented yet: murk only degrades the camera.
- **P5 (video):** the runs above give footage for all 3 scenarios. The camera panel now shows the contact during inspections (`inspections/*_cam.png`, and `cam/` frames at the inspection pose).
- **P1 (beauty render, 03:10 check by P2):** `render/timeline.py` classifies events by keywords in the log's `event` caption. The `drop_object` caption ("a rock appears") and `murk_off` caption ("water clears") didn't match, so dropped objects would never be revealed and the water would never clear in the RTX replay. Fixed on the sim side: the captions are now "an object (<cls>) appears" and "visibility returns". Runs recorded before this fix with `drop_object`/`murk_off` keep the old wording. The `full_*` runs only used `uncover_bomb` and `murk_on`, so they're unaffected. Mission captions are now kept matching your parser; please tell P2 if you change the keywords.
- **NEW (P2, Sun morning): the robot revisits uncertain contacts.** After the survey it goes back to every contact classed `uncertain` and inspects it again from a different side (P5's next-best clear side), then calls `classify_contact` again. This is on by default; `--no-revisit` turns it off. Tested in `revisit2_clear_sand_s3` (first looks forced uncertain with the test-only `--force-uncertain` flag): c1 and c2 went uncertain → UXO, the tyre c3 went uncertain → non-UXO, and the robot stayed at least 4.3 m from every keep-out centre. If a contact is still uncertain after two views, or no other side is reachable, it's flagged for diver/ROV confirmation.
  - **Log/data format (additive):** the second verdict is logged as `revisit verdict cX: <class> (was uncertain from the <side>; …)`. It's deliberately not `verdict cX:`, so `render/timeline.py` and `viz/mission_compose.py` still key on the first look, and both parse it fine. Before a revisit, the log line starts `REVISIT: cX was uncertain from the …`. `targets.json` holds the **final** verdict, plus `revisited: true` and `first_look: {class, confidence, reason, side}`. `viewpoints.json` keeps one entry per contact, plus `revisit: {chosen, inspection, …}`.
  - **P5 (and P1 for the HUD):** the verdict card at the first-look step reads `targets.json`, so a revisited contact would already show the final class there. Show `first_look` at the `verdict cX` step, and the final class plus `viewpoints[cX].revisit.inspection` at the `revisit verdict cX` step.
- **Hero runs (P2 check, Sun morning):** `hero_clear_sand_s5` and `hero_clear_sand_s7` were recorded at 02:10 and 02:16 UTC on code from before the event fix (`d102e5a`, 02:06 UTC): the bomb fired at once while the robot was inspecting/holding. **Don't use s5**: the bomb was placed 1 m ahead of the robot during an inspection and it passed 0.2 m from it. s7 is OK (closest approach 4.1 m), and P1's beauty render of it is fine to use. Any new take on current `main` defers the event until the robot is surveying again.
- **P3:** pick the hero seed(s) for the final take. Seeds 1 (clear_sand), 2 (half_buried) and 4 (murky) are known to work end to end.

---

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
- ✅ Revisit: contacts classed `uncertain` get a second look from another side at the end of the survey (Sun morning).
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
