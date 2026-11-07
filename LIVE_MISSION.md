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
| M1 | Loop runs with stand-ins | P2 | A mission run with a live-triggered `uncover_bomb`: the robot holds, detours and resumes, with `mission_log.jsonl` written. Stand-in detector and planner |
| M2 | Real detector in the loop | P1 + P2 | `detect_frame` + `ContactTracker` replace the stand-ins; the bomb is detected live |
| M3 | Real imagine + classify in the loop | P5 + P4 + P2 | Inspection from the chosen angle outside the circle; verdict from `classify_contact` |
| M4 | Mission video | P5 + P3 | The composed mission view from one continuous run, with a human-triggered change |
| M5 | Record final takes | all | 2–3 takes; pick the best |

M1 doesn't wait on anyone, and P1, P4 and P5 can write their functions against the formats above in parallel. Post in the group chat if you need to change any format here (same rule as PLAN.md §11).
