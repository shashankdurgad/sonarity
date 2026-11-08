# Sonarity

**An autonomous underwater survey robot that finds unexploded ordnance (UXO), keeps its distance, and *looks before it dives*: it imagines what it might be seeing before it moves closer.**

**Team:** Aditya Menon, Magizh Shankar, Shashank Durgad, Ibaad Muhammad, Samaira Khan.

Agentverse 2026, Robotics track. Everything here is **simulated**: NVIDIA Isaac Sim 6.0.1 + [OceanSim](https://github.com/umfieldrobotics/OceanSim) (forward-looking sonar and camera), run on Modal GPUs.

## The problem

Wartime munitions still lie on the seabed where offshore wind farms, cables and pipelines are being built. Every site has to be surveyed before anyone digs. The critical failure mode is a survey vehicle **passing close to a mine**: influence mines can be triggered by a vessel's magnetic, acoustic or pressure signature, so the robot must notice a threat with its own sensors, stop, and route around it, while the survey carries on.

## What the robot does (one continuous live run)

1. **Surveys** the site in lawnmower lanes with a forward-looking sonar and a camera.
2. **A person changes the scene mid-run** (e.g. seabed scour uncovers a bomb ahead). The robot is not told.
3. **Detects** the new contact with its sonar (YOLOv8 detector), **holds** before reaching it, draws a **keep-out circle** and **re-plans** its remaining route around it (grid A*).
4. **Imagines before it moves:** it pictures the contact as a *bomb* and as a *rock* from north, east, south and west with an on-board sonar model, scores where the two look most different, and picks the **best side outside the keep-out zone**.
5. **Inspects** from that side (real sonar + camera), **classifies** it (UXO / non-UXO / uncertain, with a reason), and **resumes** the survey.
6. If a verdict is **uncertain**, or was made while the camera was degraded by **murky water**, it **revisits** the contact later from the next-best side.
7. Outputs a prioritised **target list** for a human disposal team. *Detection and prioritisation only: humans dispose.*

**Final take** (`final_clear_sand_s1`, 699 steps): the bomb was triggered live, detected by sonar and held for; all 5 objects found; closest approach to any object 4.2–4.4 m with a 4 m keep-out (no breaches); 2 aerial bombs + 1 sphere mine = UXO, tyre = non-UXO; the rock stayed `uncertain`, was revisited and flagged for diver/ROV confirmation. Distances are scaled to a 32 m test site; real standoffs are far larger.

## How it works

```
OceanSim (Isaac Sim) ── sonar + camera ──► detector (YOLOv8) ──► contact tracker
                                                                    │
         ┌──────────────────────── mission loop (sim/mission.py) ───┘
         │  SURVEY → HOLD → REPLAN (keep-out + A*) → IMAGINE → INSPECT → CLASSIFY → RESUME (→ REVISIT)
         ▼
  imagine_contact (agent/imagine.py + sim/visit.py)   classify_contact (agent/planner.py)
  bomb vs rock from 4 sides, < 1 s, on board          Gemma (multimodal) on camera + sonar; local rule
                                                       fallback; mine-like is never cleared as safe
```

**Autonomy:** holding, keep-out, re-planning and the choice of inspection angle are always computed on the robot from its own sensor data, never from the simulator's ground truth. Classification uses **Gemma 4** (`gemma-4-26b-a4b-it`), an open multimodal model: it gets the camera and sonar images from the chosen side plus the first sonar detection, and returns a verdict with its reasoning in plain language (8 s timeout). Today it is called through the Gemini API; because Gemma is open-weight it can run on the robot's own GPU. If the call fails, the on-board rule decides, so the robot never depends on the link to stay safe.

### The imagination step

A bomb is long and thin and casts a long straight sonar shadow; a rock is round. They only look clearly different from some sides. `agent/imagine.py` renders both hypotheses with a lightweight NumPy sonar model (`sim/visit.py`: heightfield, shadow casting, beam pattern) at the detected position, size and heading, scores each side by how different the two imagined images are, skips sides inside any keep-out zone, and returns the pose to drive to. 16 imagined views in about 0.4 s on a CPU.

**Side experiment:** we also tried zero-shot imagination with NVIDIA's Cosmos 3 Edge world model (`experiments/cosmos/`). It could imagine our seabed with a bomb or a rock, but couldn't reliably show a given object from a chosen side (the object was missing in 8 of 9 clips in the controlled run), so the robot keeps its own model. Fine-tuning a world model on our sim data is the natural next step.

## Repository layout

| Path | What it is | Owner |
|---|---|---|
| `sim/` | OceanSim session (`session.py`), live mission loop (`mission.py`, `mission_logic.py`), scene builder (`scene.py`), sonar geometry, object specs, lawnmower survey, NumPy sonar model for imagination (`visit.py`) | P2, P3, P5 |
| `perception/` | YOLOv8 sonar detector, training, contact tracker, detector service | P1 |
| `agent/` | `imagine.py` (angle choice), `planner.py` (classification + target list) | P5, P4 |
| `viz/` | `mission_compose.py` (mission video with the robot's reasoning), `compose.py` (batch demo video) | P5 |
| `render/` | Cinematic RTX replay of a logged mission with a HUD (`beauty.py`, `hud.py`, `timeline.py`) | P1 |
| `infra/` | Modal jobs: Isaac image, survey, live mission + live event trigger, beauty render | P2 |
| `eval/`, `tools/`, `tests/` | Stress-test scoring, run/scene validation, smoke tests | all |
| `experiments/cosmos/` | Zero-shot imagination test with Cosmos 3 Edge | P5 |
| `runs/sample/` | Small sample run in the data contract format | |
| `PLAN.md`, `LIVE_MISSION.md` | Original plan, data contract (§11), live-mission design and status | |

## Running it

Requirements: Python 3.10+ and `pip install -r requirements.txt` for the local parts; a Modal account with an NGC secret for the simulator (see `infra/image.py`); `GEMINI_API_KEY` (Google AI key) for Gemma verdicts (optional).

```bash
pip install -r requirements.txt

# Smoke tests + batch pipeline on the sample run (no GPU)
python tests/test_smoke.py
python run_demo.py runs/sample --no-gemma

# Live mission on Modal (Isaac Sim + OceanSim on an L40S)
python -P -m modal run infra/mission.py::main --seed 1 --scenario clear_sand --tag final
python -P -m modal run infra/mission.py::trigger --event uncover_bomb     # a person presses this, any time
python -P -m modal run infra/mission.py::trigger --event murk_on          # optional second change

# Videos from a finished run
python viz/mission_compose.py runs/<run_id>                                 # mission view with the robot's reasoning
python -P -m modal run infra/beauty.py --run-id <run_id> --shards 8 --mode RaytracedLighting   # cinematic replay
```

Run folders follow the contract in `PLAN.md` §11: `mission_log.jsonl`, `sonar/`, `cam/`, `chase/`, `contacts.json`, `viewpoints.json`, `imagined/`, `inspections/`, `targets.json`, `world.json`.

## Limitations

- Simulation only; distances and keep-out radii are scaled to a 32 m test site.
- The detector sometimes sees low objects late and sometimes labels bombs `other`; the mission holds for any contact near the route, so this is safe but affects verdicts.
- Gemma is currently called remotely, with an on-board fallback; running it on the robot is the next step.
- The imagination model is a simplified sonar (no multipath or water-column effects).

## Credits

Built by **Team Sonarity** (Aditya Menon, Magizh Shankar, Shashank Durgad, Ibaad Muhammad, Samaira Khan) at Agentverse 2026. OceanSim by the University of Michigan Field Robotics Group; NVIDIA Isaac Sim; Ultralytics YOLOv8; Google Gemma; NVIDIA Cosmos 3 Edge (experiment). Marine-life models in the beauty render are CC BY (see the credits file in the render assets).
