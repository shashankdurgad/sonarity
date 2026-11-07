# "Look Before You Dive": proof-of-concept plan
**A simulated underwater robot surveys a North Sea wind-farm site in OceanSim, finds suspicious objects, works out the best angle to check each one, and hands humans a prioritised target list. It never touches anything.**

**Deadline: submit by 01:45, hard stop 02:00, Sunday 8 Nov (GMT).** Goal: a believable demo that shows the direction, not a robust system.

The original 24-hour plan is in git history (`git show be98c51:PLAN.md`).

---

## 1. What changed

| Original | Now | Why |
|---|---|---|
| Full OceanSim environment with 5 scenarios | **Minimal OceanSim scene: generated seabed, simple objects, sonar + camera** | Keeps the real sensor simulation, drops most of the setup |
| AUV with hydrodynamics and PID control | **Sensors follow a scripted path** | Nothing to tune, and it looks the same on video |
| Closed loop in Isaac | **Separate Modal jobs that pass files between steps** | Each person can work and test alone |
| Live Cosmos server, LoRA on thousands of episodes | **Imagination rendered in OceanSim**; Cosmos only as a stretch | Cuts the riskiest training and a live service |
| YOLOv8s on real + large synthetic set | **YOLOv8n, two short runs** | Training fits in about an hour |

**Removed:** LeRobot, EnvHub, ACT, LoRA, MarineGym, ROS2, TensorRT, swarm, silt plume, magnetometer.

**Python sonar sim:** fallback only. Nobody builds it unless the 20:00 gate in §3 fails.

---

## 2. Architecture

```
 modal: sim/survey.py     scripted lawnmower through the OceanSim scene
        │                 → sonar/*.png, cam/*.png, poses.csv, world.json, labels/
        ▼
 perception/detect.py     YOLOv8n on the sonar frames → contacts.json
        ▼
 modal: sim/visit.py      for each contact:
        │                 1. imagine: in a clean scene, render "bomb" vs "rock"
        │                    from 3–4 angles and pick the angle where they differ most
        │                 2. inspect: render the real scene from that angle
        │                 → viewpoints.json, imagined/*.png, inspections/*.png
        ▼
 agent/planner.py         Gemini reads the crops, writes verdicts → targets.json
        ▼
 viz/compose.py           sonar | camera | map | imagined views | verdict → demo.mp4
```

Each step reads and writes files in `runs/<id>/`. Only the two sim jobs and training need a GPU, so everything else runs on a laptop against `runs/sample/`.

**Why this counts as imagination:** before moving, the robot predicts what each hypothesis would look like from each angle, then goes where the answer would be clearest. The imagine step must use only the detected position and size, never `world.json`. Ground truth is for labels and scoring.

---

## 3. Proving OceanSim on Modal (top priority)

**Known facts:** the `nvcr.io/nvidia/isaac-sim:5.0.0` image exists, and OceanSim targets Isaac Sim 5.0. OceanSim has headless camera scripts, but no headless sonar script, so we write that.

**Main unknown: rendering on Modal.** The sonar is built on an RTX camera, so it needs Vulkan. Modal runs containers on gVisor, which supports Vulkan only if the graphics capability is exposed, and we found no confirmation that Modal does this. Also, a forum report shows camera scripts crashing on a headless L40S with a too-new driver. Test this first.

| By | Step | Pass when |
|---|---|---|
| **17:30** | **1. GPU + Vulkan:** L40S container from the 5.0.0 image | `nvidia-smi` and `vulkaninfo --summary` both see the L40S. Record the driver version |
| **18:15** | **2. Isaac renders:** headless script with a cube and camera | A PNG with the cube visible |
| **19:15** | **3. OceanSim sonar:** extension in `/isaac-sim/extsUser`, `ImagingSonarSensor` aimed at a sphere | A sonar PNG with the sphere visible, plus an underwater camera PNG |
| **20:00** | **4. Survey pass (GATE A):** path over seabed with 4 objects | MP4 of sonar + camera with objects visible and labels saved |

**Try these before calling it failed:**
- Add `/etc/vulkan/icd.d/nvidia_icd.json` to the image, and unset `CUDA_VISIBLE_DEVICES`
- Try the `5.1.0` image, then another RTX-capable Modal GPU type
- Ask Modal support or their community about Vulkan, starting at step 1 if it fails

**Decision at 20:00:** if step 4 hasn't passed, write down what failed and what was tried. Then move to the Python sonar sim (§9), which takes about 1.5 hours and keeps everything downstream the same. If you have a decisive lead at 20:00, you can extend once to 20:30, which means dropping the hero-shot polish.

**Setup notes:**
- Run the image with `modal.Image.from_registry(..., add_python=...)` and start Isaac as a subprocess through `/isaac-sim/python.sh`. Set `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y`
- Keep the shader cache on a Volume so only the first start is slow
- Use `modal shell` for debugging, but stop it when done, because it holds a GPU
- **Don't use OceanSim's Google Drive assets:** use generated terrain and simple primitives with semantic labels. Start with the sonar at `max_range` about 10 m, `hori_res` about 1000, and altitude 2–3 m, then tune

---

## 4. Components

| Part | Choice | Notes |
|---|---|---|
| Sim | OceanSim on Isaac Sim 5.0, Modal L40S | Forward-looking imaging sonar + underwater camera |
| Scene | Generated seabed; bomb (cylinder + fins), sphere mine, rock, tyre; some part-buried | Primitives keep it simple |
| Scenarios | `clear_sand`, `half_buried`, `murky` | `murky` changes the camera only |
| Detector | YOLOv8n, 2 classes: `mine_like`, `other` | Trained on Modal L40S |
| Real sonar data | `rehan9599/drishti-sss`: 5,205 YOLO images, about 2 GB, CC-BY-SA-4.0 | Map `mine_cylinder` → `mine_like`; `shipwreck`, `submarine_pipeline`, `ghost_net` → `other` |
| Synthetic data | About 500–1,000 OceanSim frames, labelled from sim semantics | From the survey job |
| Planner | Gemini, structured JSON output | Confirm the model name and key in the first hour. Fall back to a template if the API fails |
| Chase shots | Extra Isaac camera in the survey job | Simple AUV prim. Optional |
| World model (stretch) | Cosmos3-Edge (4B), zero-shot, pre-rendered for the hero contact | Only if Gate C is early. Labelled as a concept |

**Training:**
1. **Run 1:** `drishti-sss`, about 30 epochs. Record mAP on its test split. This doesn't depend on the sim, so it starts right away.
2. **Run 2:** continue on OceanSim frames + real data, about 20 epochs. Record mAP on held-out seeds.

`drishti-sss` is side-scan sonar and OceanSim is forward-looking, so run 1 won't transfer well. **Run 2 is the one the demo needs.** Report the two scores separately. Expect 1–2 GPU-hours total.

**Budget:** $500 Modal credit, with a $450 limit. Expect to spend **under $60**.

---

## 5. Team

| Person | Owns | First deliverable |
|---|---|---|
| **Shashank:** training + Modal | `perception/`, Modal account and spend, Cosmos stretch | YOLO run 1 by 18:30 |
| **P2:** OceanSim on Modal | `modal/`, steps 1–3 in §3, `sim/survey.py` job | Step 1 by 17:30 |
| **P3:** scene + data | `sim/scene.py`, scenarios, labelled frames | Step 4 with P2 by 20:00 |
| **P4:** planner + integration | `agent/planner.py`, `run_demo.py`, `viz/metrics.py`, smoke test, merges, calling cuts | Pipeline on stubs by 18:00 |
| **P5:** imagination + demo output | `agent/imagine.py`, `sim/visit.py`, `viz/compose.py` | Angle scoring and `compose.py` on sample files by 19:00 |

The video is not anyone's job until the end: the team records it together from the pipeline output after feature freeze.

**Handoffs (estimates):**
- **P2 → P3, P5:** working Isaac + OceanSim image by about 19:15. P3 and P5 write code against the planned interface until then.
- **P3 → Shashank:** labelled frames by about 21:00, so run 2 finishes before Gate B.
- **P3 → P5:** `sim/scene.py` usable for clean and real scenes by Gate A.
- **Shashank → P4:** run 1 weights by 18:30, run 2 weights by about 22:00.
- **P4 → P5:** a real `runs/<id>/` folder by Gate B. Until then, use `runs/sample/`.

**Between training runs,** Shashank helps P2 debug Modal. If OceanSim fails at 20:00, P2, P3 and P5 build the Python sim together.

**Modal seats:** Shashank, P2, P3 and P5 all run GPU jobs. If Starter limits seats to 3, upgrade, or have P5 run `sim/visit.py` through P3.

---

## 6. Schedule (Saturday 7 Nov → Sunday 8 Nov, GMT)

| Time | Goal |
|---|---|
| **16:45–17:15** | Kickoff: repo skeleton, `CLAUDE.md`, file formats (§7), stubs and `runs/sample/`. Accounts and keys |
| **17:15–20:00** | P2 + P3 prove OceanSim (§3). Shashank trains run 1. P4 connects stubs. P5 builds angle scoring and compose on samples |
| **GATE A, 20:00** | **OceanSim survey video, or the documented decision to fall back** |
| **20:00–22:30** | P3 generates frames → Shashank trains run 2 → P4 connects real pieces and Gemini. P5 gets `sim/visit.py` running in OceanSim |
| **GATE B, 22:30** | **Baseline: target list and `demo.mp4`, inspecting from a fixed angle. Save it. This is submittable** |
| **22:30–00:00** | P5 adds imagination to the loop, with P3's help. P2 adds chase shots. P4 runs metrics |
| **GATE C, 00:00** | Hero contact shows imagined views → chosen angle → verdict |
| **00:00–00:45** | Record 3 scenarios × 3–5 seeds and choose the best footage |
| **00:45** | **Feature freeze** |
| **00:45–01:45** | Team records the video from pipeline output (simple edit only), P4 finishes the README, final checks. **Submit by 01:45** |

**Cut in this order when behind:** Cosmos → chase shots → scenarios beyond `clear_sand` → imagination on all contacts (keep the hero contact) → Gemini (use a template) → YOLO run 2 quality. Never cut Gate B.

---

## 7. File formats (agree at kickoff)

```jsonc
// world.json: ground truth, used only by labels, metrics and render
{"seed": 3, "scenario": "half_buried",
 "objects": [{"id": "o1", "cls": "aerial_bomb", "xyz": [42.0, 118.5, -0.2], "yaw": 0.7, "burial": 0.3}]}

// contacts.json
[{"id": "c1", "xy": [41.8, 118.9], "size_m": [1.8, 0.5], "cls": "mine_like", "conf": 0.81,
  "frame": "sonar/000412.png", "crop": "crops/c1.png"}]

// viewpoints.json
[{"contact_id": "c1", "chosen": "north", "scores": {"north": 0.72, "east": 0.31, "south": 0.68, "west": 0.29},
  "imagined": {"north": ["imagined/c1_north_bomb.png", "imagined/c1_north_rock.png"]},
  "inspection": "inspections/c1_north.png"}]

// targets.json
[{"contact_id": "c1", "class": "UXO", "confidence": 0.86, "priority": 1, "standoff_m": 50,
  "reason": "Long, narrow return with a fin-like tail from the north view"}]
```
`poses.csv` columns: `t,x,y,depth,yaw,frame`. `class` is `UXO`, `non-UXO` or `uncertain`. `standoff_m` is illustrative.

---

## 8. Repo and the video

```
├── CLAUDE.md          # rules for every Claude session
├── run_demo.py        # runs every step, Modal jobs included
├── modal/{image.py, sim_jobs.py, train_jobs.py}
├── sim/{scene.py, survey.py, visit.py}
├── perception/{train.py, detect.py}
├── agent/{imagine.py, planner.py}
├── viz/{compose.py, metrics.py}   # compose = demo output view
├── tests/test_smoke.py  # runs run_demo.py on stubs
├── runs/sample/         # example of every file, committed
└── video/{script.md, shots.md}
```

**Working rules:** one owner per folder; small commits; pull before push; P4 runs the smoke test after every merge. Check outputs by eye at every gate.

| Time | Shot |
|---|---|
| 0–7 s | The problem: wartime munitions under the North Sea, wind farms being built on top |
| 7–17 s | Survey begins, sonar and murky camera side by side, contacts appear |
| 17–37 s | **Hero:** a contact appears → imagined "bomb vs rock" from 4 angles → picks the clearest → inspects → verdict with reason |
| 37–47 s | Fast cuts: tyre and rock cleared, part-buried bomb flagged |
| 47–54 s | Numbers: detector mAP, contacts found, classification accuracy, all from real runs |
| 54–60 s | Target list on the site map, handed to the disposal team. "Look before you dive." |

**Keep the claims honest:** say "simulated" on screen and report only measured numbers. Confirm the source of the 1.3 million tonnes figure before recording.

---

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| **OceanSim can't render on Modal by 20:00** | **Python sonar sim:** NumPy heightfield + shadow casting + speckle, writing the same files. P2, P3 and P5 build it in about 1.5 h; `sim/visit.py` uses it too. Gate B holds at 22:30 and the top-down map replaces camera shots |
| Sonar import fails headless (`omni.ui` import, viewport code) | Avoid `make_sonar_viewport`; patch or stub only the UI parts |
| Slow cold starts | Shader cache on a Volume; batch work into one job per scenario |
| Sonar is too slow at default settings | Lower `hori_res`, use fewer frames, and hold still at each path point |
| Detector weak on OceanSim frames | More OceanSim frames in run 2; for the demo, train only on them and report real-data scores separately |
| Imagined views don't differ by angle | Make shapes less symmetric; otherwise use fixed-angle footage and show imagination for the hero contact only |
| Gemini slow or failing | Cache by contact; use template verdicts |
| Cosmos gated, won't fit or looks bad | Drop it. It isn't part of any gate |
| Broken main | One owner per folder; P4 restores the last passing commit |

---

## 10. First-hour checklist

- [ ] Names assigned to P2–P5, everyone can push
- [ ] Modal installed, workspace shared, $450 limit set
- [ ] L40S container from `isaac-sim:5.0.0` running; `vulkaninfo` result posted to the team
- [ ] Hugging Face token; `drishti-sss` downloading on Modal
- [ ] Gemini key tested with one JSON call
- [ ] `CLAUDE.md`, stubs, `runs/sample/` and smoke test committed
- [ ] Everyone runs `python run_demo.py` on stubs
