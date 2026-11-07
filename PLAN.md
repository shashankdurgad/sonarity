# "Look Before You Dive" — 24h Hackathon Plan
### An autonomous underwater robot that surveys North Sea wind-farm sites for unexploded WWII ordnance

---

## In plain English: what this project is

**The problem.** Up to about 1.3 million tonnes of wartime munitions are thought to lie on or just under the North Sea seabed, exactly where the UK and its neighbours are building offshore wind farms. Before anyone can lay a cable or sink a foundation, the seabed has to be surveyed and every suspicious object checked. Today that means a survey ship towing sonar for weeks, specialists manually reviewing the data, and divers or remote vehicles sent back to re-inspect each suspicious contact. It is slow, expensive, and the hardest part is telling a bomb from a rock, a tyre or a lump of pipe.

**What we build.** A simulated autonomous underwater vehicle (AUV) that surveys a patch of seabed on its own, finds suspicious objects, and *imagines what it would see from different angles before moving* so it can decide the fastest way to confirm what each object is. It never touches anything. It hands a prioritised **"master target list"** to human disposal teams. Its brain has three layers:

1. **Mission planner (Gemini 4 Argon)** — plans the survey, decides which contacts need a closer look, and writes the final target list with reasons.
2. **Imagination (NVIDIA Cosmos world model)** — predicts what the sonar and camera *would* show from each candidate viewpoint, so the planner can pick the most informative one instead of circling blindly. In Tier 2 we fine-tune it on our own synthetic sonar so it genuinely understands sonar shadows.
3. **Perception + control (small NVIDIA-stack models)** — a sonar object detector trained on real Hugging Face sonar data plus synthetic sonar, and a depth/heading controller that flies the vehicle.

All of it runs in **NVIDIA Isaac Sim** via **OceanSim**, an open-source underwater extension with physics-based camera and imaging-sonar rendering.

**The steps.**
1. Get an AUV flying a survey pattern over a procedurally generated North Sea seabed, with sonar and camera output, on Modal L40S GPUs.
2. Scatter objects: mines, aerial bombs, rocks, tyres, pipe sections, debris. Some half-buried.
3. Train a sonar detector on real + synthetic sonar.
4. Wire up the loop: survey → detect contact → imagine viewpoints → pick → inspect → classify → log.
5. (Tier 2) Generate thousands of synthetic survey passes and fine-tune the world model on them. Show before/after.
6. (Tier 3) Multiple AUVs, murky-water and silt-plume handling, buried-object magnetometer cue.
7. Cut a 60-second video.

**The outcome.** A 60-second video of an AUV autonomously surveying a wind-farm site in murky water, reasoning out loud about each contact, choosing inspection angles by imagining them first, and producing a target list for the humans. Plus an open synthetic sonar dataset and sim environment on Hugging Face.

**One line:** *"Survey the seabed before anyone has to dive."*

---

## Assumptions and open questions

Answer these before kickoff if you can; the plan picks defaults otherwise.

| # | Question | Default assumed |
|---|---|---|
| 1 | Is the $250 all Modal GPU credit, or does it also need to cover storage/CPU? | All GPU; CPU + storage ≈ $5–10 extra |
| 2 | Max concurrent L40S on the workspace? | Assume **4** concurrent; plan scales to 8–10 if allowed |
| 3 | Does anyone have an Isaac Sim / Omniverse account already? (needed for some asset downloads) | One person creates it at T+0 |
| 4 | Gemini 4 Argon API key confirmed working? | Yes; `gemini-3.8-flash` fallback |
| 5 | Is Cosmos 3 gated access already granted on someone's HF account? | Request at T+0; fallback Cosmos Predict2-2B (ungated) |
| 6 | Do we want one AUV or a swarm in the video? | One AUV baseline; swarm is Tier 3 |

---

## 0. The pitch (narration skeleton)

> The seabed around Britain hides hundreds of thousands of tonnes of unexploded wartime bombs, right where we're building the wind farms that will power the country. Every site must be surveyed first, and the hard part isn't finding objects, it's telling a bomb from a rock. Our robot surveys on its own, and before it moves it imagines what each viewpoint would reveal, so it inspects smarter, not longer. Look before you dive.

---

## 1. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ MISSION PLANNER — Gemini 4 Argon                                   │
│ in: survey map, detector contacts, imagined views                  │
│ out: survey pattern, revisit queue, viewpoint choice, target list  │
└───────────────┬────────────────────────────────────────────────────┘
                │ candidate viewpoints for a contact
┌───────────────▼────────────────────────────────────────────────────┐
│ WORLD MODEL — Cosmos 3 Edge (4B)  [Tier 2: LoRA on synthetic sonar]│
│ in: current sonar/camera frame + candidate trajectory              │
│ out: imagined sonar strip / camera clip from that viewpoint        │
└───────────────┬────────────────────────────────────────────────────┘
                │ chosen viewpoint
┌───────────────▼────────────────────────────────────────────────────┐
│ PERCEPTION — YOLO sonar detector (real HF data + synthetic)        │
│ CONTROL — depth/heading/waypoint PID + station-keeping             │
└───────────────┬────────────────────────────────────────────────────┘
                │ thruster commands
┌───────────────▼────────────────────────────────────────────────────┐
│ SIMULATOR — OceanSim on Isaac Sim 5.0                              │
│ procedural seabed, scattered UXO + decoys, imaging sonar, camera,  │
│ turbidity/fog, (Tier 3) silt plume, fake magnetometer              │
└────────────────────────────────────────────────────────────────────┘
```

### Components and sources

| Component | What | Source / notes |
|---|---|---|
| Simulator | **OceanSim** — public repo, Isaac Sim 5.0 GA compatible; physics-based underwater camera + GPU imaging sonar; terrain-generation example; standalone synthetic-data-generation scripts; ROS2 example | github.com/umfieldrobotics/OceanSim |
| Vehicle dynamics fallback | **MarineGym** (Isaac Sim hydrodynamics plugin, 5 UUV models) if its code is available; otherwise a hand-written buoyancy + quadratic drag model (≈50 lines) | github / arXiv 2503.09203 |
| World model | **Cosmos 3 Edge** (4B, fits one L40S for inference and LoRA) | huggingface.co/collections/nvidia/cosmos3 |
| Planner | **Gemini 4 Argon** (`gemini-4-argon`), structured JSON output; fallback `gemini-3.8-flash` | Gemini API |
| Sonar detector | Ultralytics YOLOv8s/v11s trained on **`rehan9599/drishti-sss`** (HF, YOLO-format side-scan tiles incl. mine-like contacts, wrecks, hard negatives, CC-BY-SA-4.0) + our synthetic OceanSim sonar | Hugging Face |
| Extra real sonar | MILCO/NOMBO mine dataset (figshare, 1,170 images); AI4Shipwrecks (286 images); UATD forward-looking sonar (9k images) | figshare / GitHub |
| Data format / glue | **LeRobot** dataset format + EnvHub wrapper for the OceanSim env | github.com/huggingface/lerobot |
| Compute | Modal, **L40S only** (48 GB), ≈ $1.95/hr | modal.com |

---

## 2. Interface contracts (agree in the first 30 min)

```python
# Observation (sim → everyone)
obs = {
  "sonar": np.uint8[512, 512],          # imaging-sonar frame (polar→cartesian), 1 channel
  "sss_strip": np.uint8[256, W],        # side-scan waterfall accumulated over last N s (optional)
  "cam": np.uint8[360, 640, 3],         # forward camera, turbidity applied
  "pose": np.float32[7],                # x,y,z (depth), quat
  "vel": np.float32[6],
  "altitude": float,                    # height above seabed (DVL-style)
  "mag": float,                         # Tier 3: fake magnetometer anomaly (nT)
  "t": float,
}

# Action (controller → sim)
action = np.float32[4]                  # surge, sway, heave, yaw-rate (normalised -1..1)

# Contact (detector → planner)
contact = {"id": str, "xy": [x, y], "cls": str, "conf": float, "snippet": base64_png}

# Viewpoint candidate (planner → world model → planner)
viewpoint = {"id": str, "label": str, "waypoints": [[x,y,z], ...], "rationale": str}

# Imagination request (anyone → world-model server)
POST /imagine {frame: b64, modality: "sonar"|"cam", actions: [[s,sw,h,yr], ...] | None, prompt: str, n_frames: 16}
→ {clip: b64_mp4, frames: [b64_png, ...]}

# Planner verdict (planner → log/overlay)
verdict = {"contact_id": str, "class": "UXO"|"non-UXO"|"uncertain", "confidence": 0-1,
           "reason": str, "standoff_m": float, "priority": int}

# LeRobotDataset episode features
observation.images.sonar, observation.images.cam, observation.state, action, task, scenario_id, labels (per-object gt)
```

---

## 3. Budget model ($250, L40S only, ≈$1.95/hr → ~128 GPU-hours total)

Everything runs on L40S. No H100 means: keep the world model at **Edge 4B**, LoRA not full fine-tune, 8–16-frame windows, 256–360p. Parallelism comes from several L40S running *independent* jobs, never one job split across GPUs.

| Bucket | GPU-hours | Cost | Notes |
|---|---|---|---|
| Sim bring-up & scenario dev (1 GPU, intermittent) | 14 | $27 | Scale-to-zero between runs |
| World-model inference server (intermittent) | 10 | $20 | `container_idle_timeout=90s` |
| Sonar detector training (YOLO, 2 runs) | 2 | $4 | Tiny |
| **Tier 2** synthetic data generation (4 GPUs × 1.5h) | 6 | $12 | 3–6k episodes |
| **Tier 2** Cosmos LoRA fine-tune (2 configs in parallel × 5h) | 10 | $20 | |
| **Tier 2/3** LeRobot ACT controller training | 2 | $4 | Optional |
| Parallel evaluation/recording passes (4 GPUs × 2h × 2 passes) | 16 | $31 | Baseline pass + fine-tuned pass |
| **Tier 3** extras (swarm, plume, re-records) | 12 | $23 | Only if ahead of schedule |
| Shader-cache image build, mistakes, OOMs | — | $25 | Buffer |
| CPU, storage, egress | — | $8 | |
| **Total** | **~72 GPU-h** | **≈$174** | Leaves ~$75 headroom |

Rules:
- Bake Isaac Sim's shader cache into the Modal image (or a Volume) before fanning out, or every container burns 3–5 min of GPU at startup.
- Every Modal function has `container_idle_timeout ≤ 120` and `timeout` set. No GPU sits idle.
- One person owns the Modal dashboard and checks spend at every gate.

---

## 4. Tier 0 → Tier 3: what each adds

| | **Tier 0 — Baseline** | **Tier 1 — Imagination** | **Tier 2 — Taught world model** | **Tier 3 — Stretch** |
|---|---|---|---|---|
| Sim | OceanSim seabed, 1 AUV, sonar + cam, 6 object types, turbidity | + 5 scenarios (clear, murky, sand ripples, half-buried, debris field) | + mass synthetic data gen | + silt plume, multi-AUV, fake magnetometer |
| Perception | YOLO on real HF sonar + synthetic | same | retrained with 10× more synthetic | + buried-object cue from magnetometer |
| World model | none (planner picks viewpoints by rules) | Cosmos 3 Edge **zero-shot**, camera-view imagination only | **LoRA on our sonar+cam data**; action-conditioned sonar imagination | fine-tuned on plume data too |
| Planner | Gemini: lawnmower survey + revisit queue + target list | + scores imagined views, picks viewpoint | same, now with meaningful sonar imagination | + task allocation across AUVs |
| Control | PID depth/heading/waypoint | same | optional LeRobot ACT for close inspection | same |
| Hero shot | AUV surveys, finds and classifies contacts, outputs list | 2×2 imagined viewpoints → pick → inspect | **before/after**: zero-shot vs fine-tuned imagination vs truth | 3 AUVs splitting a site; plume avoided |
| Est. cost | ~$55 | ~$80 | ~$150 | ~$200+ |

**Build in order. Each tier is a complete, submittable demo.** Do not start a tier until the previous one has recorded footage in `/data/best/`.

---

## 5. Timeline (T = kickoff). Feature freeze T+20:00, edit freeze T+23:00.

### Phase 0 — T+0:00 → T+0:30 — Kickoff
- Confirm answers to the open questions. Check Modal L40S concurrency.
- Repo (layout §7), Modal workspace, Volume `/data`, HF org + dataset repo.
- Request Cosmos 3 gated access on HF (one account). Create Omniverse/NVIDIA developer account for Isaac Sim assets.
- Agree §2 interfaces. Video workstream starts script v1 from §6.

### Phase 1 — T+0:30 → T+4:00 — AUV in the water (CRITICAL PATH)
**Sim workstream (2 people on this):**
- Modal image: Isaac Sim 5.0 container + OceanSim installed per `docs/subsections/installation.md`. Accept EULAs via env vars. Run once during build to populate shader cache; persist cache dir to a Volume.
- Run OceanSim's terrain-generation example and SDG scripts headless (`--no-window` / `headless=True`). Confirm: (a) imaging-sonar frame as numpy, (b) camera frame with turbidity, (c) vehicle moves on thruster commands, (d) reset with seed → new seabed.
- Vehicle: use OceanSim's demo vehicle if present; otherwise a BlueROV2-like rigid body with buoyancy + drag. Write `sim/auv_env.py` (reset/step/render) and `sim/controller.py` (PID depth/heading/waypoint).
- `sim/objects.py`: spawn library — cylinder mine, aerial bomb (cylinder + fins), sphere mine, rock (random mesh), tyre (torus), pipe section, crate/debris. Random partial burial (sink z by 0–60% of height). Record ground-truth class + pose per object.

**World-model workstream:**
- `models/world_model_server.py` on L40S serving Cosmos 3 Edge; `/imagine` with text prompt + frame. Test on a real underwater photo and a real sonar tile from `drishti-sss`. Note how bad sonar imagination is zero-shot (this is the "before" footage, save it).

**Perception workstream:**
- `snapshot_download("rehan9599/drishti-sss")`; train YOLOv8s on L40S (~40 min). Save weights to Volume. Measure mAP on its test split.

**Planner workstream:**
- Gemini 4 Argon key; JSON-schema prompt that takes a survey map + contact list and returns a revisit queue and viewpoint candidates. Test on fake inputs. Measure latency.

**Data & visuals workstream:**
- `data/lerobot_writer.py` (dummy test), `viz/compose.py` (sonar waterfall | camera | map | imagined 2×2 layout), `viz/overlays.py` (planner text, standoff circles).

**GATE 1 (T+4:00):** AUV flies a lawnmower pattern headless on Modal; we have an MP4 with sonar + camera showing a scattered object. **Hard deadline T+5:30 → Fallback F1.**

### Phase 2 — T+4:00 → T+8:00 — Tier 0 complete
- Sim: 5 scenarios as config overrides — `clear_sand`, `murky` (high turbidity), `ripples` (rippled seabed texture confuses sonar), `half_buried`, `debris_field` (many decoys).
- Perception: fine-tune YOLO on real + ~500 synthetic OceanSim sonar tiles; export to TensorRT if easy (NVIDIA points).
- Planner + control: `agent/mission_loop.py` Tier 0 — survey → detector contacts → rule-based revisit (fly a circle at standoff) → classify from multiple views → Gemini writes target list with reasons.
- Data & visuals: recording hooks (sonar, cam, chase-cam, map, planner JSON) → `/data/runs/<id>/`; start `video/shots.md`.

**GATE 2 (T+8:00):** Full Tier 0 loop produces a target list on `clear_sand`. Record it. *You now have a submittable video.*

### Phase 3 — T+8:00 → T+12:00 — Tier 1 (imagination in the loop) + Tier 2 data gen starts
- Planner: for each contact, propose 3–4 viewpoints; call `/imagine` (camera modality, zero-shot) per viewpoint; score via Gemini ("which view would best reveal a cylinder with fins?"); go. Overlay the 2×2 imagined grid + reasoning text.
- Sim + Data: **fan out synthetic generation to 4 L40S** (~1.5h): scripted survey passes over randomised seabeds, all 5 scenarios, 20–40 s episodes, sonar + cam + actions + object labels → LeRobotDataset shards → merge → push to HF as `<org>/northsea-uxo-survey-v1`. Target 3–6k episodes. Include "bad approach" episodes (wrong angle, too high, so shadows are ambiguous) — the world model must learn why angle matters.
- World model: write `models/convert_to_cosmos.py` (LeRobot → Cosmos forward-dynamics format; sonar as single-channel repeated to RGB).

**GATE 3 (T+12:00):** Tier 1 footage recorded on 2+ scenarios. ≥2k episodes on the Volume.

### Phase 4 — T+12:00 → T+18:00 — Tier 2 (fine-tune) while polishing
- World model: launch **2 LoRA configs on 2 L40S** (Cosmos 3 Edge; rank 16; bf16; 8–16-frame windows; 256p sonar, 360p cam; lr 1e-4 vs 3e-5). Checkpoint every 30 min; eval on 20 held-out episodes: imagined vs real sonar strip, frame error. **First checkpoint clearly better than zero-shot is enough.** Swap the best LoRA into the server.
- Planner: switch to action-conditioned imagination (controller pre-computes the trajectory for each viewpoint → world model rolls it out). Keep text-prompt mode behind a flag.
- Perception: retrain YOLO with the full synthetic set; report mAP before/after on the real test split (a real number for the video).
- Optional: LeRobot ACT policy for the close-inspection orbit, trained on scripted passes (1 L40S, 2h). Use only if it beats PID visually.
- Data & visuals: before/after 3-panel renderer (zero-shot | fine-tuned | ground truth), same start frame, same actions.

**GATE 4 (T+18:00):** Fine-tuned LoRA in the loop; before/after clip recorded on `half_buried` and `ripples`.

### Phase 5 — T+18:00 → T+20:00 — Recording pass + Tier 3 if ahead
- **4 L40S in parallel**: 5 scenarios × 5 seeds, Tier 2 loop. Record everything. Compute: contacts found / total, classification accuracy UXO vs non-UXO, avg inspections per contact (Tier 0 rules vs Tier 2 imagination — fewer is the win), survey time.
- Tier 3 only if Gate 4 passed early:
  - **Silt plume**: particle effect triggered when altitude < 1 m and thrust high; world model (or a heuristic) predicts it; planner picks up-current approach. 10-second clip.
  - **Multi-AUV**: 3 vehicles, Gemini allocates survey lanes and revisits. 10-second overhead shot.
  - **Magnetometer**: dipole anomaly for ferrous objects; lets the planner flag fully buried items the sonar can't see. One clip.

**Feature freeze T+20:00.**

### Phase 6 — T+20:00 → T+24:00 — Edit & submit
- Edit from `shots.md`. Voice-over, title cards, numbers card. Export 1080p/60s. Submit by T+23:30. Push README + HF dataset card.

---

## 6. The video — storyboard

| Time | Shot | Source |
|---|---|---|
| 0–7s | Dark water. Text: "1.3 million tonnes of wartime munitions lie under the North Sea." Cut to a wind-farm map. "We build wind farms on top of them." | title cards + map |
| 7–15s | AUV drops in, lawnmower survey begins; sonar waterfall builds on the side; murky camera view | `clear_sand` → `murky` runs |
| 15–30s | **Hero**: a contact appears. 2×2 imagined views from four angles fade in. Planner text: "North approach at 2 m: shadow would show fins → choose." AUV flies it; sonar reveals a finned cylinder; "UXO — aerial bomb, 0.91" with a red standoff circle | Tier 1/2 loop, `half_buried` |
| 30–40s | **Before/after**: zero-shot imagination (mush) vs fine-tuned (correct shadow) vs truth | 3-panel renderer |
| 40–50s | Fast cuts: tyre → "non-UXO", rock → "non-UXO", debris field, (Tier 3: plume avoided / 3 AUVs) | scenario runs |
| 50–56s | Numbers card: episodes generated, detector mAP before→after, inspections per contact before→after, classification accuracy | metrics |
| 56–60s | Master target list on the site map handed to "EOD team". "Look before you dive." Logos: Isaac Sim · Cosmos · LeRobot · Gemini · Modal. HF link | title card |

Shot rules: every run saves sonar, camera, chase-cam, map, imagined clips, planner JSON. Record 3× more than needed. The safety line ("it never touches anything; humans dispose") appears on screen once.

---

## 7. Repo layout

```
northsea-uxo-auv/
├── README.md
├── modal/
│   ├── image_oceansim.py       # Isaac Sim 5.0 + OceanSim, shader cache volume
│   ├── image_models.py         # Cosmos 3 + YOLO + Gemini client
│   └── volumes.py
├── sim/
│   ├── auv_env.py              # reset/step/render around OceanSim
│   ├── controller.py           # PID depth/heading/waypoint, station-keep
│   ├── objects.py              # UXO + decoy spawn library, burial, GT labels
│   ├── scenarios.py            # clear_sand, murky, ripples, half_buried, debris_field
│   ├── sonar_utils.py          # polar→cartesian, waterfall accumulation
│   ├── scripted_survey.py      # lawnmower + inspection orbits → episodes
│   └── generate_data.py        # Modal fan-out
├── envhub/env.py               # LeRobot EnvHub make_env()
├── perception/
│   ├── train_yolo.py           # drishti-sss + synthetic
│   └── detector.py
├── models/
│   ├── world_model_server.py   # /imagine
│   ├── convert_to_cosmos.py
│   ├── finetune_lora.py
│   └── planner.py              # Gemini 4 Argon → JSON
├── agent/mission_loop.py
├── data/{lerobot_writer.py, push_to_hub.py}
├── viz/{compose.py, overlays.py, compare.py, metrics.py}
└── video/{script.md, shots.md, title_cards/}
```

---

## 8. Risks, gates, fallbacks

| # | Risk | Likelihood | Detect | Fallback |
|---|---|---|---|---|
| F1 | OceanSim won't run headless on Modal (Isaac 5.0 install, Vulkan, EULA, missing demo vehicle) | Medium | Gate 1, hard stop T+5:30 | Plain Isaac Sim + Replicator: fog volume for turbidity, ray-cast "sonar" (depth + incidence-angle shading + speckle noise), hand-rolled buoyancy. Lower fidelity, same story. Last resort: HoloOcean (Unreal) — not NVIDIA, avoid unless desperate |
| F2 | Cosmos 3 gated access not granted in time | Medium | T+2:00 | Cosmos Predict2-2B (ungated); fine-tune the same way |
| F3 | Zero-shot sonar imagination is garbage | Near-certain | T+3:00 | That's the "before" footage. Tier 1 uses camera-modality imagination; Tier 2 fixes sonar |
| F4 | LoRA OOM / diverges on L40S | Medium | first 30 min | rank 8, 8-frame windows, 256p, batch 1 + grad accumulation; any checkpoint beating zero-shot is enough |
| F5 | Gemini latency too high for the loop | Medium | T+1:00 | Flash for proposals, Argon for final verdicts; cache per scenario |
| F6 | Shader compile on every container | High | first fan-out | Bake into image/Volume before Phase 3 |
| F7 | Sonar from sim looks nothing like real sonar → detector doesn't transfer | Medium | Phase 2 | Train detector mostly on real HF data; use synthetic only to add decoy classes. Add speckle/blur augmentation |
| F8 | Only 1–2 L40S allowed concurrently | Medium | T+0:30 | Data gen takes 3–6h on 1 GPU instead; start it at T+8 and run through the night; skip second LoRA config |
| F9 | Running out of credit | Low with this budget | every gate | Tier 3 is the first thing cut; then second LoRA config; then recording pass count |

---

## 9. First-hour checklist

- [ ] Open questions answered; L40S concurrency known
- [ ] Modal workspace, Volume, idle-timeouts policy agreed
- [ ] HF org + `northsea-uxo-survey-v1` repo; Cosmos 3 access requested
- [ ] NVIDIA developer / Omniverse account for Isaac assets
- [ ] Repo created; §2 interfaces pasted into README
- [ ] Sim: OceanSim repo cloned, installation.md read, Modal image building
- [ ] World model: Cosmos 3 Edge downloading to Volume
- [ ] Perception: `drishti-sss` downloaded; YOLO training launched
- [ ] Planner: Gemini key working; JSON schema test on a sonar tile
- [ ] Data & visuals: LeRobot dummy write; compose.py on placeholders
- [ ] Video: script v1; music shortlisted; `shots.md` created

---

## 10. Why this beats the alternatives (for the README / judges)

- **Real, current, UK-relevant problem** with real industry demand, not a toy.
- **World model used for active perception**, not just planning: "imagine the view, then go" is a genuinely different use of a WAM than arm manipulation.
- **Sonar is a domain foundation models have never seen**, which makes the synthetic-data + fine-tune story honest and visible in the before/after.
- **NVIDIA stack end to end** where it touches the robot: Isaac Sim, OceanSim, Cosmos, TensorRT detector. Hugging Face for data and glue.
- **Safety by design**: detection and prioritisation only; humans dispose.
