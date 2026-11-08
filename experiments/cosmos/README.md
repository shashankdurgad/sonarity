# Side experiment: zero-shot imagination with NVIDIA Cosmos 3 Edge (P5)

**Question:** our robot picks its inspection side by imagining "bomb vs rock" with a small hand-built
sonar model (`sim/visit.py`, `agent/imagine.py`, under 1 s on the robot). Could a general world model,
with no training on our scene, do that imagining instead?

**Model:** `nvidia/Cosmos3-Edge` (4B, open, built to run on the robot's own GPU), image-to-video via
Diffusers, on one L40S (Modal, personal account). About 17 s per imagined 5 s clip (61 frames, 832x480).
Input: a real chase-view frame from mission `m3full_clear_sand_s3` (`start.png`). It imagines camera views,
not sonar.

## Results

| Run | What we asked | What happened |
|---|---|---|
| 1 (`results/`) | Bomb or rock on the seabed; camera **circles round** to a side-on or end-on view | Bomb and rock were imagined well **side-on** (`results/grid.png`), but Cosmos ignored the "circle round" instruction and always drove straight in. In the "end-on" bomb clip the bomb vanished. Viewpoint not controlled. |
| 2 (`results/run2/`) | Same straight-in motion; the **bomb is turned** (across the path vs nose-on); 3 seeds | The object is **missing in 8 of 9 clips** (no bomb in any bomb clip; a faint rock in one rock clip). Scores are background noise; marked invalid in `results/run2/summary.json`. |

**Conclusion:** zero-shot, Cosmos 3 Edge can imagine our seabed with a bomb or a rock, but it can't
reliably place a given object in a given pose or show it from a chosen side, which is what choosing an
inspection angle needs. This supports keeping our fast, reliable on-board sonar model; fine-tuning a world
model like Cosmos on our simulator data is the natural next step.

Notes: NVIDIA's content-safety guardrail (`nvidia/Cosmos-1.0-Guardrail`) is gated and wasn't available, so
it was skipped (prompts are plain seabed scenes). Pixel-difference scoring is crude and the run is small.

## Re-run

```bash
modal run experiments/cosmos/zero_shot.py   # needs a Modal secret "huggingface" (HF_TOKEN); writes results/run2/
python3 experiments/cosmos/score.py         # scores + grid
```
