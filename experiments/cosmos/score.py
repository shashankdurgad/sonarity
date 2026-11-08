"""Score run 2 of the Cosmos zero-shot experiment: how different do bomb and rock look, side-on vs end-on?

    python3 experiments/cosmos/score.py      # reads results/run2/, writes summary.json and grid.png there
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "run2")
meta = json.load(open(os.path.join(RUN, "meta.json")))
seeds = meta["seeds"]


def grey(name):
    im = Image.open(os.path.join(RUN, f"{name}_last.png")).convert("L").filter(ImageFilter.GaussianBlur(4))
    return np.asarray(im, float) / 255


per_seed = {view: [round(float(np.abs(grey(f"bomb_{view}_s{s}") - grey(f"rock_s{s}")).mean()), 4) for s in seeds]
            for view in ("side_on", "end_on")}
mean = {v: round(float(np.mean(x)), 4) for v, x in per_seed.items()}
wins = sum(a > b for a, b in zip(per_seed["side_on"], per_seed["end_on"]))
summary = {**meta, "bomb_vs_rock_difference_per_seed": per_seed, "mean": mean,
           "side_on_wins": f"{wins}/{len(seeds)} seeds", "zero_shot_choice": max(mean, key=mean.get),
           "on_board_model_predicts": "side_on"}
json.dump(summary, open(os.path.join(RUN, "summary.json"), "w"), indent=2)
print(json.dumps({k: summary[k] for k in ("bomb_vs_rock_difference_per_seed", "mean", "side_on_wins",
                                          "zero_shot_choice")}, indent=2))

# grid: rows = seeds, columns = bomb side-on | bomb end-on | rock
tw, th = 384, 222
G = Image.new("RGB", (3 * tw + 40, len(seeds) * (th + 30) + 30), (20, 20, 30))
d = ImageDraw.Draw(G)
for c, label in enumerate(["bomb side-on", "bomb end-on", "rock"]):
    d.text((c * (tw + 20) + 6, 8), label, fill=(255, 255, 255))
for r, s in enumerate(seeds):
    for c, name in enumerate([f"bomb_side_on_s{s}", f"bomb_end_on_s{s}", f"rock_s{s}"]):
        G.paste(Image.open(os.path.join(RUN, f"{name}_last.png")).convert("RGB").resize((tw, th)),
                (c * (tw + 20), 30 + r * (th + 30)))
G.save(os.path.join(RUN, "grid.png"))
