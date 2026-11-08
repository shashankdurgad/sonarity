"""Side experiment (P5): zero-shot imagination with NVIDIA Cosmos3-Edge.

Our robot imagines "bomb vs rock" with a hand-built sonar model (sim/visit.py) and goes to the side
where they differ most. This asks a general world model, with no training on our scene, to do the
same imagining from the robot's camera view, starting from a real chase-view frame over empty seabed.

Run 1 asked the camera to circle round the object; Cosmos ignored that and always drove straight in,
so the viewpoint never changed. Run 2 (this file) keeps the robot's motion identical (straight in,
stop) and turns the OBJECT instead, which is the same as viewing it from a different side:
  side_on: the bomb lies ACROSS the robot's path (full length visible)
  end_on:  the bomb points straight AT the robot, nose first
A rock looks the same from every side, so one rock video per seed serves both views.
Score = how different the imagined bomb and rock frames are; our on-board model predicts side_on wins.

Runs on the person's own Modal account (L40S). Needs a Modal secret "huggingface" with HF_TOKEN.

    modal run experiments/cosmos/zero_shot.py      # writes experiments/cosmos/results/run2/
    python3 experiments/cosmos/score.py            # scores + grid (needs numpy, Pillow)
"""
import io
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "run2")
SEEDS = [0, 1, 2]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install("diffusers @ git+https://github.com/huggingface/diffusers.git", "accelerate", "av",
                 "cosmos_guardrail", "huggingface_hub", "imageio", "imageio-ffmpeg", "torch", "torchvision",
                 "transformers", "pillow", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
app = modal.App("p5-cosmos-zero-shot")
cache = modal.Volume.from_name("cosmos-cache", create_if_missing=True)

MOTION = ("The robot glides slowly straight forward and stops about 4 metres from the object, "
          "which stays clearly visible in the centre of the frame, ahead of the robot.")
OBJECTS = {
    "bomb_side_on": ("A rusty World War II aerial bomb lying on the sandy seabed ACROSS the robot's path, its long axis "
                     "running from the left of the frame to the right, so its full length is seen side-on: a long, "
                     "thin, dark grey metal cylinder about 1.9 m long and 0.5 m wide, rounded nose on the left, four "
                     "flat tail fins on the right."),
    "bomb_end_on": ("A rusty World War II aerial bomb lying on the sandy seabed pointing STRAIGHT AT the robot, its "
                    "rounded nose facing the camera and its body extending directly away from the viewer, so it is "
                    "seen end-on and looks short and compact: a dark grey metal cylinder 0.5 m wide, tail fins "
                    "partly hidden behind the nose."),
    "rock": ("A rough, rounded grey boulder about 1 m across lying on the sandy seabed, irregular and lumpy, with "
             "no straight edges."),
}


def prompt(key: str) -> dict:
    """Prompt in the structured JSON form Cosmos3 expects (fields as in NVIDIA's example)."""
    obj = OBJECTS[key]
    return {
        "subjects": [
            {"description": "A small yellow autonomous underwater survey robot seen from just behind and above, "
                            "in the lower centre of the frame.",
             "relationship": "Point-of-view vehicle surveying the seabed", "location": "Bottom centre foreground",
             "action": MOTION, "number_of_subjects": 1},
            {"description": obj, "relationship": "The object the robot is inspecting",
             "location": "On the seabed in the centre of the frame, ahead of the robot",
             "action": "Lies completely still on the seabed", "number_of_subjects": 1},
        ],
        "background_setting": "Flat sandy seabed about 20 m deep in murky blue-green water, visibility about 10 m.",
        "lighting": {"conditions": "Dim, diffuse underwater light", "direction": "From above, scattered by the water"},
        "cinematography": {"camera_motion": "Follows the robot smoothly from behind, moving straight forward",
                           "framing": "Wide shot", "camera_angle": "Slightly above and behind the robot"},
        "style_medium": "Live-action video", "artistic_style": "Realistic underwater survey footage",
        "context": "An underwater robot inspecting an unknown object on the seabed during a mine-clearance survey.",
        "temporal_caption": f"{MOTION} {obj}",
        "resolution": {"W": 832, "H": 480}, "aspect_ratio": "16,9", "duration": "5s", "fps": 12,
    }


@app.function(image=image, gpu="L40S", timeout=2400, volumes={"/cache": cache},
              secrets=[modal.Secret.from_name("huggingface")])
def imagine(start_png: bytes) -> dict:
    import time

    import numpy as np
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from huggingface_hub import hf_hub_download
    from PIL import Image

    t0 = time.time()
    try:
        pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Edge", torch_dtype=torch.bfloat16,
                                                   enable_safety_checker=True)
        guard = True
    except Exception as e:  # guardrail weights unavailable: run without NVIDIA's content filter
        print(f"[cosmos] safety checker unavailable ({e!r}); continuing without it", flush=True)
        pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Edge", torch_dtype=torch.bfloat16,
                                                   enable_safety_checker=False)
        guard = False
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=12.0,
                                                         use_karras_sigmas=False)
    neg = json.load(open(hf_hub_download("nvidia/Cosmos3-Edge", "assets/negative_prompt.json")))
    cache.commit()
    load_s = time.time() - t0
    print(f"[cosmos] loaded in {load_s:.0f}s (guardrail {guard})", flush=True)

    start = Image.open(io.BytesIO(start_png)).convert("RGB")
    out = {"load_s": round(load_s, 1), "guardrail": guard, "videos": {}, "last": {}, "gen_s": {}}
    for seed in SEEDS:
        for key in OBJECTS:
            t = time.time()
            res = pipe(prompt=json.dumps(prompt(key)), negative_prompt=json.dumps(neg), image=start,
                       num_frames=61, height=480, width=832, fps=12.0, num_inference_steps=20,
                       guidance_scale=6.0, generator=torch.Generator(device="cuda").manual_seed(seed),
                       add_resolution_template=False, add_duration_template=False)
            frames = res.video[0] if isinstance(res.video[0], list) else res.video
            name = f"{key}_s{seed}"
            path = f"/tmp/{name}.mp4"
            export_to_video(frames, path, fps=12, macro_block_size=1)
            last = frames[-1]
            if not isinstance(last, Image.Image):
                a = np.asarray(last)
                last = Image.fromarray((a * 255 if a.max() <= 1 else a).astype("uint8"))
            buf = io.BytesIO()
            last.save(buf, format="PNG")
            out["videos"][name] = open(path, "rb").read()
            out["last"][name] = buf.getvalue()
            out["gen_s"][name] = round(time.time() - t, 1)
            print(f"[cosmos] {name}: {out['gen_s'][name]}s", flush=True)
    return out


@app.local_entrypoint()
def main():
    os.makedirs(OUT, exist_ok=True)
    res = imagine.remote(open(os.path.join(HERE, "start.png"), "rb").read())
    for k, v in res["videos"].items():
        open(os.path.join(OUT, f"{k}.mp4"), "wb").write(v)
    for k, v in res["last"].items():
        open(os.path.join(OUT, f"{k}_last.png"), "wb").write(v)
    meta = {"model": "nvidia/Cosmos3-Edge", "gpu": "L40S", "seeds": SEEDS, "load_s": res["load_s"],
            "guardrail": res["guardrail"], "gen_s": res["gen_s"]}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
