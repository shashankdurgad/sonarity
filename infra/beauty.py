"""Cinematic RTX beauty replay of a finished mission, rendered on Modal L40S (see render/beauty.py).

Look test (a few frames, compare render modes):
    python -P -m modal run infra/beauty.py --run-id p3scene_clear_sand_s3 --frames 30,170,400 --variants PathTracing,RaytracedLighting
Full render (frames split across GPUs) + HUD composite -> runs/<id>/beauty/<id>_beauty.mp4:
    python -P -m modal run infra/beauty.py --run-id hero_clear_sand_s5 --shards 8
Only re-composite (after HUD tweaks):
    python -P -m modal run infra/beauty.py --run-id hero_clear_sand_s5 --compose-only
"""
import os
import pathlib
import subprocess
import sys

import modal

sys.path.insert(0, os.path.dirname(__file__))
from image import CACHE_MOUNT, cache_volume, isaac_image, link_caches  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("uxo-beauty")
runs_volume = modal.Volume.from_name("uxo-runs", create_if_missing=True)
render_image = isaac_image.add_local_dir(REPO / "render", "/root/repo/render", ignore=["__pycache__"])
compose_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core", "curl")
    .pip_install("numpy<2", "pillow")
    .run_commands("mkdir -p /fonts && curl -fsSL -o /fonts/Rajdhani-SemiBold.ttf "
                  "https://github.com/google/fonts/raw/main/ofl/rajdhani/Rajdhani-SemiBold.ttf || true",
                  "curl -fsSL -o /fonts/Rajdhani-Bold.ttf "
                  "https://github.com/google/fonts/raw/main/ofl/rajdhani/Rajdhani-Bold.ttf || true",
                  "curl -fsSL -o /fonts/ShareTechMono.ttf "
                  "https://github.com/google/fonts/raw/main/ofl/sharetechmono/ShareTechMono-Regular.ttf || true")
    .add_local_dir(REPO / "render", "/root/repo/render", ignore=["__pycache__"])
    .add_local_dir(REPO / "sim", "/root/repo/sim", ignore=["__pycache__"])
    .add_local_python_source("image")  # this module imports infra/image.py at load time
)


@app.function(image=render_image, gpu="L40S", timeout=3600 * 3, cpu=8, memory=32768,
              volumes={CACHE_MOUNT: cache_volume, "/runs": runs_volume})
def render_shard(run_id: str, start: int = 0, end: int = -1, frames: str = "", variants: str = "",
                 spp: int = 48, mode: str = "PathTracing", width: int = 1920, height: int = 1080,
                 out_sub: str = "frames", dump: bool = False, debug_asset: str = "") -> str:
    link_caches()
    runs_volume.reload()
    out = f"/runs/{run_id}/beauty/{out_sub}"
    cmd = ["/isaac-sim/python.sh", "render/beauty.py", "--run", f"/runs/{run_id}", "--out", out,
           "--start", str(start), "--end", str(end), "--spp", str(spp), "--mode", mode,
           "--width", str(width), "--height", str(height)]
    if frames:
        cmd += ["--frames", frames]
    if variants:
        cmd += ["--variants", variants]
    if dump:
        cmd += ["--dump-settings"]
    if debug_asset:
        cmd += ["--debug-asset", debug_asset]
    p = subprocess.run(cmd, cwd="/root/repo", capture_output=True, text=True)
    lines = [ln for ln in (p.stdout + p.stderr).splitlines()
             if ln.startswith(("[beauty", "SETTING", "Traceback", "  File", "Error", "RuntimeError", "ValueError",
                               "KeyError", "TypeError", "AttributeError", "NameError", "IndexError"))
             or "Error" in ln[:60]]
    cache_volume.commit()
    runs_volume.commit()
    return f"[shard {start}-{end} rc={p.returncode}]\n" + "\n".join(lines[-80:])


@app.function(image=compose_image, timeout=3600, cpu=16, memory=32768, volumes={"/runs": runs_volume})
def compose(run_id: str, out_sub: str = "frames") -> str:
    runs_volume.reload()
    p = subprocess.run([sys.executable, "render/hud.py", "--run", f"/runs/{run_id}", "--frames", f"/runs/{run_id}/beauty/{out_sub}"],
                       cwd="/root/repo", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/root/repo"})
    runs_volume.commit()
    return (p.stdout + p.stderr)[-4000:]


@app.function(image=compose_image, timeout=600, volumes={"/runs": runs_volume})
def n_frames(run_id: str) -> int:
    sys.path.insert(0, "/root/repo")
    from render.timeline import Timeline

    return Timeline(f"/runs/{run_id}").n_frames


@app.local_entrypoint()
def main(run_id: str, shards: int = 8, frames: str = "", variants: str = "", spp: int = 48,
         mode: str = "PathTracing", width: int = 1920, height: int = 1080, compose_only: bool = False,
         dump: bool = False, no_compose: bool = False, debug_asset: str = ""):
    if frames:  # look test
        print(render_shard.remote(run_id, frames=frames, variants=variants, spp=spp, mode=mode,
                                  width=width, height=height, out_sub="look", dump=dump, debug_asset=debug_asset))
        print(f"look frames: modal volume get uxo-runs {run_id}/beauty/look <dir>")
        return
    if not compose_only:
        n = n_frames.remote(run_id)
        size = -(-n // shards)
        print(f"{run_id}: {n} frames -> {shards} shards of {size}")
        calls = [render_shard.spawn(run_id, i * size, min((i + 1) * size, n), spp=spp, mode=mode,
                                    width=width, height=height) for i in range(shards)]
        for c in calls:
            try:
                print(c.get())
            except Exception as e:  # noqa: BLE001
                print("shard failed:", repr(e))
    if not no_compose:
        print(compose.remote(run_id))
