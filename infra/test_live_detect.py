"""Checks for the live detector (LIVE_MISSION.md, P1).

compare: on held-out surveys, does detect_frame + ContactTracker (live) match run() (batch), and how
         long does one detect_frame call take on an L40S?
isaac:   can ultralytics run inside Isaac 6.0.1's own Python without changing its torch?

python -P -m modal run infra/test_live_detect.py            # both
python -P -m modal run infra/test_live_detect.py --only compare
"""
import json
import os
import pathlib
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image import isaac_image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS = "/runs/yolo_run2/weights/best.pt"
SURVEYS = ["train_clear_sand_s106", "train_half_buried_s112"]
runs_vol = modal.Volume.from_name("uxo-runs")

slim = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.3.40", "numpy<2")
    .add_local_dir(REPO / "perception", "/root/perception")
    .add_local_dir(REPO / "sim", "/root/sim")
    .add_local_python_source("image")
)
app = modal.App("uxo-test-live-detect")


@app.function(image=slim, gpu="L40S", timeout=1800, volumes={"/runs": runs_vol})
def compare() -> dict:
    import csv
    import shutil
    import time

    import numpy as np
    from PIL import Image

    sys.path.insert(0, "/root")
    from perception.detect import ContactTracker, detect_frame, load_model, run

    load_model(WEIGHTS)
    out = {}
    for name in SURVEYS:
        src = pathlib.Path("/runs") / name
        work = pathlib.Path("/tmp") / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(src, work, ignore=shutil.ignore_patterns("cam", "chase", "crops", "contacts.json"))
        batch = run(str(work), weights=WEIGHTS)

        tracker, times, events = ContactTracker(), [], []
        with open(work / "poses.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            sonar = np.array(Image.open(work / row["frame"]).convert("L"))
            t0 = time.perf_counter()
            dets = detect_frame(sonar, row, weights=WEIGHTS)
            times.append(time.perf_counter() - t0)
            for c in tracker.update(dets, row["frame"], sonar):
                if c["status"] == "new":
                    events.append({"frame": row["frame"], "id": c["id"], "cls": c["cls"], "xy": c["xy"]})
        live_dir = pathlib.Path("/tmp") / (name + "_live")
        live = tracker.save(str(live_dir))
        assert (live_dir / "crops" / "c1.png").exists() or not live

        world = json.loads((src / "world.json").read_text())
        objs = world["objects"] if isinstance(world, dict) else world

        def score(contacts):
            res = []
            for o in objs:
                d = min((float(np.hypot(c["xy"][0] - o["xyz"][0], c["xy"][1] - o["xyz"][1])) for c in contacts),
                        default=None)
                res.append({"obj": o.get("id"), "cls": o.get("cls"), "nearest_m": round(d, 2) if d is not None else None})
            return res

        t = np.array(times[3:])  # skip warm-up
        out[name] = {
            "frames": len(rows),
            "batch": [{k: c[k] for k in ("id", "cls", "conf", "xy")} for c in batch],
            "live": [{k: c[k] for k in ("id", "cls", "conf", "xy", "hits")} for c in live],
            "batch_vs_truth": score(batch), "live_vs_truth": score(live),
            "first_confirmed": events,
            "detect_frame_ms": {"median": round(float(np.median(t)) * 1000, 1), "p95": round(float(np.percentile(t, 95)) * 1000, 1)},
        }
    return out


ISAAC_CHECK = r"""
import importlib.metadata as md, sys, time
import torch
print("ISAAC_PY", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", md.version("numpy"))
from ultralytics import YOLO
import numpy as np
m = YOLO("/runs/yolo_run2/weights/best.pt")
img = np.zeros((512, 512, 3), np.uint8)
m.predict(img, verbose=False)
t0 = time.perf_counter()
for _ in range(20):
    m.predict(img, verbose=False, imgsz=640)
print("ISAAC_YOLO_MS", round((time.perf_counter() - t0) / 20 * 1000, 1), "torch after", torch.__version__)
"""


@app.function(image=isaac_image, gpu="L40S", timeout=1800, volumes={"/runs": runs_vol})
def isaac() -> str:
    import subprocess

    def sh(cmd):
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"$ {cmd}\n[exit {p.returncode}]\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"

    py = "/isaac-sim/python.sh"
    # Isaac only puts torch on sys.path via an extension; find it so plain python.sh can import it.
    log = [sh("find /isaac-sim/exts /isaac-sim/extscache /isaac-sim/kit -maxdepth 4 -type d -name torch -path '*prebundle*' 2>/dev/null")]
    found = subprocess.run("find /isaac-sim/exts /isaac-sim/extscache /isaac-sim/kit -maxdepth 4 -type d -name torch -path '*prebundle*' 2>/dev/null | head -1",
                           shell=True, capture_output=True, text=True).stdout.strip()
    if found:
        os.environ["PYTHONPATH"] = os.path.dirname(found) + ":" + os.environ.get("PYTHONPATH", "")
    log.append(sh(f"{py} -c 'import torch, numpy; print(torch.__version__, torch.__file__, numpy.__version__)'"))
    # --no-deps so pip can't swap Isaac's torch/numpy; then add only the small pure deps ultralytics needs.
    log.append(sh(f"{py} -m pip install --no-deps ultralytics==8.3.40 ultralytics-thop"))
    log.append(sh(f"{py} -m pip install --no-deps opencv-python-headless py-cpuinfo pandas seaborn tqdm pyyaml psutil requests matplotlib"))
    open("/tmp/check.py", "w").write(ISAAC_CHECK)
    log.append(sh(f"{py} /tmp/check.py"))
    return "\n".join(log)


@app.function(image=slim, timeout=1800, volumes={"/runs": runs_vol})
def service(name: str = SURVEYS[0]) -> dict:
    """Mission-like: CPU container calls the deployed uxo-detector each frame; tracker runs here."""
    import csv
    import time

    import numpy as np
    from PIL import Image

    sys.path.insert(0, "/root")
    from perception.detect import ContactTracker
    from perception.detector_service import remote_detect_frame

    src = pathlib.Path("/runs") / name
    rows = list(csv.DictReader(open(src / "poses.csv")))
    tracker, times = ContactTracker(), []
    for row in rows:
        sonar = np.array(Image.open(src / row["frame"]).convert("L"))
        t0 = time.perf_counter()
        dets = remote_detect_frame(sonar, row)
        times.append(time.perf_counter() - t0)
        tracker.update(dets, row["frame"], sonar)
    t = np.array(times[3:])
    return {"survey": name, "first_call_s": round(times[0], 2),
            "round_trip_ms": {"median": round(float(np.median(t)) * 1000), "p95": round(float(np.percentile(t, 95)) * 1000)},
            "contacts": [(c["id"], c["cls"], c["xy"], c["hits"]) for c in tracker.contacts]}


ISAAC_TORCH_CHECK = r"""
import time, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()
UsdGeom.Cube.Define(stage, "/World/Cube")
for _ in range(10):
    app.update()
import torch, numpy as np
from ultralytics import YOLO
print("ISAAC_PY", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", np.__version__, flush=True)
m = YOLO("/runs/yolo_run2/weights/best.pt")
img = np.zeros((512, 512, 3), np.uint8)
m.predict(img, verbose=False, imgsz=640)
t0 = time.perf_counter()
for _ in range(20):
    m.predict(img, verbose=False, imgsz=640)
    app.update()
print("ISAAC_YOLO_PLUS_UPDATE_MS", round((time.perf_counter() - t0) / 20 * 1000, 1), flush=True)
import warp as wp
wp.init(); a = wp.zeros(10, device="cuda"); print("WARP_OK", a.device, flush=True)
app.close()
print("ISAAC_TORCH_DONE", flush=True)
"""

TORCH_PIP = "torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126"


@app.function(image=isaac_image, gpu="L40S", timeout=2400, volumes={"/runs": runs_vol})
def isaac_torch() -> str:
    import subprocess

    def sh(cmd):
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"$ {cmd}\n[exit {p.returncode}]\n{p.stdout[-2500:]}\n{p.stderr[-2500:]}"

    py = "/isaac-sim/python.sh"
    log = [sh(f"{py} -c 'import numpy; print(numpy.__version__)'"),
           sh(f"{py} -m pip install -q {TORCH_PIP}"),
           sh(f"{py} -m pip install -q --no-deps ultralytics==8.3.40 ultralytics-thop py-cpuinfo pandas seaborn tqdm"),
           sh(f"{py} -c 'import numpy; print(\"numpy after\", numpy.__version__)'")]
    open("/tmp/check2.py", "w").write(ISAAC_TORCH_CHECK)
    log.append(sh(f"{py} /tmp/check2.py 2>&1 | grep -E 'ISAAC_|WARP_OK|Error|error:|Traceback|Segmentation' | head -40"))
    return "\n".join(log)


@app.local_entrypoint()
def main(only: str = "") -> None:
    calls = {}
    if only in ("", "compare"):
        calls["compare"] = compare.spawn()
    if only == "isaac_torch":
        calls["isaac_torch"] = isaac_torch.spawn()
    if only == "bridge":
        calls["bridge"] = bridge.spawn()
    if only == "service":
        calls["service"] = service.spawn()
    if only in ("", "isaac"):
        calls["isaac"] = isaac.spawn()
    for name, call in calls.items():
        try:
            res = call.get()
        except Exception as e:  # report and keep going
            res = f"FAILED: {e!r}"
        print(f"===== {name} =====")
        print(json.dumps(res, indent=2) if isinstance(res, dict) else res)


@app.function(image=isaac_image, timeout=600, volumes={"/runs": runs_vol})
def bridge() -> str:
    """Same setup as infra/mission.py: proxy in this (bare) python, client in Isaac's python."""
    import subprocess

    sys.path.insert(0, "/root/repo")
    from perception.detect_bridge import serve

    serve()
    code = ("import sys; sys.path.insert(0, '/root/repo'); import numpy as np; from PIL import Image; "
            "from perception.detect_bridge import bridge_detect_frame as f; "
            "s = np.array(Image.open('/runs/train_clear_sand_s106/sonar/000060.png').convert('L')); "
            "print('BRIDGE_DETS', f(s, {'x': 0, 'y': 0, 'yaw': 0}))")
    p = subprocess.run(["/isaac-sim/python.sh", "-c", code], capture_output=True, text=True, cwd="/root/repo")
    return p.stdout[-2000:] + p.stderr[-2000:]
