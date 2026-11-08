"""Live mission on Modal (owner: P2). See LIVE_MISSION.md.

Start a mission (downloads runs/mission_<scenario>_s<seed>/ when it finishes):
    python -P -m modal run infra/mission.py::main --seed 3 [--lanes 2] [--auto "uncover_bomb@60"]
While it runs, from another terminal:
    python -P -m modal run infra/mission.py::status
    python -P -m modal run infra/mission.py::trigger --event uncover_bomb     # or drop_object --cls rock, murk_on, murk_off
"""
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time

import modal

sys.path.insert(0, os.path.dirname(__file__))
from image import CACHE_MOUNT, cache_volume, isaac_image, link_caches  # noqa: E402

app = modal.App("uxo-mission")
runs_volume = modal.Volume.from_name("uxo-runs", create_if_missing=True)
EVENTS = "uxo-mission-events"


@app.function(image=isaac_image, gpu="L40S", timeout=3600, volumes={CACHE_MOUNT: cache_volume, "/runs": runs_volume},
              secrets=[modal.Secret.from_name("gemini")])  # GEMINI_API_KEY for agent.planner.classify_contact
def mission(run_id: str, seed: int, scenario: str, lanes: int, auto: list[str], use_yolo: bool = True,
            extra: list[str] = (), bus_name: str = EVENTS) -> bytes:
    link_caches()
    bus = modal.Dict.from_name(bus_name, create_if_missing=True)
    bus.clear()
    bus["status"] = {"state": "STARTING", "run_id": run_id}
    out, ev_file, st_file = f"/runs/{run_id}", "/tmp/mission_events.jsonl", "/tmp/mission_status.json"
    os.makedirs(out, exist_ok=True)
    open(ev_file, "w").close()
    cmd = ["/isaac-sim/python.sh", "sim/mission.py", "--out", out, "--seed", str(seed), "--scenario", scenario,
           "--lanes", str(lanes), "--events-file", ev_file, "--status-file", st_file]
    for a in auto:
        cmd += ["--auto-event", a]
    cmd += list(extra)
    if use_yolo:  # Isaac's Python has no torch/modal: proxy frames to the deployed uxo-detector
        sys.path.insert(0, "/root/repo")
        from perception.detect_bridge import serve
        print(f"[mission] YOLO detector via {serve()}", flush=True)
    proc = subprocess.Popen(cmd, cwd="/root/repo")

    def relay():  # Modal Dict <-> files the Isaac process reads/writes
        while proc.poll() is None:
            for k in [k for k in bus.keys() if str(k).startswith("ev_")]:
                ev = bus.pop(k)
                with open(ev_file, "a") as f:
                    f.write(json.dumps(ev) + "\n")
                print(f"[relay] event {ev}", flush=True)
            if os.path.exists(st_file):
                try:
                    bus["status"] = {**json.load(open(st_file)), "run_id": run_id}
                except (ValueError, OSError):
                    pass
            time.sleep(0.5)

    threading.Thread(target=relay, daemon=True).start()
    rc = proc.wait()
    bus["status"] = {"state": "FINISHED", "rc": rc, "run_id": run_id}
    cache_volume.commit()
    runs_volume.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=run_id)
    return buf.getvalue()


@app.local_entrypoint()
def main(seed: int = 3, scenario: str = "clear_sand", lanes: int = 0, auto: str = "", tag: str = "mission",
         standin: bool = False, no_revisit: bool = False, force_uncertain: bool = False, private_bus: bool = False):
    """--standin uses the blob stand-in instead of YOLO (needs: modal deploy perception/detector_service.py).
    --no-revisit: don't go back to uncertain contacts at the end. --force-uncertain: TEST ONLY.
    --private-bus: use an event bus of its own (uxo-mission-events-<run_id>) so unattended test runs
    (--auto uncover_bomb@90,...) can't clear or steal events meant for a mission someone is recording."""
    run_id = f"{tag}_{scenario}_s{seed}"
    data = mission.remote(run_id, seed, scenario, lanes, [a for a in auto.split(",") if a], not standin,
                          ["--no-revisit"] * no_revisit + ["--force-uncertain"] * force_uncertain,
                          f"{EVENTS}-{run_id}" if private_bus else EVENTS)
    os.makedirs("runs", exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall("runs", filter="data")
    print(f"Saved runs/{run_id}/ ({len(data) / 1e6:.1f} MB) - watch runs/{run_id}/mission.mp4")


@app.local_entrypoint()
def trigger(event: str = "uncover_bomb", cls: str = "aerial_bomb", level: float = 0.75):
    bus = modal.Dict.from_name(EVENTS, create_if_missing=True)
    bus[f"ev_{time.time():.3f}"] = {"type": event, "cls": cls, "level": level}
    print(f"sent {event}; mission status: {bus.get('status')}")


@app.local_entrypoint()
def status():
    print(modal.Dict.from_name(EVENTS, create_if_missing=True).get("status"))
