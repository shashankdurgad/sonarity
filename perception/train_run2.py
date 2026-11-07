"""YOLO run 2: continue from the real-sonar weights on labelled OceanSim frames.

Splits by seed: val = 105, 111; test = 106, 112; train = every other survey.
Run:     python -P -m modal run --detach perception/train_run2.py
Dry run: python -P -m modal run perception/train_run2.py --dry-run
Output:  uxo-runs Volume, yolo_run2/ (weights/best.pt, results.json, contacts_check.json)
"""
import json
import pathlib
import random
import re
import shutil

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.3.40", "numpy<2")
    .add_local_dir(REPO / "perception", "/root/perception")
    .add_local_dir(REPO / "sim", "/root/sim")
)
app = modal.App("uxo-yolo-run2", image=image)
runs_vol = modal.Volume.from_name("uxo-runs", create_if_missing=True)
RUNS = pathlib.Path("/runs")
VAL, TEST = {105, 111}, {106, 112}
START = RUNS / "yolo_run1_1024/weights/best.pt"
NAMES = ["mine_like", "other"]


def surveys(dry: bool) -> dict[str, list[pathlib.Path]]:
    out = {"train": [], "val": [], "test": []}
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "labels").exists()):
        if d.name.endswith("_quick") and not dry:
            continue
        m = re.search(r"_s(\d+)", d.name)
        seed = int(m.group(1)) if m else -1
        out["val" if seed in VAL else "test" if seed in TEST else "train"].append(d)
    if dry:
        out["val"] = out["test"] = out["train"][:1]
    return out


def add(split_dir: pathlib.Path, survey: pathlib.Path, keep_empty_ratio: float | None, rng: random.Random) -> dict:
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    labelled, empty = [], []
    for lab in sorted((survey / "labels").glob("*.txt")):
        if (survey / "sonar" / (lab.stem + ".png")).exists():
            (labelled if lab.stat().st_size else empty).append(lab)
    if keep_empty_ratio is not None:
        rng.shuffle(empty)
        empty = empty[: int(len(labelled) * keep_empty_ratio)]
    boxes = [0, 0]
    for lab in labelled + empty:
        for line in lab.read_text().splitlines():
            boxes[int(line.split()[0])] += 1
        stem = f"{survey.name}_{lab.stem}"
        shutil.copy(lab, split_dir / "labels" / f"{stem}.txt")
        shutil.copy(survey / "sonar" / f"{lab.stem}.png", split_dir / "images" / f"{stem}.png")
    return {"labelled": len(labelled), "empty": len(empty), "mine_like_boxes": boxes[0], "other_boxes": boxes[1]}


@app.function(gpu="L40S", timeout=3600, cpu=8, memory=32768, volumes={"/runs": runs_vol})
def train(dry_run: bool = False) -> dict:
    import sys

    from ultralytics import YOLO

    sys.path.insert(0, "/root")
    from perception.detect import run as detect_run

    runs_vol.reload()
    name = "yolo_run2_dry" if dry_run else "yolo_run2"
    rng = random.Random(0)
    ds = pathlib.Path("/tmp/ds")
    stats = {}
    groups = surveys(dry_run)
    for split, items in groups.items():
        stats[split] = {"surveys": [s.name for s in items]}
        for survey in items:
            one = add(ds / split, survey, 1.0 if split == "train" else None, rng)
            for k, v in one.items():
                stats[split][k] = stats[split].get(k, 0) + v
    print("DATA", json.dumps(stats), flush=True)
    (ds / "data.yaml").write_text(f"path: {ds}\ntrain: train/images\nval: val/images\ntest: test/images\nnames: {NAMES}\n")

    model = YOLO(str(START))
    model.train(data=str(ds / "data.yaml"), epochs=2 if dry_run else 60, imgsz=640, batch=32, patience=20, workers=8,
                project="/tmp/out", name=name, exist_ok=True, verbose=False, plots=True,
                hsv_h=0.0, hsv_s=0.0, fliplr=0.5, flipud=0.0, degrees=0.0)
    best_path = f"/tmp/out/{name}/weights/best.pt"
    best = YOLO(best_path)
    m = best.val(data=str(ds / "data.yaml"), split="test", imgsz=640, project="/tmp/out", name=f"{name}_test", plots=True)
    per_class = {NAMES[int(c)]: round(float(ap), 3) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    result = {"data": stats, "test_map50": round(float(m.box.map50), 3), "test_map50_95": round(float(m.box.map), 3),
              "test_ap50_per_class": per_class, "test_precision": round(float(m.box.mp), 3),
              "test_recall": round(float(m.box.mr), 3)}

    out = RUNS / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(f"/tmp/out/{name}", out)
    shutil.copytree(f"/tmp/out/{name}_test", out / "test")

    # Check perception/detect.py end to end on one held-out survey, against ground truth.
    check_src = groups["test"][0]
    check = pathlib.Path("/tmp/check")
    shutil.copytree(check_src, check, ignore=shutil.ignore_patterns("cam", "chase", "*.mp4", "_preview*", "contacts.json", "crops"))
    contacts = detect_run(str(check), weights=best_path)
    world = json.loads((check / "world.json").read_text())
    result["detect_check"] = {"survey": check_src.name, "contacts": contacts,
                              "truth": [{"cls": o["cls"], "xy": o["xyz"][:2]} for o in world["objects"]]}
    (out / "results.json").write_text(json.dumps(result, indent=1))
    runs_vol.commit()
    print("RESULT", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    print(json.dumps(train.remote(dry_run=dry_run), indent=1))
