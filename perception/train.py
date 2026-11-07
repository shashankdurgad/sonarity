"""YOLO run 1: YOLOv8n on real MILCO/NOMBO side-scan sonar (figshare 24574879, CC BY 4.0).

Classes: 0 mine_like (MILCO), 1 other (NOMBO), as in the dataset's obj.names.
Splits are by survey year, so near-identical frames can't leak between train and test.

Run: modal run --detach perception/train.py --name yolo_run1 --imgsz 640
Output: uxo-runs Volume, yolo_run1/ (best.pt, results.json, plots)
"""
import json
import pathlib
import shutil

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.3.40", "numpy<2")
)
app = modal.App("uxo-yolo-run1", image=image)
data_vol = modal.Volume.from_name("uxo-data")
runs_vol = modal.Volume.from_name("uxo-runs", create_if_missing=True)

# 2015 holds about 60% of all boxes, so it must be in train. 2018 is the only other year with
# both classes, which makes it the test set. 2021 has no NOMBO boxes; 2017 has only 2.
SPLITS = {"train": ["2010", "2015", "2021"], "val": ["2017"], "test": ["2018"]}
NAMES = ["mine_like", "other"]


def count_labels(label_dir: pathlib.Path) -> dict:
    boxes = [0, 0]
    empty = images = 0
    for lab in label_dir.glob("*.txt"):
        images += 1
        rows = [line.split() for line in lab.read_text().splitlines() if line.strip()]
        empty += not rows
        for row in rows:
            boxes[int(row[0])] += 1
    return {"images": images, "empty": empty, "mine_like_boxes": boxes[0], "other_boxes": boxes[1]}


@app.function(gpu="L40S", timeout=3600, cpu=8, memory=32768, volumes={"/data": data_vol, "/runs": runs_vol})
def train(epochs: int = 100, name: str = "yolo_run1", imgsz: int = 640, batch: int = 32) -> dict:
    from ultralytics import YOLO

    ds = pathlib.Path("/tmp/ds")
    stats = {}
    for split, years in SPLITS.items():
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for year in years:
            for img in pathlib.Path(f"/data/milco/{year}/{year}").glob("*.jpg"):
                shutil.copy(img, ds / "images" / split / img.name)
                shutil.copy(img.with_suffix(".txt"), ds / "labels" / split / (img.stem + ".txt"))
        stats[split] = count_labels(ds / "labels" / split)
    print("DATA", json.dumps(stats), flush=True)
    (ds / "data.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {NAMES}\n"
    )

    model = YOLO("yolov8n.pt")
    model.train(data=str(ds / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, patience=30, workers=8,
                project="/tmp/out", name=name, exist_ok=True, verbose=False, plots=True)
    best = YOLO(f"/tmp/out/{name}/weights/best.pt")
    m = best.val(data=str(ds / "data.yaml"), split="test", imgsz=imgsz, project="/tmp/out", name=f"{name}_test", plots=True)
    per_class = {NAMES[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    result = {"data": stats, "test_map50": float(m.box.map50), "test_map50_95": float(m.box.map),
              "test_ap50_per_class": per_class, "test_precision": float(m.box.mp), "test_recall": float(m.box.mr)}
    print("RESULT", json.dumps(result), flush=True)

    out = pathlib.Path("/runs") / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(f"/tmp/out/{name}", out)
    shutil.copytree(f"/tmp/out/{name}_test", out / "test")
    (out / "results.json").write_text(json.dumps(result, indent=1))
    runs_vol.commit()
    return result


@app.local_entrypoint()
def main(name: str = "yolo_run1", imgsz: int = 640, batch: int = 32) -> None:
    print(json.dumps(train.remote(name=name, imgsz=imgsz, batch=batch), indent=1))
