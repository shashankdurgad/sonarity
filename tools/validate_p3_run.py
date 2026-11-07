"""Check P3 run-file structure. Does not judge sonar realism or label accuracy.

Usage: python validate_p3_run.py runs/p3_check --overlays p3_qa
Dependency: python -m pip install Pillow
Class IDs assumed: 0=mine_like, 1=other. Confirm with the training owner.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(root, overlays=None):
    errors = []
    root = Path(root)
    if not root.is_dir():
        return [f"Run directory does not exist: {root}"], 0
    try:
        world = json.loads((root / "world.json").read_text(encoding="utf-8"))
        if not isinstance(world, dict) or set(world) != {"seed", "scenario", "objects"}:
            raise ValueError("expected exactly seed, scenario, objects")
        if not isinstance(world["seed"], int) or isinstance(world["seed"], bool):
            raise ValueError("seed must be an integer")
        if world["scenario"] not in {"clear_sand", "half_buried", "murky"}:
            raise ValueError("unknown scenario")
        if not isinstance(world["objects"], list) or not world["objects"]:
            raise ValueError("objects must be a nonempty list")
        seen = set()
        for obj in world["objects"]:
            if not isinstance(obj, dict) or set(obj) != {"id", "cls", "xyz", "yaw", "burial"}:
                raise ValueError("object keys must be id, cls, xyz, yaw, burial")
            oid = obj["id"]
            if not isinstance(oid, str) or not re.fullmatch(r"o[1-9][0-9]*", oid) or oid in seen:
                raise ValueError("object IDs must be unique o1, o2, ...")
            seen.add(oid)
            if obj["cls"] not in {"aerial_bomb", "sphere_mine", "rock", "tyre"}:
                raise ValueError(f"invalid class for {oid}")
            if not isinstance(obj["xyz"], list) or len(obj["xyz"]) != 3 or not all(map(finite, obj["xyz"])):
                raise ValueError(f"invalid xyz for {oid}")
            if not finite(obj["yaw"]):
                raise ValueError(f"invalid yaw for {oid}")
            if not finite(obj["burial"]) or not 0 <= obj["burial"] <= 1:
                raise ValueError(f"invalid burial for {oid}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"world.json: {exc}")

    sonar = sorted((root / "sonar").glob("*.png"))
    if not sonar:
        errors.append("No sonar PNG frames found")
    stems = {p.stem for p in sonar}
    for folder, suffix in (("cam", ".png"), ("labels", ".txt")):
        actual = {p.stem for p in (root / folder).glob(f"*{suffix}")}
        if actual != stems:
            errors.append(f"{folder}: missing stems {sorted(stems-actual)[:8]}; extra stems {sorted(actual-stems)[:8]}")

    for frame in sonar:
        if not re.fullmatch(r"[0-9]{6}", frame.stem):
            errors.append(f"Invalid frame name: {frame.name}")
        for path, size, mode in ((frame, (512, 512), "L"), (root / "cam" / frame.name, (640, 360), "RGB")):
            try:
                with Image.open(path) as im:
                    im.load()
                    if im.format != "PNG" or im.size != size or im.mode != mode:
                        errors.append(f"{path}: expected PNG {size} {mode}; got {im.format} {im.size} {im.mode}")
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
        label_path = root / "labels" / f"{frame.stem}.txt"
        boxes = []
        try:
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    parts = line.split()
                    if len(parts) != 5 or parts[0] not in {"0", "1"}:
                        raise ValueError("expected class_id x_center y_center width height; class 0 or 1")
                    x, y, w, h = map(float, parts[1:])
                    if not all(map(math.isfinite, (x, y, w, h))):
                        raise ValueError("nonfinite coordinate")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("coordinates must be normalized; width and height positive")
                    if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                        raise ValueError("box extends outside image")
                    boxes.append((parts[0], x, y, w, h))
                except ValueError as exc:
                    errors.append(f"{label_path}:{line_no}: {exc}")
        except OSError as exc:
            errors.append(f"{label_path}: {exc}")
        if overlays is not None:
            try:
                out = Path(overlays)
                out.mkdir(parents=True, exist_ok=True)
                with Image.open(frame) as im:
                    canvas = im.convert("RGB")
                draw = ImageDraw.Draw(canvas)
                for cls, x, y, w, h in boxes:
                    left, top, right, bottom = ((x-w/2)*512, (y-h/2)*512, (x+w/2)*512, (y+h/2)*512)
                    draw.rectangle((left, top, right, bottom), outline="red", width=2)
                    draw.text((max(0, left), max(0, top-12)), "mine_like" if cls == "0" else "other", fill="red")
                canvas.save(out / frame.name)
            except (OSError, ValueError) as exc:
                errors.append(f"Overlay {frame.name}: {exc}")

    try:
        with (root / "poses.csv").open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["t", "x", "y", "depth", "yaw", "frame"]:
                raise ValueError("header must be t,x,y,depth,yaw,frame")
            frames = set()
            previous_t = -math.inf
            for line_no, row in enumerate(reader, 2):
                try:
                    values = [float(row[k]) for k in ("t", "x", "y", "depth", "yaw")]
                    if not all(map(math.isfinite, values)) or values[0] < 0 or values[3] < 0:
                        raise ValueError("invalid number, negative time or negative depth")
                    if values[0] < previous_t:
                        raise ValueError("timestamps must not decrease")
                    previous_t = values[0]
                    name = row["frame"]
                    if not re.fullmatch(r"sonar/[0-9]{6}\.png", name or ""):
                        raise ValueError("frame must be sonar/000000.png style")
                    if name in frames:
                        raise ValueError("duplicate frame")
                    frames.add(name)
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"poses.csv:{line_no}: {exc}")
            expected = {f"sonar/{p.name}" for p in sonar}
            if frames != expected:
                errors.append("poses.csv: frame references do not match sonar images")
    except (OSError, ValueError) as exc:
        errors.append(f"poses.csv: {exc}")
    return errors, len(sonar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--overlays", type=Path, help="Optional NEW QA output folder, outside sonar/ and cam/")
    args = parser.parse_args()
    if args.overlays is not None:
        a, b = args.run.resolve(), args.overlays.resolve()
        if b == a or a in b.parents:
            parser.error("Keep overlays outside the run folder to avoid modifying training inputs")
    errors, count = validate(args.run, args.overlays)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"FAIL: {len(errors)} issue(s)")
        sys.exit(1)
    print(f"PASS: {count} aligned sonar/camera/label frames and valid world/poses structure")
    print("Structural checks only: inspect overlays to verify sonar boxes and visibility by eye.")
