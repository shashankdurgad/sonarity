"""Compare the sonar datasets on the uxo-data Volume and draw labelled samples.

Run: OUT=/path/to/dir modal run infra/compare_datasets.py
"""
import collections
import hashlib
import io
import json
import os
import pathlib
import random
import re

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("pillow")
app = modal.App("uxo-compare-data", image=image)
vol = modal.Volume.from_name("uxo-data")
EXTS = {".jpg", ".jpeg", ".png"}
CLR = ["#ff4040", "#40ff40", "#40a0ff", "#ffff40", "#ff40ff"]


def parse(label: pathlib.Path) -> list[list[float]]:
    rows = []
    for line in label.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
    return rows


def summarise(pairs: list[tuple[pathlib.Path, pathlib.Path]]) -> dict:
    from PIL import Image

    boxes = collections.Counter()
    images_with = collections.Counter()
    sizes = collections.Counter()
    empty = missing = bad = 0
    areas = collections.defaultdict(list)
    for img, lab in pairs:
        if not lab.exists():
            missing += 1
            continue
        try:
            rows = parse(lab)
        except ValueError:
            bad += 1
            continue
        if not rows:
            empty += 1
        for c in {int(r[0]) for r in rows}:
            images_with[c] += 1
        for r in rows:
            boxes[int(r[0])] += 1
            areas[int(r[0])].append(r[3] * r[4])
    for img, _ in random.Random(0).sample(pairs, min(200, len(pairs))):
        with Image.open(img) as im:
            sizes[f"{im.width}x{im.height}"] += 1
    return {
        "images": len(pairs), "empty_labels": empty, "missing_labels": missing, "unparsable": bad,
        "boxes_by_class": dict(sorted(boxes.items())), "images_by_class": dict(sorted(images_with.items())),
        "median_box_area_frac": {k: round(sorted(v)[len(v) // 2], 5) for k, v in sorted(areas.items())},
        "common_sizes": dict(sizes.most_common(4)),
    }


def montage(pairs, want_class, title, n=12) -> bytes:
    from PIL import Image, ImageDraw

    rnd = random.Random(1)
    chosen = []
    shuffled = pairs[:]
    rnd.shuffle(shuffled)
    for img, lab in shuffled:
        if lab.exists():
            rows = parse(lab)
            if (want_class is None and not rows) or any(int(r[0]) == want_class for r in rows):
                chosen.append((img, rows))
                if len(chosen) == n:
                    break
    cell = 320
    sheet = Image.new("RGB", (cell * 4, cell * 3 + 24), "black")
    ImageDraw.Draw(sheet).text((4, 6), title, fill="white")
    for i, (img, rows) in enumerate(chosen):
        im = Image.open(img).convert("RGB")
        d = ImageDraw.Draw(im)
        for r in rows:
            c, x, y, w, h = r
            W, H = im.size
            d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                        outline=CLR[int(c) % 5], width=max(2, W // 200))
        im.thumbnail((cell, cell))
        sheet.paste(im, ((i % 4) * cell, 24 + (i // 4) * cell))
    b = io.BytesIO()
    sheet.save(b, "PNG")
    return b.getvalue()


@app.function(volumes={"/data": vol}, timeout=900, cpu=4, memory=8192)
def compare() -> dict:
    vol.reload()
    out, images = {}, {}
    root = pathlib.Path("/data")

    milco = [(p, p.with_suffix(".txt")) for p in sorted((root / "milco").rglob("*")) if p.suffix.lower() in EXTS]
    out["milco_all"] = summarise(milco)
    by_year = collections.defaultdict(list)
    for pair in milco:
        by_year[pair[0].parent.name].append(pair)
    out["milco_by_year"] = {y: summarise(v) for y, v in sorted(by_year.items())}
    hashes = collections.Counter(hashlib.md5(p.read_bytes()).hexdigest() for p, _ in milco)
    out["milco_exact_duplicate_images"] = sum(c - 1 for c in hashes.values() if c > 1)
    t = root / "milco" / "Training"
    out["milco_training_text"] = {p.name: p.read_text(errors="replace")[:400] for p in t.rglob("*.txt")}

    for split in ("train", "val", "test"):
        d = root / "drishti" / split / "images"
        if not d.exists():
            continue
        pairs = [(p, root / "drishti" / split / "labels" / (p.stem + ".txt")) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
        out[f"drishti_{split}"] = summarise(pairs)
        origins = collections.Counter(re.split(r"[_\-.]", p.stem)[0] + ("_synth" if "synth" in p.stem else "") for p, _ in pairs)
        out[f"drishti_{split}"]["filename_prefixes"] = dict(origins.most_common(12))
        real_mines = [x for x in pairs if "synth" not in x[0].stem]
        if split == "train":
            for c, name in ((4, "mine_cylinder"), (2, "shipwreck"), (1, "pipeline")):
                images[f"drishti_{name}.png"] = montage(real_mines, c, f"drishti {split} {name} (non-synth filenames)")
            mines_synth = [x for x in pairs if "synth" in x[0].stem]
            out["drishti_train_mine_boxes_in_synth_named_files"] = sum(
                sum(int(r[0]) == 4 for r in parse(l)) for _, l in mines_synth if l.exists())
    readme = root / "drishti" / "README.md"
    out["drishti_readme"] = readme.read_text(errors="replace")[:3000] if readme.exists() else None
    for c in sorted(out["milco_all"]["boxes_by_class"]):
        images[f"milco_class{c}.png"] = montage(milco, c, f"milco class {c}")
    images["milco_empty.png"] = montage(milco, None, "milco empty labels")
    return {"report": out, "images": images}


@app.local_entrypoint()
def main() -> None:
    r = compare.remote()
    d = pathlib.Path(os.environ.get("OUT", "."))
    d.mkdir(parents=True, exist_ok=True)
    for name, data in r["images"].items():
        (d / name).write_bytes(data)
    print(json.dumps(r["report"], indent=1))
