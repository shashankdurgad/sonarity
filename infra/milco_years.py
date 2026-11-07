"""Per-year label counts and labelled sample sheets for MILCO/NOMBO.

Run: OUT=/path/to/dir modal run infra/milco_years.py
"""
import collections
import io
import json
import os
import pathlib
import random

import modal

app = modal.App("uxo-milco-years", image=modal.Image.debian_slim(python_version="3.11").pip_install("pillow"))
vol = modal.Volume.from_name("uxo-data")


@app.function(volumes={"/data": vol}, timeout=300, cpu=2)
def report() -> dict:
    from PIL import Image, ImageDraw

    counts, sheets = {}, {}
    for year in ("2010", "2015", "2017", "2018", "2021"):
        c = collections.Counter()
        labelled, sizes = [], collections.Counter()
        for lab in sorted(pathlib.Path(f"/data/milco/{year}/{year}").glob("*.txt")):
            rows = [[float(x) for x in l.split()] for l in lab.read_text().splitlines() if l.strip()]
            c["images"] += 1
            c["labelled_images"] += bool(rows)
            for r in rows:
                c["milco" if int(r[0]) == 0 else "nombo"] += 1
            if rows:
                labelled.append((lab.with_suffix(".jpg"), rows))
        counts[year] = dict(c)
        random.Random(0).shuffle(labelled)
        sheet = Image.new("RGB", (1200, 600), "black")
        for i, (img, rows) in enumerate(labelled[:8]):
            im = Image.open(img).convert("RGB")
            sizes[im.size] += 1
            d = ImageDraw.Draw(im)
            W, H = im.size
            for k, x, y, w, h in [r[:5] for r in rows]:
                d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                            outline="#ff3030" if int(k) == 0 else "#30ff30", width=max(2, W // 150))
            im.thumbnail((300, 300))
            sheet.paste(im, ((i % 4) * 300, (i // 4) * 300))
        counts[year]["sample_sizes"] = {f"{a}x{b}": n for (a, b), n in sizes.items()}
        b = io.BytesIO()
        sheet.save(b, "PNG")
        sheets[year] = b.getvalue()
    return {"counts": counts, "sheets": sheets}


@app.local_entrypoint()
def main() -> None:
    r = report.remote()
    out = pathlib.Path(os.environ.get("OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    for y, data in r["sheets"].items():
        (out / f"milco_{y}.png").write_bytes(data)
    print(json.dumps(r["counts"], indent=1))
