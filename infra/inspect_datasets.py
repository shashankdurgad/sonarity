"""Download both sonar datasets to a Volume and report their structure and labels.

Run: modal run infra/inspect_datasets.py
"""
import collections
import hashlib
import json
import pathlib
import zipfile

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests", "huggingface_hub", "pillow")
app = modal.App("uxo-inspect-data", image=image)
vol = modal.Volume.from_name("uxo-data", create_if_missing=True)
ROOT = pathlib.Path("/data")
UA = {"User-Agent": "uxo-dataset-inspector/0.1"}


def tree(root: pathlib.Path, depth: int = 3) -> list[str]:
    rows = []
    for d in sorted(p for p in [root, *root.rglob("*")] if p.is_dir()):
        rel = d.relative_to(root)
        if len(rel.parts) > depth:
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        exts = collections.Counter(f.suffix.lower() for f in files)
        rows.append(f"{rel}/ files={len(files)} {dict(exts)}")
    return rows[:70]


@app.function(volumes={"/data": vol}, timeout=1500, cpu=4, memory=8192)
def fetch_and_report() -> dict:
    import requests
    from huggingface_hub import snapshot_download

    out = {}
    fig = ROOT / "milco"
    fig.mkdir(parents=True, exist_ok=True)
    meta = requests.get("https://api.figshare.com/v2/articles/24574879", headers=UA, timeout=60).json()
    for f in meta["files"]:
        target = fig / f["name"]
        done = fig / (f["name"] + ".done")
        if done.exists():
            continue
        with requests.get(f["download_url"], headers=UA, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(target, "wb") as w:
                for chunk in r.iter_content(1 << 20):
                    w.write(chunk)
        assert hashlib.md5(target.read_bytes()).hexdigest() == (f["computed_md5"] or f["supplied_md5"]), f["name"]
        with zipfile.ZipFile(target) as z:
            z.extractall(fig / target.stem)
        target.unlink()
        done.touch()
        vol.commit()
    out["milco_tree"] = tree(fig)

    dri = ROOT / "drishti"
    snapshot_download("rehan9599/drishti-sss", repo_type="dataset", local_dir=dri, max_workers=2)
    vol.commit()
    out["drishti_tree"] = tree(dri)
    for y in dri.rglob("*.yaml"):
        out.setdefault("drishti_yaml", {})[str(y.relative_to(dri))] = y.read_text()[:800]
    readme = dri / "README.md"
    out["drishti_readme"] = readme.read_text()[:3500] if readme.exists() else None

    samples = {}
    for name, base in (("milco", fig), ("drishti", dri)):
        seen = collections.Counter()
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".json", ".xml", ".md"} and seen[p.suffix.lower()] < 3:
                seen[p.suffix.lower()] += 1
                samples[f"{name}:{p.relative_to(base)}"] = p.read_text(errors="replace")[:300]
    out["samples"] = samples
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(fetch_and_report.remote(), indent=1))
