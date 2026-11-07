"""Report what is already on the uxo-data Volume, without downloading anything.

Run: modal run infra/inspect_report.py
"""
import collections
import json
import pathlib

import modal

app = modal.App("uxo-inspect-report", image=modal.Image.debian_slim(python_version="3.11"))
vol = modal.Volume.from_name("uxo-data")


@app.function(volumes={"/data": vol}, timeout=300, cpu=2)
def report() -> dict:
    out = {}
    for name in ("milco", "drishti"):
        base = pathlib.Path("/data") / name
        by_dir = collections.defaultdict(collections.Counter)
        samples = {}
        for p in base.rglob("*"):
            if not p.is_file() or ".cache" in p.parts:
                continue
            suffix = p.suffix.lower()
            by_dir[str(p.parent.relative_to(base))][suffix] += 1
            if suffix in {".txt", ".csv", ".json", ".xml", ".yaml", ".names"} and sum(k.endswith(suffix) for k in samples) < 3:
                samples[str(p.relative_to(base))] = p.read_text(errors="replace")[:250]
        out[name] = {"dirs": {k: dict(v) for k, v in sorted(by_dir.items())[:60]}, "samples": samples}
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(report.remote(), indent=1))
