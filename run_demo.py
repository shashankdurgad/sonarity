#!/usr/bin/env python3
"""Orchestrates the full pipeline: survey -> detect -> imagine -> plan -> compose.

Usage:
    python run_demo.py                          # run on stubs (runs/sample)
    python run_demo.py runs/001                 # run on a specific run directory
    python run_demo.py runs/001 --no-gemma     # skip Gemma, use template planner
    python run_demo.py runs/001 --skip sim      # skip sim steps (use existing files)
    python run_demo.py runs/001 --only planner  # run only the planner step
"""

import argparse
import json
import os
import sys
import time


STEPS = ["sim", "detect", "imagine", "planner", "metrics", "compose"]


def step_sim(run_dir: str, **_):
    """Run the survey simulation (Modal GPU job). Stub: checks files exist."""
    required = ["world.json", "poses.csv"]
    required_dirs = ["sonar", "cam"]
    for f in required:
        path = os.path.join(run_dir, f)
        if not os.path.exists(path):
            print(f"[sim] STUB: {path} not found. Run sim/survey.py on Modal to generate.")
            return False
    for d in required_dirs:
        path = os.path.join(run_dir, d)
        if not os.path.isdir(path) or not os.listdir(path):
            print(f"[sim] STUB: {path}/ empty. Run sim/survey.py on Modal to generate.")
            return False
    n_sonar = len([f for f in os.listdir(os.path.join(run_dir, "sonar")) if f.endswith(".png")])
    n_cam = len([f for f in os.listdir(os.path.join(run_dir, "cam")) if f.endswith(".png")])
    print(f"[sim] Found {n_sonar} sonar frames, {n_cam} camera frames")
    return True


def step_detect(run_dir: str, **_):
    """Run the YOLO detector on sonar frames."""
    contacts_path = os.path.join(run_dir, "contacts.json")
    if not os.path.exists(contacts_path):
        from perception.detect import run as detect_run
        detect_run(run_dir)
    if not os.path.exists(contacts_path):
        print(f"[detect] FAIL: {contacts_path} not produced")
        return False
    with open(contacts_path) as f:
        contacts = json.load(f)
    print(f"[detect] {len(contacts)} contacts")
    return True


def step_imagine(run_dir: str, **_):
    """Run imagination + inspection. Stub: checks viewpoints.json exists."""
    viewpoints_path = os.path.join(run_dir, "viewpoints.json")
    if not os.path.exists(viewpoints_path):
        try:
            from agent.imagine import run as imagine_run
            imagine_run(run_dir)
        except ImportError:
            print(f"[imagine] STUB: {viewpoints_path} not found and agent.imagine unavailable.")
            return False
    with open(viewpoints_path) as f:
        viewpoints = json.load(f)
    print(f"[imagine] {len(viewpoints)} viewpoints")
    return True


def step_planner(run_dir: str, use_gemma: bool = True, **_):
    """Run the Gemma planner to classify contacts."""
    from agent.planner import run as planner_run
    planner_run(run_dir, use_gemma=use_gemma)
    return True


def step_metrics(run_dir: str, **_):
    """Compute metrics against ground truth."""
    from viz.metrics import run as metrics_run
    metrics_run(run_dir)
    return True


def step_compose(run_dir: str, **_):
    """Compose the demo video. Stub: checks for compose module."""
    try:
        from viz.compose import run as compose_run
        compose_run(run_dir)
    except ImportError:
        print(f"[compose] STUB: viz.compose not available yet. Skipping video composition.")
        return True
    return True


STEP_FUNCS = {
    "sim": step_sim,
    "detect": step_detect,
    "imagine": step_imagine,
    "planner": step_planner,
    "metrics": step_metrics,
    "compose": step_compose,
}


def main():
    parser = argparse.ArgumentParser(description="Run the UXO detection pipeline")
    parser.add_argument("run_dir", nargs="?", default="runs/sample")
    parser.add_argument("--no-gemma", "--no-gemini", dest="no_gemma", action="store_true",
                        help="Use template planner instead of Gemma")
    parser.add_argument("--skip", nargs="*", default=[], help="Steps to skip")
    parser.add_argument("--only", nargs="*", default=[], help="Run only these steps")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: {args.run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    steps_to_run = args.only if args.only else [s for s in STEPS if s not in args.skip]

    print(f"=== Pipeline: {args.run_dir} ===")
    print(f"Steps: {', '.join(steps_to_run)}\n")

    for step_name in steps_to_run:
        if step_name not in STEP_FUNCS:
            print(f"Unknown step: {step_name}", file=sys.stderr)
            sys.exit(1)

        print(f"--- {step_name} ---")
        t0 = time.time()
        ok = STEP_FUNCS[step_name](args.run_dir, use_gemma=not args.no_gemma)
        elapsed = time.time() - t0
        status = "OK" if ok else "SKIPPED"
        print(f"--- {step_name}: {status} ({elapsed:.1f}s) ---\n")

    print("=== Done ===")


if __name__ == "__main__":
    main()
