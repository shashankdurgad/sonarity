"""Smoke test: run the pipeline on runs/sample/ and validate outputs match the contract."""

import csv
import json
import os
import subprocess
import sys

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "sample")
VALID_CLASSES = {"UXO", "non-UXO", "uncertain"}
VALID_OBJ_CLASSES = {"aerial_bomb", "sphere_mine", "rock", "tyre"}
VALID_SCENARIOS = {"clear_sand", "half_buried", "murky"}
VALID_DIRECTIONS = {"north", "east", "south", "west"}
VALID_DET_CLASSES = {"mine_like", "other"}


def test_sample_dir_exists():
    assert os.path.isdir(SAMPLE_DIR), f"{SAMPLE_DIR} missing"


def test_world_json():
    with open(os.path.join(SAMPLE_DIR, "world.json")) as f:
        world = json.load(f)
    assert "seed" in world
    assert world["scenario"] in VALID_SCENARIOS
    for obj in world["objects"]:
        assert obj["id"].startswith("o")
        assert obj["cls"] in VALID_OBJ_CLASSES
        assert len(obj["xyz"]) == 3
        assert 0.0 <= obj["burial"] <= 1.0


def test_poses_csv():
    with open(os.path.join(SAMPLE_DIR, "poses.csv")) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert set(reader.fieldnames) == {"t", "x", "y", "depth", "yaw", "frame"}
    for row in rows:
        assert row["frame"].startswith("sonar/")
        assert os.path.exists(os.path.join(SAMPLE_DIR, row["frame"]))


def test_sonar_frames():
    sonar_dir = os.path.join(SAMPLE_DIR, "sonar")
    files = sorted(f for f in os.listdir(sonar_dir) if f.endswith(".png"))
    assert len(files) > 0
    for f in files:
        assert len(f) == 10  # 000000.png
        assert f[:6].isdigit()


def test_cam_frames():
    cam_dir = os.path.join(SAMPLE_DIR, "cam")
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".png"))
    assert len(files) > 0


def test_contacts_json():
    with open(os.path.join(SAMPLE_DIR, "contacts.json")) as f:
        contacts = json.load(f)
    ids = set()
    for c in contacts:
        assert c["id"].startswith("c")
        assert c["id"] not in ids
        ids.add(c["id"])
        assert len(c["xy"]) == 2
        assert len(c["size_m"]) == 2
        assert c["cls"] in VALID_DET_CLASSES
        assert 0.0 <= c["conf"] <= 1.0
        assert os.path.exists(os.path.join(SAMPLE_DIR, c["frame"]))
        assert os.path.exists(os.path.join(SAMPLE_DIR, c["crop"]))


def test_viewpoints_json():
    with open(os.path.join(SAMPLE_DIR, "contacts.json")) as f:
        contact_ids = {c["id"] for c in json.load(f)}
    with open(os.path.join(SAMPLE_DIR, "viewpoints.json")) as f:
        viewpoints = json.load(f)
    for v in viewpoints:
        assert v["contact_id"] in contact_ids
        assert v["chosen"] in VALID_DIRECTIONS
        assert set(v["scores"].keys()) <= VALID_DIRECTIONS
        for path_list in v["imagined"].values():
            for p in path_list:
                assert os.path.exists(os.path.join(SAMPLE_DIR, p))
        assert os.path.exists(os.path.join(SAMPLE_DIR, v["inspection"]))


def test_targets_json():
    with open(os.path.join(SAMPLE_DIR, "contacts.json")) as f:
        contact_ids = {c["id"] for c in json.load(f)}
    with open(os.path.join(SAMPLE_DIR, "targets.json")) as f:
        targets = json.load(f)
    target_contact_ids = set()
    for t in targets:
        assert t["contact_id"] in contact_ids
        target_contact_ids.add(t["contact_id"])
        assert t["class"] in VALID_CLASSES
        assert 0.0 <= t["confidence"] <= 1.0
        assert isinstance(t["priority"], int)
        assert isinstance(t["standoff_m"], (int, float))
        assert isinstance(t["reason"], str) and len(t["reason"]) > 0
    # Every contact should have a target
    assert target_contact_ids == contact_ids


def test_pipeline_on_stubs():
    """Run run_demo.py on sample data with template planner."""
    result = subprocess.run(
        [sys.executable, "run_demo.py", "runs/sample", "--no-gemini"],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"
    assert "Done" in result.stdout


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
