"""Score targets.json against world.json ground truth."""

import json
import math
import os
import sys


def _match_contacts_to_objects(contacts: list, objects: list, threshold_m: float = 5.0) -> dict:
    """Match each contact to the nearest ground-truth object within threshold."""
    matches = {}  # contact_id -> object or None
    used_objects = set()
    for c in contacts:
        cx, cy = c["xy"]
        best_obj = None
        best_dist = threshold_m
        for obj in objects:
            if obj["id"] in used_objects:
                continue
            ox, oy = obj["xyz"][0], obj["xyz"][1]
            dist = math.hypot(cx - ox, cy - oy)
            if dist < best_dist:
                best_dist = dist
                best_obj = obj
        if best_obj:
            matches[c["id"]] = best_obj
            used_objects.add(best_obj["id"])
        else:
            matches[c["id"]] = None
    return matches


UXO_CLASSES = {"aerial_bomb", "sphere_mine"}


def compute(run_dir: str) -> dict:
    """Compute metrics for a run. Returns a dict of scores."""
    with open(os.path.join(run_dir, "world.json")) as f:
        world = json.load(f)
    with open(os.path.join(run_dir, "contacts.json")) as f:
        contacts = json.load(f)
    with open(os.path.join(run_dir, "targets.json")) as f:
        targets = json.load(f)

    objects = world["objects"]
    target_map = {t["contact_id"]: t for t in targets}
    matches = _match_contacts_to_objects(contacts, objects)

    gt_uxo_count = sum(1 for o in objects if o["cls"] in UXO_CLASSES)
    gt_nonuxo_count = len(objects) - gt_uxo_count

    # Detection: how many ground-truth objects were found by any contact
    detected_ids = {m["id"] for m in matches.values() if m is not None}
    detection_rate = len(detected_ids) / len(objects) if objects else 0.0

    # Classification accuracy
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    for c in contacts:
        t = target_map.get(c["id"])
        gt_obj = matches.get(c["id"])
        if t is None or gt_obj is None:
            if t is not None and gt_obj is None:
                # False detection
                fp += (1 if t["class"] == "UXO" else 0)
            continue
        gt_is_uxo = gt_obj["cls"] in UXO_CLASSES
        pred_is_uxo = t["class"] == "UXO"
        total += 1
        if gt_is_uxo == pred_is_uxo:
            correct += 1
        if pred_is_uxo and gt_is_uxo:
            tp += 1
        elif pred_is_uxo and not gt_is_uxo:
            fp += 1
        elif not pred_is_uxo and gt_is_uxo:
            fn += 1
        else:
            tn += 1

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # Missed UXO (most important safety metric)
    detected_uxo_ids = set()
    for c in contacts:
        gt_obj = matches.get(c["id"])
        t = target_map.get(c["id"])
        if gt_obj and gt_obj["cls"] in UXO_CLASSES and t and t["class"] == "UXO":
            detected_uxo_ids.add(gt_obj["id"])
    missed_uxo = gt_uxo_count - len(detected_uxo_ids)

    results = {
        "scenario": world.get("scenario", "unknown"),
        "seed": world.get("seed", 0),
        "gt_objects": len(objects),
        "gt_uxo": gt_uxo_count,
        "gt_non_uxo": gt_nonuxo_count,
        "contacts_detected": len(contacts),
        "detection_rate": round(detection_rate, 3),
        "classification_accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "missed_uxo": missed_uxo,
    }
    return results


def run(run_dir: str) -> dict:
    results = compute(run_dir)
    out_path = os.path.join(run_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[metrics] {results}")
    print(f"[metrics] Wrote {out_path}")
    return results


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/sample"
    run(run_dir)
