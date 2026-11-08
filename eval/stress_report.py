"""Score finished live missions against ground truth (for comparing conditions: clear / murky / half-buried,
mid-run murk, dropped objects, uncovered bombs).

    python3 eval/stress_report.py runs/stress_murky_s4 runs/stress_clear_sand_s1 ... [--json out.json]

Per run: objects found, sonar range at first detection, closest approach to each object, keep-out
breaches, reaction to each environment change (steps until detected / held), and verdicts vs truth.
Standard library only.
"""
import argparse
import json
import math
import os
from collections import Counter

UXO = {"aerial_bomb", "sphere_mine"}
MATCH_M = 2.5


def _load(run):
    w = json.load(open(os.path.join(run, "world.json")))
    L = [json.loads(ln) for ln in open(os.path.join(run, "mission_log.jsonl")) if ln.strip()]
    c = json.load(open(os.path.join(run, "contacts.json")))
    c = c.get("contacts", c) if isinstance(c, dict) else c
    t = json.load(open(os.path.join(run, "targets.json")))
    t = t.get("targets", t) if isinstance(t, dict) else t
    return w, L, c, t


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def score(run):
    w, L, contacts, targets = _load(run)
    objs = w["objects"]
    verdict = {x["contact_id"]: x for x in targets}

    # environment changes (first step of each caption)
    events, prev = [], ""
    for s in L:
        ev = s.get("event") or ""
        if ev and ev != prev:
            events.append({"step": s["step"], "caption": ev})
        prev = ev
    murk_steps = 0
    murky = w.get("scenario") == "murky"
    for s in L:
        ev = (s.get("event") or "").lower()
        if "visibility drops" in ev:
            murky = True
        elif "visibility" in ev:
            murky = w.get("scenario") == "murky"
        murk_steps += murky

    # when each object first showed up as a confirmed contact, and from how far
    first_seen = {}
    for s in L:
        for c in s.get("contacts") or []:
            for o in objs:
                if o["id"] not in first_seen and _d(c["xy"], o["xyz"]) < MATCH_M:
                    first_seen[o["id"]] = (s["step"], _d((s["pose"]["x"], s["pose"]["y"]), o["xyz"]))

    # objects present from the start vs revealed by an event (first step they could be seen)
    appear = {}
    bomb_events = [e for e in events if any(k in e["caption"].lower() for k in ("bomb", "object"))]
    for e in bomb_events:  # nearest not-yet-assigned object seen after the event is the revealed one
        cands = [o for o in objs if o["id"] in first_seen and first_seen[o["id"]][0] >= e["step"]
                 and o["id"] not in appear]
        if cands:
            o = min(cands, key=lambda o: first_seen[o["id"]][0])
            appear[o["id"]] = e["step"]

    per_obj = []
    for o in objs:
        poses = [s for s in L if s["step"] >= appear.get(o["id"], 0)]
        closest = min((_d((s["pose"]["x"], s["pose"]["y"]), o["xyz"]) for s in poses), default=None)
        match = min(contacts, key=lambda c: _d(c["xy"], o["xyz"]), default=None)
        if match is not None and _d(match["xy"], o["xyz"]) > MATCH_M:
            match = None
        v = verdict.get(match["id"]) if match else None
        fs = first_seen.get(o["id"])
        per_obj.append({
            "id": o["id"], "truth": o["cls"], "uxo": o["cls"] in UXO, "burial": o.get("burial", 0),
            "found": match is not None, "contact": match["id"] if match else None,
            "detector_cls": match["cls"] if match else None,
            "first_seen_step": fs[0] if fs else None, "detect_range_m": round(fs[1], 1) if fs else None,
            "revealed_at_step": appear.get(o["id"]),
            "reaction_steps": (fs[0] - appear[o["id"]]) if fs and o["id"] in appear else None,
            "closest_m": round(closest, 2) if closest is not None else None,
            "verdict": v["class"] if v else None, "object_type": v.get("object_type") if v else None,
            "source": v.get("source") if v else None,
        })

    breaches = 0
    for s in L:
        p = (s["pose"]["x"], s["pose"]["y"])
        if any(_d(p, k["xy"]) < k["r"] - 0.25 for k in s.get("keep_out") or []):
            breaches += 1
    false_contacts = [c["id"] for c in contacts if all(_d(c["xy"], o["xyz"]) > MATCH_M for o in objs)]

    def held_after(step):
        for s in L:
            if s["step"] >= step and s["state"] in ("HOLD", "REPLAN", "BACKOFF"):
                return s["step"] - step
        return None

    reactions = []
    for e in bomb_events:
        oid = next((k for k, v in appear.items() if v == e["step"]), None)
        reactions.append({"step": e["step"], "caption": e["caption"][:80], "object": oid,
                          "detected_after_steps": first_seen[oid][0] - e["step"] if oid else None,
                          "held_after_steps": held_after(e["step"])})

    uxo = [p for p in per_obj if p["uxo"]]
    return {
        "run": os.path.basename(run.rstrip("/")), "scenario": w.get("scenario"), "seed": w.get("seed"),
        "steps": len(L), "states": dict(Counter(s["state"] for s in L)), "murky_steps": murk_steps,
        "events": [e["caption"][:90] for e in events],
        "found": f'{sum(p["found"] for p in per_obj)}/{len(per_obj)}',
        "false_contacts": false_contacts,
        "min_uxo_clearance_m": min((p["closest_m"] for p in uxo if p["closest_m"] is not None), default=None),
        "min_any_clearance_m": min((p["closest_m"] for p in per_obj if p["closest_m"] is not None), default=None),
        "keepout_breach_steps": breaches,
        "uxo_cleared_as_safe": [p["id"] for p in uxo if p["verdict"] == "non-UXO"],  # the dangerous error
        "uxo_called_uxo": f'{sum(p["verdict"] == "UXO" for p in uxo)}/{len(uxo)}',
        "safe_called_uxo": [p["id"] for p in per_obj if not p["uxo"] and p["verdict"] == "UXO"],
        "uncertain": [p["id"] for p in per_obj if p["verdict"] == "uncertain"],
        "median_detect_range_m": sorted(p["detect_range_m"] for p in per_obj if p["detect_range_m"])[
            len([p for p in per_obj if p["detect_range_m"]]) // 2] if any(p["detect_range_m"] for p in per_obj) else None,
        "reactions": reactions, "objects": per_obj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = [score(r) for r in a.runs]
    hdr = ["run", "steps", "found", "false", "min UXO m", "min any m", "breach", "UXO->UXO", "UXO cleared",
           "safe->UXO", "uncertain", "det range m"]
    print(" | ".join(hdr))
    for r in res:
        print(" | ".join(str(x) for x in [r["run"], r["steps"], r["found"], len(r["false_contacts"]),
                                          r["min_uxo_clearance_m"], r["min_any_clearance_m"], r["keepout_breach_steps"],
                                          r["uxo_called_uxo"], r["uxo_cleared_as_safe"] or "-",
                                          r["safe_called_uxo"] or "-", r["uncertain"] or "-", r["median_detect_range_m"]]))
        for e in r["reactions"]:
            print(f"    event @{e['step']}: {e['caption'][:60]} -> detected +{e['detected_after_steps']} steps, "
                  f"held +{e['held_after_steps']} steps")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
