"""Sonar detector: batch `run(run_dir)` (PLAN §11) and live `detect_frame` + `ContactTracker` (LIVE_MISSION.md).

Weights come from `weights`, else $UXO_WEIGHTS, else perception/weights/best.pt
(download: gh release download weights-run2 -p yolo_run2_best.pt -O perception/weights/best.pt).
Detections from nearby frames are merged into one contact per object.

Live use, inside the mission loop:
    tracker = ContactTracker()
    dets = detect_frame(sonar, pose)                  # sonar: 512x512 uint8, pose: {x, y, yaw}
    for c in tracker.update(dets, frame_name, sonar):  # confirmed contacts that are new or changed
        if c["status"] in ("new", "reclassified") and c["cls"] == "mine_like": ...

Isaac 6.0.1 has no torch, so in the mission use the deployed service instead of detect_frame:
    from perception.detector_service import remote_detect_frame   # same signature and output
    tracker.save(run_dir)                             # contacts.json + crops/, same format as run()
"""
import csv
import json
import math
import os

from sim.sonar_geometry import DEFAULT, SIZE, pixel_to_world

NAMES = ["mine_like", "other"]
MERGE_RADIUS_M = 2.5
CROP_MARGIN_PX = 12
_MODELS: dict = {}


def load_model(weights: str | None = None):
    """Load YOLO once per process and reuse it."""
    from ultralytics import YOLO

    weights = weights or os.environ.get("UXO_WEIGHTS") or os.path.join(os.path.dirname(__file__), "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"detector weights not found: {weights} (set UXO_WEIGHTS)")
    if weights not in _MODELS:
        _MODELS[weights] = YOLO(weights)
    return _MODELS[weights]


def _to_dets(result, pose: dict, frame: str | None = None) -> list[dict]:
    """YOLO boxes in one sonar frame -> detections in world coordinates."""
    dets = []
    for box in result.boxes:
        x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
        w, h = x1 - x0, y1 - y0
        dets.append({
            "xy": [round(float(v), 3) for v in pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2, pose)],
            # Rough: long axis across the image means the object lies across the robot's heading.
            "yaw": round(pose["yaw"] if h >= w else pose["yaw"] + math.pi / 2, 3),
            "size_m": [round(max(w, h) * DEFAULT.m_per_px, 2), round(min(w, h) * DEFAULT.m_per_px, 2)],
            "cls": NAMES[int(box.cls)], "conf": round(float(box.conf), 3),
            "frame": frame, "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        })
    return dets


def detect_frame(sonar, pose: dict, conf: float = 0.25, weights: str | None = None, device=None) -> list[dict]:
    """Detect objects in one sonar image (H x W uint8, or H x W x 3) taken at `pose` {x, y, yaw}.

    Returns [{"xy", "size_m", "cls", "conf", "box", "yaw"}], xy in world metres.
    """
    import numpy as np

    img = np.asarray(sonar)
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)  # training PNGs were read as 3 equal channels
    pose = {k: float(pose[k]) for k in ("x", "y", "yaw")}
    result = load_model(weights).predict(img, conf=conf, verbose=False, device=device, imgsz=640)[0]
    dets = _to_dets(result, pose)
    for d in dets:
        d.pop("frame")
    return dets


def merge_detections(detections: list[dict], radius: float = MERGE_RADIUS_M, min_hits: int = 2) -> list[dict]:
    """Group detections within `radius` metres; keep each group's best frame and its summed-confidence class."""
    groups: list[list[dict]] = []
    for det in sorted(detections, key=lambda d: -d["conf"]):
        for group in groups:
            if math.dist(det["xy"], group[0]["xy"]) <= radius:
                group.append(det)
                break
        else:
            groups.append([det])
    contacts = []
    for group in groups:
        if len(group) < min_hits:
            continue
        votes = {name: sum(d["conf"] for d in group if d["cls"] == name) for name in NAMES}
        cls = max(votes, key=votes.get)
        best = next(d for d in group if d["cls"] == cls)
        n = len(group)
        contacts.append({**best, "cls": cls, "hits": n,
                         "xy": [round(float(sum(d["xy"][i] for d in group) / n), 2) for i in range(2)]})
    return sorted(contacts, key=lambda c: -c["conf"])


def run(run_dir: str, weights: str | None = None, conf: float = 0.25, device=None, min_hits: int = 2) -> list[dict]:
    from PIL import Image

    model = load_model(weights)
    with open(os.path.join(run_dir, "poses.csv")) as f:
        poses = list(csv.DictReader(f))

    detections = []
    for row in poses:
        pose = {k: float(row[k]) for k in ("x", "y", "yaw")}
        path = os.path.join(run_dir, row["frame"])
        result = model.predict(path, conf=conf, verbose=False, device=device, imgsz=640)[0]
        detections += _to_dets(result, pose, row["frame"])

    crops_dir = os.path.join(run_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    contacts = []
    for i, c in enumerate(merge_detections(detections, min_hits=min_hits), start=1):
        x0, y0, x1, y1 = c["box"]
        m = CROP_MARGIN_PX
        area = (max(int(x0) - m, 0), max(int(y0) - m, 0), min(int(x1) + m, SIZE), min(int(y1) + m, SIZE))
        Image.open(os.path.join(run_dir, c["frame"])).crop(area).save(os.path.join(crops_dir, f"c{i}.png"))
        contacts.append({"id": f"c{i}", "xy": c["xy"], "yaw": c["yaw"], "size_m": c["size_m"], "cls": c["cls"],
                         "conf": c["conf"], "frame": c["frame"], "crop": f"crops/c{i}.png"})
    with open(os.path.join(run_dir, "contacts.json"), "w") as f:
        json.dump(contacts, f, indent=2)
    print(f"[detect] {len(detections)} detections -> {len(contacts)} contacts", flush=True)
    return contacts


class ContactTracker:
    """Incremental merge_detections: builds contacts frame by frame while the robot moves.

    A detection joins the nearest track within `radius` m (else starts a new one). A track becomes a
    contact once it has `min_hits` detections; ids c1, c2, ... are given in confirmation order and
    never change. A contact's class can still flip as more evidence arrives (conf-weighted vote).
    """

    def __init__(self, radius: float = MERGE_RADIUS_M, min_hits: int = 2):
        self.radius, self.min_hits = radius, min_hits
        self.tracks: list[dict] = []   # {"dets": [...], "id": str | None, "crop": ndarray | None}
        self._next_id = 1

    @staticmethod
    def _centre(track: dict) -> list[float]:
        n = len(track["dets"])
        return [sum(d["xy"][i] for d in track["dets"]) / n for i in range(2)]

    def _contact(self, track: dict) -> dict:
        votes = {name: sum(d["conf"] for d in track["dets"] if d["cls"] == name) for name in NAMES}
        cls = max(votes, key=votes.get)
        best = max((d for d in track["dets"] if d["cls"] == cls), key=lambda d: d["conf"])
        return {"id": track["id"], "xy": [round(float(v), 2) for v in self._centre(track)], "yaw": best["yaw"],
                "size_m": best["size_m"], "cls": cls, "conf": best["conf"], "frame": best["frame"],
                "box": best["box"], "hits": len(track["dets"]), "crop": f"crops/{track['id']}.png"}

    def update(self, dets: list[dict], frame_name: str, sonar=None) -> list[dict]:
        """Add one frame's detections. Returns confirmed contacts touched by this frame, each with
        "status": "new" (just confirmed), "reclassified" (cls changed this frame) or "updated".
        Pass `sonar` to keep crops for save(). Needs no torch: run it anywhere, e.g. inside Isaac."""
        touched, newly, before = [], set(), {id(t): self._contact(t)["cls"] for t in self.tracks if t["id"]}
        for det in sorted(dets, key=lambda d: -d["conf"]):
            det = {**det, "frame": frame_name}
            near = [(math.dist(det["xy"], self._centre(t)), t) for t in self.tracks]
            near = [(dist, t) for dist, t in near if dist <= self.radius]
            if near:
                track = min(near, key=lambda p: p[0])[1]
                track["dets"].append(det)
            else:
                track = {"dets": [det], "id": None, "crop": None, "best_conf": -1.0}
                self.tracks.append(track)
            if sonar is not None and det["conf"] > track["best_conf"]:
                track["best_conf"] = det["conf"]
                track["crop"] = self._crop(sonar, det["box"])
            if track["id"] is None and len(track["dets"]) >= self.min_hits:
                track["id"] = f"c{self._next_id}"
                self._next_id += 1
                newly.add(track["id"])
            if track["id"] is not None and all(t is not track for t in touched):
                touched.append(track)
        out = []
        for t in touched:
            c = self._contact(t)
            status = "new" if t["id"] in newly else "reclassified" if before.get(id(t)) != c["cls"] else "updated"
            out.append({**c, "status": status})
        return out

    @staticmethod
    def _crop(sonar, box):
        import numpy as np

        img = np.asarray(sonar)
        x0, y0, x1, y1 = box
        m = CROP_MARGIN_PX
        return img[max(int(y0) - m, 0):min(int(y1) + m, img.shape[0]),
                   max(int(x0) - m, 0):min(int(x1) + m, img.shape[1])].copy()

    @property
    def contacts(self) -> list[dict]:
        """All confirmed contacts, contacts.json format (plus "hits")."""
        out = [self._contact(t) for t in self.tracks if t["id"] is not None]
        for c in out:
            c.pop("box")
        return sorted(out, key=lambda c: int(c["id"][1:]))

    def save(self, run_dir: str) -> list[dict]:
        """Write contacts.json and crops/<id>.png (crops only for tracks that were given a sonar image)."""
        from PIL import Image

        os.makedirs(os.path.join(run_dir, "crops"), exist_ok=True)
        for t in self.tracks:
            if t["id"] is not None and t["crop"] is not None:
                Image.fromarray(t["crop"]).save(os.path.join(run_dir, "crops", f"{t['id']}.png"))
        contacts = self.contacts
        with open(os.path.join(run_dir, "contacts.json"), "w") as f:
            json.dump(contacts, f, indent=2)
        return contacts
