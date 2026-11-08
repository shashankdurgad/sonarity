"""YOLO as a warm Modal service, for code that can't import torch (Isaac 6.0.1's Python has none).

Deploy once (and again after replacing weights):  python -P -m modal deploy perception/detector_service.py
A warm container keeps serving the OLD code/weights after a redeploy: also run
    modal container list   then   modal container stop -y <uxo-detector container id>
Call from anywhere with Modal credentials, e.g. the mission loop inside Isaac:
    from perception.detector_service import remote_detect_frame
    dets = remote_detect_frame(sonar, pose)        # same output as perception.detect.detect_frame
Then feed dets to perception.detect.ContactTracker (pure Python, runs in the mission container).
Weights: uxo-runs Volume, yolo_run2/weights/best.pt (override with UXO_WEIGHTS in the service).
"""
import io
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
APP_NAME, CLS_NAME = "uxo-detector", "Detector"
WEIGHTS = "/runs/yolo_run2/weights/best.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.3.40", "numpy<2")
    .add_local_dir(REPO / "perception", "/root/perception")
    .add_local_dir(REPO / "sim", "/root/sim")
)
app = modal.App(APP_NAME, image=image)


@app.cls(gpu="L40S", volumes={"/runs": modal.Volume.from_name("uxo-runs")}, scaledown_window=900, timeout=600)
class Detector:
    @modal.enter()
    def load(self):
        import sys

        import numpy as np

        sys.path.insert(0, "/root")
        from perception.detect import detect_frame, load_model

        load_model(WEIGHTS)
        detect_frame(np.zeros((512, 512), np.uint8), {"x": 0, "y": 0, "yaw": 0}, weights=WEIGHTS)  # warm up
        self._detect = detect_frame

    @modal.method()
    def detect(self, sonar_png: bytes, pose: dict, conf: float = 0.25) -> str:
        import numpy as np
        from PIL import Image

        sonar = np.array(Image.open(io.BytesIO(sonar_png)).convert("L"))
        import json

        # JSON text: callers (the mission's bare python) have no numpy to unpickle numpy scalars
        return json.dumps(self._detect(sonar, pose, conf=conf, weights=WEIGHTS))


_handle = None


def remote_detect_frame(sonar, pose: dict, conf: float = 0.25) -> list[dict]:
    """Send one sonar frame (H x W uint8 array, or PNG bytes) to the deployed detector."""
    global _handle
    if not isinstance(sonar, (bytes, bytearray)):  # bytes path needs no numpy/PIL (bare Modal python)
        import numpy as np
        from PIL import Image

        img = np.asarray(sonar)
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        sonar = buf.getvalue()
    if _handle is None:
        _handle = modal.Cls.from_name(APP_NAME, CLS_NAME)()
    pose = {k: float(pose[k]) for k in ("x", "y", "yaw")}
    import json

    return json.loads(_handle.detect.remote(bytes(sonar), pose, conf))
