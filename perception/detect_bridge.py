"""Let Isaac's Python (no torch, no modal) use the deployed uxo-detector service.

Outer Modal function (has modal):   url = serve()            # localhost proxy, sets UXO_DETECT_URL
Isaac process (stdlib only):        from perception.detect_bridge import bridge_detect_frame
                                    dets = bridge_detect_frame(sonar, pose)   # = detect_frame output
"""
import io
import json
import os
import threading
import urllib.error
import urllib.request

PORT = 8765


def serve(port: int = PORT) -> str:
    """Start the proxy in a daemon thread; returns its URL and exports UXO_DETECT_URL for children."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from perception.detector_service import remote_detect_frame

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                pose = json.loads(self.headers["X-Pose"])
                out = json.dumps(remote_detect_frame(body, pose)).encode()
                self.send_response(200)
            except Exception as e:  # report to the caller, keep serving
                out = json.dumps({"error": repr(e)}).encode()
                self.send_response(500)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/detect"
    os.environ["UXO_DETECT_URL"] = url
    return url


def bridge_detect_frame(sonar, pose: dict, timeout: float = 5.0) -> list[dict]:
    import numpy as np
    from PIL import Image

    img = np.asarray(sonar)
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    req = urllib.request.Request(os.environ["UXO_DETECT_URL"], data=buf.getvalue(), method="POST",
                                 headers={"X-Pose": json.dumps({k: float(pose[k]) for k in ("x", "y", "yaw")})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # surface the proxy's real error
        raise RuntimeError(f"detector proxy {e.code}: {e.read()[:300]!r}") from None
