"""Operator web interface: a small JSON API plus the static single-page UI.

Deliberately built on http.server with no framework. The board has no JS
toolchain, and adding FastAPI/uvicorn (or a CDN-hosted UI framework) would add
install weight and a run-time network dependency for no benefit here.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .config import WebConfig

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 8 * 1024 * 1024  # generous, but bounded


class FrameSlot:
    """Holds the most recent annotated frame for the web view to pick up."""

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._version = 0
        self._lock = threading.Lock()

    def update(self, frame: np.ndarray, quality: int) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._version += 1

    @property
    def jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._version


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    service = None  # injected by make_handler
    slot = None

    def log_message(self, fmt, *a):  # silence per-request access logs
        pass

    # ------------------------------------------------------------- utilities

    def _send(self, code: int, body: bytes, content_type: str, extra=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(
            code,
            json.dumps(payload, default=_json_default).encode(),
            "application/json",
        )

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    # ------------------------------------------------------------------ GET

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._serve_static("index.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                ctype = {
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".svg": "image/svg+xml",
                }.get(Path(name).suffix, "application/octet-stream")
                return self._serve_static(name, ctype)
            if path == "/stream":
                return self._serve_stream()
            if path == "/api/status":
                return self._json(self.service.status())
            if path == "/api/people":
                return self._json({"people": self.service.people()})
            if path == "/api/config":
                return self._json(self.service.config_dict())
            if path == "/api/events":
                qs = parse_qs(urlparse(self.path).query)
                limit = min(int(qs.get("limit", ["100"])[0]), 500)
                return self._json({"events": self.service.events(limit)})
            if path == "/api/snapshot":
                jpeg, _ = self.slot.jpeg
                if jpeg is None:
                    return self._error(503, "no frame available yet")
                return self._send(200, jpeg, "image/jpeg")
            return self._error(404, "not found")
        except Exception as exc:  # noqa: BLE001
            log.exception("GET %s failed", path)
            self._error(500, str(exc))

    def _serve_static(self, name: str, ctype: str) -> None:
        # Confine static serving to the static directory.
        target = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            return self._error(404, "not found")
        self._send(200, target.read_bytes(), ctype, {"Cache-Control": "no-cache"})

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        last_version = -1
        try:
            while True:
                jpeg, version = self.slot.jpeg
                if jpeg is None or version == last_version:
                    threading.Event().wait(0.03)
                    continue
                last_version = version
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away

    # ----------------------------------------------------------------- POST

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._error(400, str(exc))

        try:
            if path == "/api/capture":
                return self._capture()
            if path == "/api/enroll":
                return self._enroll(body)
            if path == "/api/people/remove":
                return self._remove(body)
            if path == "/api/config":
                return self._json(self.service.update_config(body))
            return self._error(404, "not found")
        except Exception as exc:  # noqa: BLE001
            log.exception("POST %s failed", path)
            self._error(500, str(exc))

    def do_DELETE(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/events":
                return self._json(self.service.clear_events())
            return self._error(404, "not found")
        except Exception as exc:  # noqa: BLE001
            log.exception("DELETE %s failed", path)
            self._error(500, str(exc))

    def _capture(self) -> None:
        result = dict(self.service.submit("capture").wait())
        # The embedding travels back to the browser so it can hold a preview
        # and post the chosen set on enrol. It is 128 floats, so this is a ~2KB
        # payload, and the whole pipeline stays stateless between calls.
        if result.get("thumbnail") is not None:
            import base64

            result["thumbnail"] = base64.b64encode(result.pop("thumbnail")).decode()
        if result.get("feature") is not None:
            result["feature"] = np.asarray(result.pop("feature")).reshape(-1).tolist()
        self._json(result)

    def _enroll(self, body: dict) -> None:
        name = body.get("name", "")
        features = body.get("features") or []
        if not isinstance(features, list):
            return self._error(400, "features must be a list")
        try:
            vectors = [np.asarray(f, dtype=np.float32).reshape(-1) for f in features]
        except (TypeError, ValueError):
            return self._error(400, "features must be numeric vectors")
        result = self.service.submit("enroll_images", name=name, features=vectors).wait()
        self._json(result)

    def _remove(self, body: dict) -> None:
        try:
            self._json(self.service.remove_person(body.get("name", "")))
        except (FileNotFoundError, ValueError) as exc:
            self._error(400, str(exc))


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def make_handler(service, slot: FrameSlot):
    return type(
        "BoundHandler",
        (Handler,),
        {"service": service, "slot": slot},
    )


def serve(cfg: WebConfig, service, slot: FrameSlot) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(service, slot))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Operator UI on http://%s:%d", cfg.host, cfg.port)
    return server
