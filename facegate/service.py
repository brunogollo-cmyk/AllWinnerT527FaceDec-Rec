"""Application state shared between the detection loop and the web API.

Two problems are solved here.

1. OpenCV DNN objects are not documented as thread-safe, and the detection loop
   already saturates the CPU. So the web layer must not call the models itself.
   Instead it submits jobs to a queue that the pipeline drains between frames, and
   waits for the result. All model work stays on the one thread that owns them.

2. config.yaml documents itself with comments, so it is edited surgically by
   line rather than round-tripped through a YAML dump that would delete them.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import enroll as enroll_lib
from .config import Config, PROJECT_ROOT, describe
from .recognizer import FaceDB

log = logging.getLogger(__name__)

FACES_DIR = PROJECT_ROOT / "faces"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Writable settings, with the range each accepts. Anything not listed here is
# read-only from the API, which keeps a stray request from changing something
# structural like which model file is loaded.
BOUNDS: dict[str, tuple[float, float]] = {
    "recognizer.match_threshold": (0.0, 1.5),
    "recognizer.min_face_size": (10, 1000),
    "recognizer.reuse_window": (0.0, 10.0),
    "recognizer.reuse_min_iou": (0.0, 1.0),
    "recognizer.reuse_min_appearance": (0.0, 1.0),
    "access.liveness_threshold": (0.0, 1.0),
    "access.event_cooldown": (0.0, 3600.0),
    "detector.input_size": (64, 960),
    "detector.score_threshold": (0.0, 1.0),
    "web.quality": (30, 100),
    "camera.width": (160, 1920),
    "camera.height": (120, 1080),
    "camera.fps": (1, 60),
}

TOGGLES = {
    "access.require_liveness",
    "access.require_match",
}

# Changing any of these needs the pipeline to be rebuilt to take effect.
RESTART_REQUIRED = {"detector.input_size"}

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")


class InvalidName(ValueError):
    """Raised when a submitted person name is not safe to use on disk."""


def validate_name(name: str) -> str:
    """Reject anything that could escape the faces directory or confuse a shell."""
    if not isinstance(name, str):
        raise InvalidName("name must be a string")
    name = name.strip()
    if not name:
        raise InvalidName("name is required")
    if "\x00" in name or "/" in name or "\\" in name:
        raise InvalidName("name may not contain slashes or null bytes")
    if ".." in name:
        raise InvalidName("name may not contain '..'")
    if not NAME_RE.match(name):
        raise InvalidName(
            "name must be 1-64 characters of letters, digits, spaces, dot, dash or underscore"
        )
    return name


def safe_person_dir(name: str) -> Path:
    """Return faces/<name>, verified to stay inside faces/ after resolution."""
    clean = validate_name(name)
    base = FACES_DIR.resolve()
    target = (base / clean).resolve()
    if target != base and base not in target.parents:
        raise InvalidName("name resolves outside the faces directory")
    return base / clean


@dataclass
class Job:
    """A unit of model work handed to the pipeline thread."""

    kind: str
    payload: dict
    done: threading.Event
    result: object = None
    error: str | None = None

    def resolve(self, result: object) -> None:
        self.result = result
        self.done.set()

    def fail(self, message: str) -> None:
        self.error = message
        self.done.set()

    def wait(self, timeout: float = 15.0) -> object:
        if not self.done.wait(timeout):
            raise TimeoutError("timed out waiting for the detection loop")
        if self.error:
            raise RuntimeError(self.error)
        return self.result


class Service:
    """Owns the running pipeline and mediates every request to it."""

    def __init__(self, cfg: Config, db: FaceDB, pipeline=None) -> None:
        self.cfg = cfg
        self.db = db
        self.pipeline = pipeline
        self.started = time.time()
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ jobs

    def submit(self, kind: str, **payload) -> Job:
        job = Job(kind=kind, payload=payload, done=threading.Event())
        self._queue.put(job)
        return job

    def drain(self) -> int:
        """Run every queued job. Called by the pipeline thread only."""
        handled = 0
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return handled
            try:
                job.resolve(self._execute(job))
            except Exception as exc:  # noqa: BLE001 - report, never kill the loop
                log.exception("Job %s failed", job.kind)
                job.fail(str(exc))
            finally:
                handled += 1

    def _execute(self, job: Job):
        if job.kind == "capture":
            return self._do_capture()
        if job.kind == "enroll_images":
            return self._do_enroll_images(**job.payload)
        raise ValueError(f"unknown job kind {job.kind!r}")

    # --------------------------------------------------------------- actions

    def _do_capture(self) -> dict:
        """Grab one frame, embed the face in it, and judge liveness.

        Does not enrol: the returned embedding is handed back to the caller so
        the UI can show a preview and the operator can decide to keep it.
        """
        if self.pipeline is None:
            raise RuntimeError("pipeline is not running")
        frame = self.pipeline.grabber.read()
        if frame is None:
            raise RuntimeError("camera returned no frame")

        detector = self.pipeline.detector
        faces = detector.detect(frame)
        if not faces:
            return {
                "ok": False,
                "reason": enroll_lib.REASON_NO_FACE,
                "face_count": 0,
            }

        face = max(faces, key=_area)
        feature = self.pipeline.recognizer.embed(frame, face)
        if feature is None:
            return {
                "ok": False,
                "reason": enroll_lib.REASON_ALIGN,
                "face_count": len(faces),
            }

        live = None
        if self.pipeline.liveness is not None:
            live = self.pipeline.liveness.check(frame, face.box)

        match = self.db.match(feature, self.cfg.recognizer.match_threshold)
        return {
            "ok": True,
            "reason": None,
            "face_count": len(faces),
            "thumbnail": enroll_lib.make_thumbnail(frame),
            "feature": feature,
            "similarity": round(match.similarity, 4),
            "known": match.name,
            "liveness": None if live is None else round(live, 4),
        }

    def _do_enroll_images(self, name: str, features: list) -> dict:
        """Commit captured embeddings for a person and persist the database.

        Any embeddings the person already had are replaced, because the UI
        captures a complete fresh set each time. Otherwise repeatedly enrolling
        someone would silently accumulate near-duplicate vectors.
        """
        clean = validate_name(name)
        if not features:
            raise ValueError("no usable face captures to enrol")

        self.db.remove_person(clean)
        for feature in features:
            self.db.add(clean, np.asarray(feature, dtype=np.float32))
        self.db.save()

        similarity = self.similarity_for(clean)
        return {
            "name": clean,
            "enrolled": len(features),
            "similarity": None if similarity != similarity else round(similarity, 4),
            "low_similarity": bool(similarity == similarity and similarity < 0.7),
        }

    # ---------------------------------------------------------------- people

    def people(self) -> list[dict]:
        counts: dict[str, int] = {}
        for owner in self.db._owner:
            counts[self.db._names[owner]] = counts.get(self.db._names[owner], 0) + 1

        similarity = self.db.mean_intra_person_similarity()
        return [
            {
                "name": name,
                "embeddings": counts.get(name, 0),
                "similarity": None
                if similarity.get(name) is None
                or similarity.get(name) != similarity.get(name)
                else round(similarity[name], 4),
            }
            for name in self.db.people
        ]

    def remove_person(self, name: str) -> dict:
        clean = validate_name(name)
        if not self.db.remove_person(clean):
            raise FileNotFoundError(f"{clean!r} is not enrolled")
        self.db.save()
        person_dir = safe_person_dir(clean)
        if person_dir.is_dir():
            for path in person_dir.iterdir():
                if path.is_file():
                    path.unlink()
            person_dir.rmdir()
        return {"removed": clean, "remaining": len(self.db.people)}

    def add_feature(self, name: str, feature: np.ndarray) -> None:
        self.db.add(name, feature)

    def save_db(self) -> None:
        self.db.save()

    def similarity_for(self, name: str) -> float:
        return self.db.mean_intra_person_similarity().get(name, float("nan"))

    # ----------------------------------------------------------------- events

    def events(self, limit: int = 100) -> list[dict]:
        path = self.cfg.access.events_path()
        if not path.exists():
            return []
        out: list[dict] = []
        try:
            with path.open() as fh:
                lines = fh.readlines()
        except OSError as exc:
            log.warning("Could not read event log: %s", exc)
            return []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def clear_events(self) -> dict:
        path = self.cfg.access.events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return {"cleared": True}

    # ------------------------------------------------------------------ stats

    def status(self) -> dict:
        p = self.pipeline
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "fps": round(p.fps, 2) if p else 0.0,
            "camera": p.grabber.device if p and p.grabber.device else None,
            "camera_connected": bool(p and p.grabber.cap is not None),
            "people": len(self.db.people),
            "embeddings": len(self.db._owner),
            "detect_ms": round(p.detector.last_ms, 1) if p else 0.0,
            "embed_ms": round(p.recognizer.last_ms, 1) if p else 0.0,
            "liveness_ms": round(p.liveness.last_ms, 1)
            if p and p.liveness
            else 0.0,
            "liveness_enabled": bool(p and p.liveness),
            "reuse_enabled": bool(p and p.cache.enabled),
            "embed_reused": p.reused if p else 0,
            "embed_fresh": p.embedded if p else 0,
            "queue_depth": self._queue.qsize(),
        }

    # ----------------------------------------------------------------- config

    def config_dict(self) -> dict:
        return yaml.safe_load(describe(self.cfg)) or {}

    def update_config(self, changes: dict) -> dict:
        """Validate, persist and apply a set of dotted-key changes."""
        applied: list[str] = []
        deferred: list[str] = []
        rejected: list[dict] = []

        for key, raw in changes.items():
            if key not in BOUNDS and key not in TOGGLES:
                rejected.append({"key": key, "error": "not a writable setting"})
                continue

            if key in BOUNDS:
                lo, hi = BOUNDS[key]
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    rejected.append({"key": key, "error": "must be a number"})
                    continue
                if not (lo <= value <= hi):
                    rejected.append(
                        {"key": key, "error": f"must be between {lo} and {hi}"}
                    )
                    continue
                value = int(value) if float(value).is_integer() else value
                _set_dotted(self.cfg, key, value)
                _write_config_value(key, value)
            else:
                flag = bool(raw)
                _set_dotted(self.cfg, key, flag)
                # Write YAML booleans as true/false rather than Python's True/False.
                _write_config_value(key, "true" if flag else "false")

            (deferred if key in RESTART_REQUIRED else applied).append(key)

        if any(
            k.startswith("recognizer.reuse_") for k in changes if k not in rejected
        ):
            self._apply_cache_settings()

        return {
            "applied": applied,
            "deferred_until_restart": deferred,
            "rejected": rejected,
        }

    def _apply_cache_settings(self) -> None:
        """Push reuse tuning into the live cache so the UI takes effect at once."""
        p = self.pipeline
        if p is None:
            return
        r = self.cfg.recognizer
        p.cache.window_s = r.reuse_window
        p.cache.min_iou = r.reuse_min_iou
        p.cache.min_appearance = r.reuse_min_appearance
        # Changing a gate invalidates every track: an embedding accepted under
        # a lenient appearance gate is not automatically acceptable under a
        # stricter one, and vice versa.
        p.cache.clear()
        p.reused = 0
        p.embedded = 0


def _area(face):
    return face.box[2] * face.box[3]


def _set_dotted(obj, dotted: str, value) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _write_config_value(key: str, value) -> None:
    """Update a single key in config.yaml, preserving all comments."""
    if not CONFIG_PATH.exists():
        return
    lines = CONFIG_PATH.read_text().splitlines(keepends=True)
    section, _, leaf = key.rpartition(".")
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if stripped.startswith(f"{section}:"):
            in_section = True
            continue
        if in_section:
            # A new top-level or dotted key ends the section.
            if stripped and not stripped[0].isspace() and ":" in stripped:
                break
            match = re.match(rf"^(\s*)({re.escape(leaf)}\s*:)(.*)$", stripped)
            if match:
                indent = match.group(1)
                lines[i] = f"{indent}{match.group(2)} {value}\n"
                break
    _atomic_write(CONFIG_PATH, "".join(lines))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
