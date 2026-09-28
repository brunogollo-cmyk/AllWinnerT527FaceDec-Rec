"""Configuration loading for facegate."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reference 5-point template for a 112x112 SFace input. Order must match the
# landmark order returned by YuNet: right eye, left eye, nose, right mouth, left mouth.
SFACE_REF_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

DEFAULTS: dict[str, Any] = {
    "camera": {
        "device": None,  # None -> resolve automatically by usb_id
        "usb_id": "1b3f:2002",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
        "reconnect_attempts": 5,
        "reconnect_delay": 2.0,
    },
    "detector": {
        "model": "models/yunet.onnx",
        "input_size": 256,
        "score_threshold": 0.7,
        "nms_threshold": 0.3,
        "top_k": 5000,
    },
    "recognizer": {
        "model": "models/face_recognition_sface_2021dec.onnx",
        "antispoof_model": "models/antispoof_minifas.onnx",
        "facedb": "data/facedb.json",
        # Run the embedding on the T527 NPU when the converted graph is present.
        # Falls back to the CPU automatically if it is missing or fails to load.
        "use_npu": True,
        "npu_model": "models/npu/sface_int8.nb",
        # Where the Allwinner VIPLite runtime lives. The system copy under
        # /usr/local/lib is preferred; this is the build tree used to produce it.
        "viplite_dir": "/home/orangepi/npu_vendor/ai-sdk/viplite-tina/lib/glibc-gcc10_2_0/v1.13",
        # Cosine similarity above which a face is accepted. Measured self-similarity
        # for the same person is ~0.82-0.88 on this camera; 0.5 leaves headroom while
        # staying well above cross-person similarity. Tune against real enrolments.
        "match_threshold": 0.5,
        "min_face_size": 60,
        # Re-embedding the same face every frame is the single most expensive
        # thing this pipeline does. These govern how long an embedding may be
        # reused; see tracking.py for why the appearance gate matters.
        # 0.0 disables reuse entirely and restores the original behaviour.
        "reuse_window": 1.5,
        "reuse_min_iou": 0.5,
        "reuse_min_appearance": 0.9,
    },
    "access": {
        # Fail closed: an unknown face is denied and logged.
        "require_match": True,
        # Require the anti-spoof model to pass before a GRANT is issued.
        "require_liveness": True,
        # Probability at or above which MiniFAS calls a face live. Real faces
        # measured 0.50-0.90 on this board, so 0.5 keeps genuine users passing
        # while rejecting clearly flat/replayed faces. Raise it if you observe
        # photo attacks getting through; lower it if live faces get refused.
        "liveness_threshold": 0.5,
        # Suppress repeated GRANT/DENY events for the same person within this window.
        "event_cooldown": 5.0,
        "events": "data/events.jsonl",
        # Optional HTTP endpoint receiving a JSON POST per access event.
        "webhook": None,
    },
    "web": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        "quality": 70,
    },
    "display": {
        "show": False,
    },
}


@dataclass
class CameraConfig:
    device: str | None = None
    usb_id: str = "1b3f:2002"
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0


@dataclass
class DetectorConfig:
    model: str = "models/yunet.onnx"
    input_size: int = 256
    score_threshold: float = 0.7
    nms_threshold: float = 0.3
    top_k: int = 5000

    def path(self) -> Path:
        return _resolve(self.model)


@dataclass
class RecognizerConfig:
    model: str = "models/face_recognition_sface_2021dec.onnx"
    antispoof_model: str = "models/antispoof_minifas.onnx"
    facedb: str = "data/facedb.json"
    use_npu: bool = True
    npu_model: str = "models/npu/sface_int8.nb"
    viplite_dir: str = (
        "/home/orangepi/npu_vendor/ai-sdk/viplite-tina/lib/glibc-gcc10_2_0/v1.13"
    )
    match_threshold: float = 0.5
    min_face_size: int = 60
    reuse_window: float = 1.5
    reuse_min_iou: float = 0.5
    reuse_min_appearance: float = 0.9

    def model_path(self) -> Path:
        return _resolve(self.model)

    def npu_model_path(self) -> Path:
        return _resolve(self.npu_model)

    def viplite_path(self) -> Path:
        return Path(self.viplite_dir)

    def antispoof_path(self) -> Path:
        return _resolve(self.antispoof_model)

    def facedb_path(self) -> Path:
        return _resolve(self.facedb)


@dataclass
class AccessConfig:
    require_match: bool = True
    require_liveness: bool = True
    liveness_threshold: float = 0.5
    event_cooldown: float = 5.0
    events: str = "data/events.jsonl"
    webhook: str | None = None

    def events_path(self) -> Path:
        return _resolve(self.events)


@dataclass
class WebConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    quality: int = 70


@dataclass
class DisplayConfig:
    show: bool = False


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    access: AccessConfig = field(default_factory=AccessConfig)
    web: WebConfig = field(default_factory=WebConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @property
    def webhook(self) -> str | None:
        return self.access.webhook


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from YAML, falling back to defaults for anything omitted."""
    data = dict(DEFAULTS)
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open() as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{cfg_path} must contain a YAML mapping")
        data = _merge(data, loaded)

    return Config(
        camera=CameraConfig(**data["camera"]),
        detector=DetectorConfig(**data["detector"]),
        recognizer=RecognizerConfig(**data["recognizer"]),
        access=AccessConfig(**data["access"]),
        web=WebConfig(**data["web"]),
        display=DisplayConfig(**data["display"]),
    )


def describe(cfg: Config) -> str:
    return yaml.safe_dump(asdict(cfg), sort_keys=False, default_flow_style=False)
