"""YuNet face detection wrapper."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig

log = logging.getLogger(__name__)


@dataclass
class Face:
    """One detected face: bounding box, landmarks and detector confidence."""

    box: tuple[int, int, int, int]  # x, y, w, h in source-frame pixels
    landmarks: np.ndarray  # (5, 2) float32, in source-frame pixels
    score: float


class FaceDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        model = cfg.path()
        if not model.exists():
            raise FileNotFoundError(
                f"Detector model not found at {model}. Run ./install.sh to fetch models."
            )
        size = (cfg.input_size, cfg.input_size)
        self.cfg = cfg
        self._net = cv2.FaceDetectorYN.create(
            str(model), "", size, cfg.score_threshold, cfg.nms_threshold, cfg.top_k
        )
        self.last_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        side = self.cfg.input_size

        # Keep the network input fixed at cfg.input_size and resize into it.
        # Setting YuNet's input size to the frame size instead would make the
        # network run at full camera resolution: on this board 640x480 costs
        # ~88ms/frame versus ~20ms at 256x256, for no accuracy gain at the
        # detection distances an access-control camera actually sees.
        if w != side or h != side:
            resized = cv2.resize(frame, (side, side))
        else:
            resized = frame

        start = time.perf_counter()
        _, faces = self._net.detect(resized)
        self.last_ms = (time.perf_counter() - start) * 1000

        if faces is None:
            return []

        # Landmarks and boxes come back in resized-image space; scale them back
        # so callers can index the original frame.
        sx = w / float(side)
        sy = h / float(side)

        out: list[Face] = []
        for f in faces:
            x, y = int(round(f[0] * sx)), int(round(f[1] * sy))
            fw, fh = int(round(f[2] * sx)), int(round(f[3] * sy))
            landmarks = np.array(
                [
                    [f[4] * sx, f[5] * sy],
                    [f[6] * sx, f[7] * sy],
                    [f[8] * sx, f[9] * sy],
                    [f[10] * sx, f[11] * sy],
                    [f[12] * sx, f[13] * sy],
                ],
                dtype=np.float32,
            )
            out.append(Face(box=(x, y, fw, fh), landmarks=landmarks, score=float(f[-1])))
        return out
