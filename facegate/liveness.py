"""Liveness (presentation-attack) detection using MiniFASNet V2.

A flat photograph or a phone screen held in front of the camera is geometrically
just a frontal face, so it will match the face database. MiniFASNet is a network
trained specifically to tell a live face from a replayed one by picking up
depth, texture and moire/screen-reflection cues that a flat image cannot
reproduce.

Input is 128x128, letterboxed and reflect-padded to match the training pipeline
exactly, normalised to [0,1] in CHW order. Output is 2 logits; class 0 is a live
face, class 1 is a spoof.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

INPUT_SIZE = 128
# The training crop takes a square region around the face that is a little
# wider than the bounding box, so the model also sees forehead and chin context.
CROP_EXPANSION = 1.7


class LivenessChecker:
    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Anti-spoof model not found at {model_path}. Run ./install.sh."
            )
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.threshold = threshold
        self.last_ms = 0.0

    def check(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> float | None:
        """Return the probability that the face is live, or None if unusable."""
        crop = self._crop(frame, box)
        if crop is None or crop.size == 0:
            return None

        blob = self._preprocess(crop)
        self.net.setInput(blob)
        start = time.perf_counter()
        out = self.net.forward()
        self.last_ms = (time.perf_counter() - start) * 1000

        logits = out.reshape(-1).astype(np.float32)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        # Class 0 is "real" in the MiniFAS label scheme.
        return float(probs[0])

    def is_live(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
        live = self.check(frame, box)
        return live is not None and live >= self.threshold

    @staticmethod
    def _crop(frame, box):
        x, y, w, h = box
        fh, fw = frame.shape[:2]
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * CROP_EXPANSION
        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(fw, cx + side / 2))
        y1 = int(min(fh, cy + side / 2))
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        old_h, old_w = crop.shape[:2]
        ratio = float(INPUT_SIZE) / max(old_h, old_w)
        scaled = (
            int(old_h * ratio),
            int(old_w * ratio),
        )
        interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, scaled, interpolation=interp)

        dh = INPUT_SIZE - resized.shape[0]
        dw = INPUT_SIZE - resized.shape[1]
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )

        if padded.shape[0] != INPUT_SIZE or padded.shape[1] != INPUT_SIZE:
            # Rounding can leave the padded image a pixel short; force exactness.
            padded = cv2.resize(padded, (INPUT_SIZE, INPUT_SIZE))

        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob.reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
