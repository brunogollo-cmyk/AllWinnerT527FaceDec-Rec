"""Reuse of face embeddings between frames.

SFace is the dominant cost in the pipeline: embedding a single face measures
~68 ms on this board against ~19 ms for detection, so a face that stays in
front of the camera for a few seconds is re-embedded on every single frame
even though nothing about it has changed.

This module caches the most recent embedding per tracked face and reuses it
while the face is judged to be the same one. The dangerous failure mode is
reusing an embedding across a *different* person who happens to occupy a
similar bounding box, which would grant them someone else's identity. Two
gates guard against that, and both must pass:

  1. box_iou - the new box must overlap the cached one closely.
  2. appearance - a 16x16 greyscale descriptor of the aligned crop must have
     high cosine similarity to the cached one. This is the gate that matters:
     a different person in the same spot looks nothing like the person who
     was there, even when the bounding box is nearly identical.

The appearance descriptor costs well under a millisecond, against 68 ms for
the embedding it protects, so checking it is close to free.

The match against the face database is NOT cached. It runs on whatever
embedding is in hand every frame, so enrolling a new person or changing
match_threshold takes effect immediately even with reuse enabled.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 16x16 greyscale = 256 values. Big enough to tell faces apart, small enough
# to be essentially free to compute.
DESCRIPTOR_SIZE = 16


def appearance_descriptor(aligned: np.ndarray | None) -> np.ndarray | None:
    """Mean-centred, L2-normalised 16x16 greyscale thumbnail of an aligned crop.

    Mean-centring removes overall brightness, so a face that stays put while
    the auto-exposure of the camera drifts still compares as the same person.
    """
    if aligned is None or aligned.size == 0:
        return None
    if aligned.ndim == 3:
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    else:
        gray = aligned
    small = cv2.resize(
        gray, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), interpolation=cv2.INTER_AREA
    )
    vec = small.astype(np.float32).reshape(-1)
    vec = vec - vec.mean()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    box: tuple[int, int, int, int]
    embedding: np.ndarray
    descriptor: np.ndarray | None
    stamp: float


class FeatureCache:
    """Per-face embedding cache keyed on box overlap plus appearance.

    `window_s` is the maximum age of a track that may be reused. It is a
    correctness knob as much as a performance one: it bounds how long a
    stale identity can survive, so a person who walks away and is replaced
    by another within the same bounding box is re-embedded no matter how
    similar the two look in silhouette.
    """

    def __init__(
        self, window_s: float, min_iou: float, min_appearance: float
    ) -> None:
        self.window_s = float(window_s)
        self.min_iou = float(min_iou)
        self.min_appearance = float(min_appearance)
        self._tracks: list[_Track] = []
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.window_s > 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def lookup(
        self, box: tuple[int, int, int, int], aligned: np.ndarray | None
    ) -> np.ndarray | None:
        """Return a reusable embedding for this face, or None to recompute."""
        if not self.enabled:
            self.misses += 1
            return None

        now = time.monotonic()
        best: _Track | None = None
        best_iou = 0.0
        for track in self._tracks:
            if now - track.stamp > self.window_s:
                continue
            overlap = box_iou(box, track.box)
            if overlap > best_iou:
                best_iou, best = overlap, track

        if best is None or best_iou < self.min_iou:
            self.misses += 1
            return None

        descriptor = appearance_descriptor(aligned)
        if descriptor is None or best.descriptor is None:
            self.misses += 1
            return None

        # The gate that actually protects against handing one person's
        # identity to another person standing in the same place.
        if float(descriptor @ best.descriptor) < self.min_appearance:
            self.misses += 1
            return None

        best.stamp = now
        self.hits += 1
        return best.embedding

    def store(
        self,
        box: tuple[int, int, int, int],
        aligned: np.ndarray | None,
        embedding: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        # Replace any track that overlaps this one; a fresh embedding here
        # supersedes whatever identity was cached for that spot.
        self._tracks = [
            t for t in self._tracks if box_iou(box, t.box) < 0.5
        ]
        self._tracks.append(
            _Track(
                tuple(box),
                embedding,
                appearance_descriptor(aligned),
                time.monotonic(),
            )
        )

    def prune(self) -> None:
        if not self._tracks:
            return
        now = time.monotonic()
        self._tracks = [t for t in self._tracks if now - t.stamp <= self.window_s]

    def clear(self) -> None:
        self._tracks.clear()
        self.hits = 0
        self.misses = 0
