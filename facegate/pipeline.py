"""Per-frame orchestration: detect, embed, match, annotate, log."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import FrameGrabber
from .config import Config
from .detector import Face, FaceDetector
from .events import EventLog
from .liveness import LivenessChecker
from .recognizer import FaceDB, FaceRecognizer, Match
from .tracking import FeatureCache

log = logging.getLogger(__name__)

GREEN = (60, 200, 60)
RED = (60, 60, 235)
GREY = (170, 170, 170)
AMBER = (0, 180, 235)


@dataclass
class FrameResult:
    image: np.ndarray
    faces: int
    detect_ms: float
    embed_ms: float
    liveness_ms: float
    total_ms: float
    names: list[str | None]


class Pipeline:
    def __init__(self, cfg: Config, db: FaceDB) -> None:
        self.cfg = cfg
        self.db = db
        self.grabber = FrameGrabber(cfg.camera)
        self.detector = FaceDetector(cfg.detector)
        self.recognizer = FaceRecognizer(cfg.recognizer)
        self.liveness = (
            LivenessChecker(
                cfg.recognizer.antispoof_path(), cfg.access.liveness_threshold
            )
            if cfg.access.require_liveness
            else None
        )
        if self.liveness is None:
            log.warning(
                "Liveness checking is DISABLED - a photograph held to the "
                "camera will be treated as a live face."
            )
        self.events = EventLog(
            cfg.access.events_path(), cfg.webhook or None
        )
        self.cache = FeatureCache(
            cfg.recognizer.reuse_window,
            cfg.recognizer.reuse_min_iou,
            cfg.recognizer.reuse_min_appearance,
        )
        if not self.cache.enabled:
            log.info("Embedding reuse DISABLED - every face is re-embedded")
        else:
            log.info(
                "Embedding reuse enabled: window=%.1fs iou>=%.2f appearance>=%.2f",
                cfg.recognizer.reuse_window,
                cfg.recognizer.reuse_min_iou,
                cfg.recognizer.reuse_min_appearance,
            )
        self.reused = 0
        self.embedded = 0
        self._fps = 0.0
        self._last_frame_time = time.monotonic()
        # Set by run.py once the Service exists, so the detection loop can drain
        # web requests on this thread.
        self.service = None

    @property
    def fps(self) -> float:
        return self._fps

    def process(self, frame: np.ndarray) -> FrameResult:
        start = time.perf_counter()

        faces = self.detector.detect(frame)
        detect_ms = self.detector.last_ms

        embed_ms = 0.0
        liveness_ms = 0.0
        names: list[str | None] = []
        for face in faces:
            match = self._identify(frame, face)
            embed_ms += self.recognizer.last_ms

            live_prob: float | None = None
            # Only spend the liveness pass on a face we would actually let in.
            # Checking every stray detection wastes ~16ms for no benefit.
            if self.liveness is not None and (
                match.is_known or not self.cfg.access.require_match
            ):
                live_prob = self.liveness.check(frame, face.box)
                if live_prob is not None:
                    liveness_ms += self.liveness.last_ms

            names.append(match.name)
            self._annotate(frame, face, match, live_prob)
            self._maybe_log(face, match, live_prob)

        total_ms = (time.perf_counter() - start) * 1000
        self.cache.prune()
        self._draw_hud(frame, len(faces), detect_ms, embed_ms, liveness_ms, total_ms)
        return FrameResult(
            frame, len(faces), detect_ms, embed_ms, liveness_ms, total_ms, names
        )

    def _identify(self, frame: np.ndarray, face: Face) -> Match:
        x, y, w, h = face.box
        # Quality gate: too small a face yields an unreliable embedding, and on a
        # cheap webcam that is the main way a bogus match slips through.
        if min(w, h) < self.cfg.recognizer.min_face_size:
            return Match(name=None, similarity=0.0)

        aligned = self.recognizer.align(frame, face)
        if aligned is None:
            return Match(name=None, similarity=0.0)

        feature = self.cache.lookup(face.box, aligned)
        if feature is not None:
            self.reused += 1
            # Reuse skips the SFace pass, so report the cost of that skipped
            # pass rather than the stale timing from the previous embed.
            self.recognizer.last_ms = 0.0
        else:
            feature = self.recognizer.embed_aligned(aligned)
            if feature is None:
                return Match(name=None, similarity=0.0)
            self.embedded += 1
            self.cache.store(face.box, aligned, feature)

        return self.db.match(feature, self.cfg.recognizer.match_threshold)

    def _maybe_log(
        self, face: Face, match: Match, live_prob: float | None = None
    ) -> None:
        recognised = match.is_known or not self.cfg.access.require_match
        passed_liveness = (
            live_prob is None
            or not self.cfg.access.require_liveness
            or live_prob >= self.cfg.access.liveness_threshold
        )

        if not recognised:
            self.events.record(
                "DENY",
                None,
                match.similarity,
                face.box,
                self.cfg.access.event_cooldown,
                detect_score=round(face.score, 3),
            )
        elif not passed_liveness:
            self.events.record(
                "DENY_SPOOF",
                match.name,
                match.similarity,
                face.box,
                self.cfg.access.event_cooldown,
                detect_score=round(face.score, 3),
                liveness=round(live_prob or 0.0, 4),
            )
        else:
            self.events.record(
                "GRANT",
                match.name,
                match.similarity,
                face.box,
                self.cfg.access.event_cooldown,
                detect_score=round(face.score, 3),
                **(
                    {"liveness": round(live_prob, 4)}
                    if live_prob is not None
                    else {}
                ),
            )

    def _annotate(
        self,
        frame: np.ndarray,
        face: Face,
        match: Match,
        live_prob: float | None = None,
    ) -> None:
        x, y, w, h = face.box
        too_small = min(w, h) < self.cfg.recognizer.min_face_size
        spoofed = (
            live_prob is not None
            and self.cfg.access.require_liveness
            and live_prob < self.cfg.access.liveness_threshold
        )

        if too_small:
            color = GREY
        elif spoofed:
            color = AMBER
        elif match.is_known:
            color = GREEN
        else:
            color = RED

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        for lx, ly in face.landmarks.astype(int):
            cv2.circle(frame, (lx, ly), 1, color, -1)

        if too_small:
            label = "too far"
        elif spoofed:
            label = f"PHOTO/SPOOF {live_prob:.2f}"
        elif match.is_known:
            label = f"{match.name} {match.similarity:.2f}"
            if live_prob is not None:
                label += f" live {live_prob:.2f}"
        elif not self.cfg.access.require_match:
            # The gate is disabled, so this face would be let in despite being
            # unknown. Say so rather than leaving a bare "unknown".
            label = f"unknown {match.similarity:.2f} (open)"
        else:
            label = f"unknown {match.similarity:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 6 if y - th - 6 > 0 else y + th + 6
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 4), color, -1)
        cv2.putText(
            frame, label, (x + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    def _draw_hud(
        self, frame, faces, detect_ms, embed_ms, liveness_ms, total_ms
    ) -> None:
        now = time.monotonic()
        delta = now - self._last_frame_time
        if delta > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1 / delta) if self._fps else 1 / delta
        self._last_frame_time = now

        text = (
            f"{self._fps:4.1f} fps | {faces} face(s) | det {detect_ms:5.1f}ms | "
            f"emb {embed_ms:5.1f}ms | live {liveness_ms:5.1f}ms"
        )
        if self.cache.enabled:
            # Surfaced because a wrong hit rate here means either a wasted
            # cache (no speedup) or a too-lenient gate (wrong identity reused).
            text += f" | reuse {self.reused}/{self.reused + self.embedded}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (6, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def run(self, on_frame=None) -> None:
        log.info("Pipeline started, press Ctrl-C to stop")
        try:
            while True:
                # Run any queued web requests on this thread, so all model work
                # stays on the single thread that owns the DNN objects.
                if self.service is not None:
                    self.service.drain()

                frame = self.grabber.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                result = self.process(frame)
                if on_frame is not None:
                    on_frame(result)
        except KeyboardInterrupt:
            log.info("Pipeline stopped")
        finally:
            self.grabber.release()
