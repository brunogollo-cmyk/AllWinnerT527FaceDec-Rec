#!/usr/bin/env python3
"""Measure real capture / detect / embed timings on this machine."""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from facegate.camera import FrameGrabber  # noqa: E402
from facegate.config import load_config  # noqa: E402
from facegate.detector import FaceDetector  # noqa: E402
from facegate.recognizer import FaceRecognizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config")
    ap.add_argument("-n", "--frames", type=int, default=100)
    ap.add_argument("--sizes", default="160,224,256,320",
                    help="comma-separated YuNet input sizes to compare")
    ap.add_argument("--offline", metavar="IMAGE",
                    help="benchmark on a still image instead of the camera")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    log = logging.getLogger("bench")

    cfg = load_config(args.config)
    recognizer = FaceRecognizer(cfg.recognizer)

    if args.offline:
        image = cv2.imread(args.offline)
        if image is None:
            log.error("Could not read %s", args.offline)
            return 1
        frames = [image] * args.frames
        log.info("Offline benchmark on %s (%dx%d)", args.offline, image.shape[1], image.shape[0])
    else:
        grabber = FrameGrabber(cfg.camera)
        log.info("Streaming %d frames from %s", args.frames, grabber.device)
        frames = []
        deadline = time.monotonic() + 30
        while len(frames) < args.frames and time.monotonic() < deadline:
            f = grabber.read()
            if f is None:
                time.sleep(0.01)
                continue
            frames.append(f)
        grabber.release()
        if not frames:
            log.error("No frames captured from %s", args.offline)
            return 1

    print()
    print(f"Captured {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    print()
    print(f"{'input':>10} {'detect ms':>12} {'detect fps':>12} {'faces/frame':>13} {'embed ms':>10}")
    print("-" * 62)

    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        cfg.detector.input_size = size
        detector = FaceDetector(cfg.detector)

        detect_ms: list[float] = []
        face_counts: list[int] = []
        embed_ms: list[float] = []

        for frame in frames:
            faces = detector.detect(frame)
            detect_ms.append(detector.last_ms)
            face_counts.append(len(faces))
            for f in faces:
                recognizer.embed(frame, f)
                embed_ms.append(recognizer.last_ms)

        avg_det = statistics.mean(detect_ms)
        avg_faces = statistics.mean(face_counts)
        embed_str = f"{statistics.mean(embed_ms):.1f}" if embed_ms else "-"
        embed_total = statistics.mean(embed_ms) * avg_faces if embed_ms else 0.0

        print(
            f"{size:>10} {avg_det:>12.1f} {1000 / avg_det:>12.1f} "
            f"{avg_faces:>13.2f} {embed_str:>10}"
        )
        budget = avg_det + embed_total
        if budget:
            print(
                f"{'':>10} {'+ embed:':>12} {budget:>11.1f} ms/frame"
                f"  -> {1000 / budget:.1f} fps ceiling"
            )
        else:
            print(f"{'':>10} no faces in frame, embedding cost not measured")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
