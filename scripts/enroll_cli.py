#!/usr/bin/env python3
"""Build the face database from images under faces/<person_name>/*.jpg.

All of the actual enrolment logic lives in facegate/enroll.py so the CLI and the
web interface cannot drift apart; this file is only argument handling and output.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facegate import enroll as enroll_lib  # noqa: E402
from facegate.config import load_config, PROJECT_ROOT  # noqa: E402
from facegate.detector import FaceDetector  # noqa: E402
from facegate.recognizer import FaceDB, FaceRecognizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", help="path to config.yaml")
    ap.add_argument("-d", "--dir", default=str(PROJECT_ROOT / "faces"),
                    help="directory of per-person subfolders")
    ap.add_argument("--allow-multi", action="store_true",
                    help="enroll the largest face when an image contains several")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    log = logging.getLogger("enroll")

    cfg = load_config(args.config)
    faces_dir = Path(args.dir)

    detector = FaceDetector(cfg.detector)
    recognizer = FaceRecognizer(cfg.recognizer)
    db = FaceDB(cfg.recognizer.facedb_path())

    try:
        summary = enroll_lib.enroll_directory(
            detector,
            recognizer,
            faces_dir,
            allow_multi=args.allow_multi,
            on_embedding=lambda name, feature: db.add(name, feature),
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        if not faces_dir.is_dir():
            log.error("Create it and add one subfolder per person, e.g. faces/alice/photo.jpg")
        else:
            log.error("Expected layout: faces/<name>/<images>.jpg")
        return 1

    if not db.people:
        log.error("Nothing enrolled - no usable faces found.")
        return 1

    db.save()

    print()
    print(f"Enrolled {len(db.people)} people from {summary.total_ok} images "
          f"({summary.total_skipped} skipped).")
    print()
    print("Within-person similarity (higher is more consistent):")
    for name, value in sorted(db.mean_intra_person_similarity().items()):
        if value != value:  # NaN
            print(f"  {name:<20} only 1 image - add more for reliable matching")
        else:
            note = ""
            if value < 0.7:
                note = "  <-- LOW: add more, or varied photos of this person"
            print(f"  {name:<20} {value:.3f}{note}")
    print()
    print(f"Match threshold is {cfg.recognizer.match_threshold}. "
          "If known people get denied, lower it; if strangers get in, raise it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
