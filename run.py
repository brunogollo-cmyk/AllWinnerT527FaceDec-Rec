#!/usr/bin/env python3
"""Start the face detection and recognition service."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facegate.config import load_config, describe  # noqa: E402
from facegate.recognizer import FaceDB  # noqa: E402
from facegate import web  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", help="path to config.yaml")
    ap.add_argument("-d", "--device", help="override camera device, e.g. /dev/video1")
    ap.add_argument("--no-web", action="store_true", help="disable the web view")
    ap.add_argument("--show", action="store_true", help="open an OpenCV window")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--list-people", action="store_true", help="print enrolled people and exit")
    ap.add_argument("--remove", metavar="NAME",
                    help="remove a person from the face database and exit")
    ap.add_argument("--print-config", action="store_true", help="print effective config and exit")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("facegate")

    cfg = load_config(args.config)
    if args.device:
        cfg.camera.device = args.device
    if args.no_web:
        cfg.web.enabled = False
    if args.show:
        cfg.display.show = True

    if args.print_config:
        print(describe(cfg))
        return 0

    db = FaceDB(cfg.recognizer.facedb_path())
    db.load()

    if args.list_people:
        if not db.people:
            print("No faces enrolled. Add images to faces/<name>/ then run enroll.py")
            return 1
        print(f"{len(db.people)} people enrolled:")
        for name in db.people:
            print(f"  - {name}")
        return 0

    if args.remove:
        if not db.remove_person(args.remove):
            print(f"No enrolled person called {args.remove!r}.")
            print(f"Enrolled: {', '.join(db.people) or '(none)'}")
            return 1
        db.save()
        print(f"Removed {args.remove}. {len(db.people)} people remain.")
        return 0

    if not db.people:
        log.warning(
            "No faces enrolled - every face will be denied. "
            "Add photos under faces/<name>/ and run: ./enroll.sh"
        )

    from facegate.pipeline import Pipeline  # imported late: cv2 init is slow
    from facegate.service import Service

    pipeline = Pipeline(cfg, db)
    slot = None
    if cfg.web.enabled:
        service = Service(cfg, db, pipeline)
        # The detection loop drains queued web requests on its own thread, so
        # the DNN objects are only ever touched from that one thread.
        pipeline.service = service
        slot = web.FrameSlot()
        web.serve(cfg.web, service, slot)
    else:
        log.warning("Web interface disabled (--no-web)")

    import cv2

    def on_frame(result):
        if slot is not None:
            slot.update(result.image, cfg.web.quality)
        if cfg.display.show:
            cv2.imshow("facegate", result.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    try:
        pipeline.run(on_frame=on_frame)
    finally:
        if cfg.display.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
