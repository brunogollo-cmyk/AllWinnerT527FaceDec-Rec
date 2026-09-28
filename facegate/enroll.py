"""Reusable face enrolment logic.

The CLI, the web API and any future caller all go through this module so the
rules for what counts as a usable enrolment image cannot drift between them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detector import FaceDetector
from .recognizer import FaceRecognizer

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB_SIZE = 160

# Reasons an image is refused, surfaced verbatim to the operator so the UI can
# explain what went wrong instead of silently dropping a photo.
REASON_UNREADABLE = "could not be read"
REASON_NO_FACE = "no face found"
REASON_MULTI_FACE = "more than one face"
REASON_ALIGN = "face alignment failed"
REASON_TOO_SMALL = "face too small"


@dataclass
class ImageResult:
    """Outcome of embedding a single image."""

    ok: bool
    source: str
    reason: str | None = None
    face_count: int = 0
    feature: np.ndarray | None = field(default=None, repr=False)
    thumbnail: bytes | None = field(default=None, repr=False)


@dataclass
class PersonSummary:
    name: str
    enrolled: int
    rejected: list[dict] = field(default_factory=list)
    similarity: float = float("nan")


@dataclass
class EnrollSummary:
    people: list[PersonSummary] = field(default_factory=list)
    total_ok: int = 0
    total_skipped: int = 0

    def as_dict(self) -> dict:
        return {
            "total_ok": self.total_ok,
            "total_skipped": self.total_skipped,
            "people": [
                {
                    "name": p.name,
                    "enrolled": p.enrolled,
                    "similarity": None
                    if p.similarity != p.similarity
                    else round(p.similarity, 4),
                    "rejected": p.rejected,
                }
                for p in self.people
            ],
        }


def make_thumbnail(image: np.ndarray, size: int = THUMB_SIZE) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


def embed_image(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    image: np.ndarray,
    source: str = "<frame>",
    allow_multi: bool = False,
    min_face_size: int = 0,
    want_thumbnail: bool = True,
) -> ImageResult:
    """Detect and embed the face in one image, following the shared rules."""
    if image is None:
        return ImageResult(False, source, REASON_UNREADABLE)

    detected = detector.detect(image)
    if not detected:
        return ImageResult(False, source, REASON_NO_FACE, 0)

    if len(detected) > 1 and not allow_multi:
        return ImageResult(
            False, source, REASON_MULTI_FACE, len(detected)
        )

    # Enrol the largest face when several are present: that is almost always
    # the subject, with other people in the background.
    face = max(detected, key=lambda f: f.box[2] * f.box[3])
    if min_face_size and min(face.box[2], face.box[3]) < min_face_size:
        return ImageResult(False, source, REASON_TOO_SMALL, len(detected))

    feature = recognizer.embed(image, face)
    if feature is None:
        return ImageResult(False, source, REASON_ALIGN, len(detected))

    thumb = make_thumbnail(image) if want_thumbnail else None
    return ImageResult(
        True, source, None, len(detected), feature=feature, thumbnail=thumb
    )


def enroll_images(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    images: list[tuple[str, np.ndarray]],
    name: str,
    allow_multi: bool = False,
    min_face_size: int = 0,
) -> PersonSummary:
    """Embed a batch of already-loaded images into one person."""
    summary = PersonSummary(name=name, enrolled=0)
    for source, image in images:
        result = embed_image(
            detector, recognizer, image, source, allow_multi, min_face_size
        )
        if result.ok:
            summary.enrolled += 1
        else:
            summary.rejected.append(
                {"source": source, "reason": result.reason}
            )
    return summary


def enroll_directory(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    faces_dir: Path,
    names: list[str] | None = None,
    allow_multi: bool = False,
    min_face_size: int = 0,
    on_embedding=None,
) -> EnrollSummary:
    """Enrol every person folder under faces_dir, or just the named ones.

    `on_embedding(name, feature)` is called for each accepted image, so a caller
    with a live FaceDB can add the embeddings as they are produced instead of
    holding the whole batch in memory first.
    """
    if not faces_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {faces_dir}")

    if names:
        people_dirs = []
        for name in names:
            person_dir = faces_dir / name
            if not person_dir.is_dir():
                raise FileNotFoundError(f"No folder for {name!r} in {faces_dir}")
            people_dirs.append(person_dir)
    else:
        people_dirs = [d for d in sorted(faces_dir.iterdir()) if d.is_dir()]

    if not people_dirs:
        raise FileNotFoundError(f"No person folders found in {faces_dir}")

    summary = EnrollSummary()
    for person_dir in people_dirs:
        images = [
            p for p in sorted(person_dir.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            log.warning("%s: no images, skipping", person_dir.name)
            continue

        person = PersonSummary(name=person_dir.name, enrolled=0)
        for path in images:
            result = embed_image(
                detector,
                recognizer,
                cv2.imread(str(path)),
                path.name,
                allow_multi,
                min_face_size,
                want_thumbnail=False,
            )
            if result.ok:
                person.enrolled += 1
                summary.total_ok += 1
                if on_embedding is not None:
                    on_embedding(person.name, result.feature)
            else:
                person.rejected.append(
                    {"source": path.name, "reason": result.reason}
                )
                summary.total_skipped += 1
                log.warning(
                    "%s/%s: %s, skipping", person_dir.name, path.name, result.reason
                )

        summary.people.append(person)
        if person.enrolled:
            log.info("%s: enrolled %d image(s)", person.name, person.enrolled)

    return summary
