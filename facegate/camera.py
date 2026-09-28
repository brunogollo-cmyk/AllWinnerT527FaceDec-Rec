"""V4L2 frame capture with USB-ID based device resolution and auto-reconnect."""
from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path

import cv2

from .config import CameraConfig

log = logging.getLogger(__name__)

# V4L2 capture devices expose a 'index' file and the parent directory name carries
# the device identity. The stable, reboot-proof way to map a node back to a camera is
# to walk /sys/class/video4linux and read the usb vid/pid of the parent USB device.
SYS_VIDEO = Path("/sys/class/video4linux")


def _usb_ids_for(device: str) -> set[str]:
    """Return the set of 'vid:pid' strings reachable from a /dev/videoN node."""
    found: set[str] = set()
    real = os.path.realpath(device)
    name = os.path.basename(real)
    link = SYS_VIDEO / name
    if not link.exists():
        return found

    # Walk up the sysfs chain until the USB device node is reached.
    for parent in link.resolve().parents:
        for vid in parent.glob("idVendor"):
            pid = parent / "idProduct"
            if pid.exists():
                found.add(f"{vid.read_text().strip()}:{pid.read_text().strip()}")
    return found


def find_device(usb_id: str) -> str | None:
    """Locate the first /dev/videoN node whose USB vid:pid matches usb_id."""
    target = usb_id.lower()
    for node in sorted(glob.glob("/dev/video*")):
        if not node[len("/dev/video") :].isdigit():
            continue
        if target in _usb_ids_for(node):
            return node
    return None


class FrameGrabber:
    """Opens a camera at a fixed format and yields frames, reconnecting on failure."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.device: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self._failures = 0
        self._open()

    def _open(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        device = self.cfg.device
        if device is None:
            device = find_device(self.cfg.usb_id)
            if device is None:
                raise RuntimeError(
                    f"No camera matching usb id {self.cfg.usb_id!r}. "
                    f"Plug it in, or set camera.device in config.yaml."
                )
        self.device = device

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera device {device}")

        # Order matters: fourcc must be set before the size is requested, otherwise
        # the driver may hand back a format the host cannot decode cheaply.
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        actual = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.cap = cap
        self._failures = 0
        log.info(
            "Camera open: %s %dx%d@%d %s (negotiated %dx%d)",
            device,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
            self.cfg.fourcc,
            actual[0],
            actual[1],
        )
        if actual != (self.cfg.width, self.cfg.height):
            log.warning(
                "Camera did not accept %dx%d, streaming %dx%d instead. "
                "Check supported formats with: v4l2-ctl -d %s --list-formats-ext",
                self.cfg.width,
                self.cfg.height,
                actual[0],
                actual[1],
                device,
            )

    def read(self):
        """Return the next frame, or None if the device has failed."""
        if self.cap is None:
            return None

        # Drop any frames the driver has queued so we always act on the newest
        # image; otherwise the reported bounding boxes lag behind reality.
        for _ in range(2):
            if not self.cap.grab():
                return self._handle_failure()
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return self._handle_failure()
        return frame

    def _handle_failure(self):
        self._failures += 1
        if self._failures < self.cfg.reconnect_attempts:
            return None
        log.warning("Camera %s lost, attempting reconnect", self.device)
        self.cap = None
        time.sleep(self.cfg.reconnect_delay)
        try:
            # Re-resolve the node: it commonly changes when the device resets.
            self.cfg.device = None
            self._open()
        except RuntimeError as exc:
            log.error("Reconnect failed: %s", exc)
        return None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
