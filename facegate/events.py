"""Access event logging: JSONL on disk plus an optional HTTP webhook."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


class EventLog:
    def __init__(self, path: Path, webhook: str | None = None) -> None:
        self.path = path
        self.webhook = webhook
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _should_emit(self, key: str, cooldown: float) -> bool:
        """Rate-limit repeat events so one person standing there is logged once."""
        now = time.monotonic()
        with self._lock:
            prev = self._last.get(key)
            if prev is not None and (now - prev) < cooldown:
                return False
            self._last[key] = now
        return True

    def record(
        self,
        decision: str,
        name: str | None,
        similarity: float,
        box: tuple[int, int, int, int] | None = None,
        cooldown: float = 5.0,
        **extra,
    ) -> None:
        key = f"{decision}:{name or 'unknown'}"
        if not self._should_emit(key, cooldown):
            return

        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "name": name,
            "similarity": round(float(similarity), 4),
            "box": list(box) if box else None,
            **extra,
        }
        with self._lock:
            with self.path.open("a") as fh:
                fh.write(json.dumps(event) + "\n")
        log.info("%s", json.dumps(event))

        if self.webhook:
            threading.Thread(
                target=self._post, args=(event,), daemon=True
            ).start()

    def _post(self, event: dict) -> None:
        try:
            req = urllib.request.Request(
                self.webhook,
                data=json.dumps(event).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).close()
        except Exception as exc:  # noqa: BLE001 - a failed webhook must not stop the loop
            log.warning("Webhook delivery failed: %s", exc)
