# HTTP API

The web UI is a thin client over these, so everything it does you can do with
`curl`.

The server binds to `web.host:web.port` (default `0.0.0.0:8080`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | uptime, fps, per-stage timings, counts |
| GET | `/api/people` | enrolled names, embedding counts, self-similarity |
| GET | `/api/events?limit=N` | recent access events, newest first |
| DELETE | `/api/events` | clear the access log |
| POST | `/api/capture` | grab one frame; returns embedding + thumbnail + liveness |
| POST | `/api/enroll` | `{"name":..., "features":[[128 floats], ...]}` |
| POST | `/api/people/remove` | `{"name":...}` |
| GET | `/api/config` | effective configuration |
| POST | `/api/config` | update writable settings |
| GET | `/api/snapshot` | current annotated frame as a JPEG |
| GET | `/stream` | MJPEG live view |

## Examples

```bash
BASE=http://board-ip:8080

curl -s $BASE/api/status
curl -s $BASE/api/people
curl -s "$BASE/api/events?limit=10"
curl -s -o frame.jpg $BASE/api/snapshot
curl -s -X DELETE $BASE/api/events
```

### `GET /api/status`

```json
{
  "uptime_s": 44.9,
  "fps": 10.92,
  "camera": "/dev/video1",
  "camera_connected": true,
  "people": 2,
  "embeddings": 12,
  "detect_ms": 15.8,
  "embed_ms": 0.0,
  "liveness_ms": 14.7,
  "liveness_enabled": true,
  "reuse_enabled": true,
  "embed_reused": 473,
  "embed_fresh": 14,
  "queue_depth": 0
}
```

`embed_ms` is `0.0` on frames where the embedding was reused — that is by
design, not a missing measurement. Watch `embed_reused` / `embed_fresh` to see
the hit rate.

### `GET /api/people`

```json
{
  "people": [
    {"name": "Alice", "embeddings": 6, "similarity": 0.760}
  ]
}
```

`similarity` is the mean intra-person match score. Below ~0.7 means that
person's enrolled photos are too inconsistent for the threshold to be trusted
for them.

### `POST /api/capture`

Takes no body. Grabs one frame from the camera, detects, embeds, and judges
liveness. Does **not** enrol — the embedding comes back so the caller can
decide.

```json
{
  "ok": true,
  "reason": null,
  "face_count": 1,
  "thumbnail": "<base64 JPEG>",
  "feature": [0.027, -0.007, ...],
  "similarity": 0.663,
  "known": "Alice",
  "liveness": 0.83
}
```

On failure, `ok` is `false` and `reason` is one of `no face found`,
`more than one face`, `face alignment failed`, or `face too small`.

### `POST /api/enroll`

```bash
curl -s -X POST $BASE/api/enroll \
  -H 'Content-Type: application/json' \
  -d '{"name":"Alice","features":[[0.1,0.2,...],[...]]}'
```

Each feature is a 128-float embedding from `/api/capture`. **This replaces**
any existing embeddings for that person, because the UI captures a complete
fresh set each time — otherwise repeated enrolment would silently accumulate
near-duplicate vectors.

```json
{
  "name": "Alice",
  "enrolled": 6,
  "similarity": 0.760,
  "low_similarity": false
}
```

`low_similarity` is `true` when the set scores below 0.7, which means capture
more varied angles.

### `POST /api/people/remove`

```bash
curl -s -X POST $BASE/api/people/remove \
  -H 'Content-Type: application/json' -d '{"name":"Alice"}'
```

Also deletes that person's photos under `faces/`. There is no soft-delete and
no way to recover an enrolment afterwards — re-enrol if it was a mistake.

### `GET` / `POST /api/config`

```bash
curl -s $BASE/api/config

curl -s -X POST $BASE/api/config \
  -H 'Content-Type: application/json' \
  -d '{"recognizer.match_threshold":0.55,"access.require_liveness":false}'
```

Only whitelisted keys are writable, each range-checked server-side. Anything
else is rejected rather than ignored:

```json
{
  "applied": ["recognizer.match_threshold"],
  "deferred_until_restart": [],
  "rejected": [
    {"key": "camera.fps", "error": "must be between 1 and 60"}
  ]
}
```

## Webhooks

Set `access.webhook` to a URL and every access decision is POSTed there as
JSON:

```json
{
  "ts": "2026-09-25T19:27:16+0000",
  "decision": "GRANT",
  "name": "Alice",
  "similarity": 0.94,
  "box": [373, 82, 235, 336],
  "detect_score": 0.94,
  "liveness": 0.83
}
```

`decision` is `GRANT`, `DENY`, or `DENY_SPOOF`. This is the hook to use if you
want a second factor somewhere else in the system.

## Access log format

The same records are appended to `data/events.jsonl`, one JSON object per line,
newest last. Every record has `ts`, `decision`, `name`, `similarity` and `box`;
`detect_score` and `liveness` are added where they apply.

Repeat decisions for the same person are suppressed for
`access.event_cooldown` seconds — a person standing at the door is logged once,
not 20 times a second.

## Concurrency

Model work happens on the detection loop's thread. `/api/capture` queues a job
and waits for that loop to run it rather than touching the DNN objects from the
HTTP thread, because OpenCV DNN objects are not thread-safe. Under load,
`queue_depth` in `/api/status` will rise; a capture waits up to 15 s before
timing out.
