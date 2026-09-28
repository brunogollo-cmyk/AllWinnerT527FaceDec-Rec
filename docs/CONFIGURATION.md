# Configuration

Everything lives in `config.yaml`. Any key you omit falls back to the built-in
default, so the file is safe to trim down to just what you want to change.

Settings marked **writable** in the web UI's Settings tab can be changed at
runtime, with range validation. The rest need a restart.

## camera

```yaml
camera:
  device: null              # null = resolve by usb_id; a path forces a node
  usb_id: "1b3f:2002"       # "vvvv:pppp" from lsusb
  width: 640                # writable
  height: 480               # writable
  fps: 30                   # writable
  fourcc: "MJPG"            # MJPG or YUYV
  reconnect_attempts: 5
  reconnect_delay: 2.0
```

| Key | Notes |
|---|---|
| `device` | Pin `/dev/videoN` if USB-id resolution picks the wrong one. |
| `usb_id` | Preferred. The node number changes across reboots; the id does not. |
| `fourcc` | MJPG decodes in hardware and is much cheaper. YUYV avoids JPEG decode entirely and measured identical in cost. |

The camera negotiates its own size. If the request is not honoured the code
logs a warning — see [INSTALL.md](INSTALL.md#camera).

## detector

```yaml
detector:
  model: "models/yunet.onnx"
  input_size: 256           # writable, needs restart
  score_threshold: 0.7      # writable
  nms_threshold: 0.3
  top_k: 5000
```

### `input_size` — the main speed lever

Measured on this board:

| `input_size` | detect |
|---|---|
| 160 | ~13 ms |
| 224 | 18 ms |
| **256 (default)** | **22 ms** |
| 320 | 30 ms |
| 512 | 81 ms |

The cost tracks *this* value, not the camera resolution: the frame is resized to
it before inference. Raising it costs real time, and there is no accuracy gain
at the distances an access-control camera actually sees.

### `score_threshold` — do not lower it

This one has a non-obvious failure mode. Face detection confidence correlates
with embedding quality at **+0.63** on this board: a face detected at a low
score is poorly localised, and a poorly localised face produces an embedding
that does not match the database.

Measured on a live face:

| `score_threshold` | frames below the 0.5 match threshold |
|---|---|
| 0.5 | 22% |
| **0.7 (default)** | **0%** |
| 0.8 | 0% |

Lowering it to catch people in bad light makes recognition *worse*, not
better. Fix the lighting instead.

## recognizer

```yaml
recognizer:
  model: "models/face_recognition_sface_2021dec.onnx"
  antispoof_model: "models/antispoof_minifas.onnx"
  facedb: "data/facedb.json"
  use_npu: true
  npu_model: "models/npu/sface_int8.nb"
  viplite_dir: "/path/to/ai-sdk/viplite-tina/lib/glibc-gcc10_2_0/v1.13"
  match_threshold: 0.5      # writable
  min_face_size: 60         # writable
  reuse_window: 1.5
  reuse_min_iou: 0.5
  reuse_min_appearance: 0.9
```

### `match_threshold` — writable

Cosine similarity needed to accept a face. Enrolled people score **0.55-0.69**
live on this camera; strangers score near zero. The default sits in a wide
empty gap.

Lower it if legitimate people get denied, raise it if strangers get in. If
enrolment photos are inconsistent the whole distribution shifts down, and the
fix is better photos, not a lower threshold.

### `min_face_size` — writable

Faces smaller than this are ignored rather than matched. A too-small face gives
an unreliable embedding, and that is the main way a false accept happens. Raise
it if the camera sees the door from too far away.

### The `reuse_*` keys

See [ARCHITECTURE.md](ARCHITECTURE.md#2-embedding-reuse). Summary:

| Key | Default | Effect |
|---|---|---|
| `reuse_window` | 1.5 s | how long an embedding may be reused. `0` disables reuse entirely. |
| `reuse_min_iou` | 0.5 | bounding-box overlap required |
| `reuse_min_appearance` | 0.9 | appearance-descriptor similarity required |

Lower `reuse_min_appearance` and you risk handing one person another's
identity. A false reuse grants access; a false miss only costs one re-embed.
Keep it biased high.

Raise `reuse_window` if one person stands at the door a long time and you want
maximum speed. Lower it if several people alternate in front of the camera and
you want identity changes picked up quickly.

## access

```yaml
access:
  require_match: true       # writable toggle
  require_liveness: true    # writable toggle
  liveness_threshold: 0.3   # writable
  event_cooldown: 5.0       # writable
  events: "data/events.jsonl"
  webhook: null
```

| Key | Notes |
|---|---|
| `require_match` | `false` logs unknown faces as a grant. Not recommended. |
| `require_liveness` | `false` disables anti-spoofing entirely. A photograph will be let in. |
| `liveness_threshold` | See [SECURITY.md](SECURITY.md#the-threshold-has-no-comfortable-value) before changing this. |
| `event_cooldown` | Suppresses repeated decisions for the same person within this window. |
| `webhook` | Optional URL that receives a JSON POST per access event. Useful for requiring a second factor elsewhere. |

## web

```yaml
web:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  quality: 70               # writable, JPEG quality of the MJPEG stream
```

**There is no authentication.** Anyone who can reach this port can enrol and
delete people and change thresholds. Bind to `127.0.0.1` and use an SSH tunnel,
or put it behind a proxy with auth, if the board is reachable beyond a trusted
LAN. See [SECURITY.md](SECURITY.md#the-web-interface-has-no-login).

## display

```yaml
display:
  show: false               # opens an OpenCV window; needs a desktop session
```

## Changing settings at runtime

The Settings tab writes to `config.yaml` **by line**, not by round-tripping it
through a YAML dump. The file is commented, and a dump would delete every
comment. The same applies to the API's `POST /api/config`.

Changes to `detector.input_size` are accepted but flagged as needing a restart,
because the network is constructed once at startup.

### The complete writable list

Nothing outside this list can be changed through the API, which is what keeps a
stray request from altering something structural like which model file is
loaded.

| Key | Range |
|---|---|
| `recognizer.match_threshold` | 0.0 - 1.5 |
| `recognizer.min_face_size` | 10 - 1000 |
| `recognizer.reuse_window` | 0.0 - 10.0 |
| `recognizer.reuse_min_iou` | 0.0 - 1.0 |
| `recognizer.reuse_min_appearance` | 0.0 - 1.0 |
| `access.liveness_threshold` | 0.0 - 1.0 |
| `access.event_cooldown` | 0.0 - 3600.0 |
| `detector.score_threshold` | 0.0 - 1.0 |
| `detector.input_size` | 64 - 960 (needs restart) |
| `web.quality` | 30 - 100 |
| `camera.width` | 160 - 1920 |
| `camera.height` | 120 - 1080 |
| `camera.fps` | 1 - 60 |
| `access.require_match` | toggle |
| `access.require_liveness` | toggle |

In particular `recognizer.use_npu`, `npu_model` and `viplite_dir` are **not**
writable — switching the NPU path needs a restart, because the graph is loaded
once at startup.
