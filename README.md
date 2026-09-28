[README.md](https://github.com/user-attachments/files/32713487/README.md)
# facegate

Real-time face detection and recognition on an Orange Pi 4A (Allwinner sun55iw3,
8x Cortex-A55, 4GB RAM, 2 TOPS VeriSilicon VIP9000 NPU). Face embedding runs
on the NPU; detection and liveness run on the CPU.

## Measured performance on this board

Numbers below are real measurements from this machine, not estimates.

| Stage | Time | Notes |
|---|---|---|
| Capture 640x480 MJPG | ~1 ms | camera, hardware JPEG decode |
| YuNet detect (256x256) | **17.6 ms** | CPU |
| SFace embed (NPU) | **~15 ms** | `recognizer.use_npu: true`, median, p95 16.8 |
| SFace embed (CPU) | ~68 ms | fallback, float32 ONNX |
| MiniFASNet liveness | ~17 ms | CPU, only on a face that would be granted |
| **Live, empty scene** | **22.0 ms/frame = 45 fps** | measured from the camera |
| **Live, one recognised face (NPU + reuse)** | **~23 ms/frame = 43 fps** | production default |
| **Live, one recognised face (CPU + reuse)** | ~23 ms/frame = 44 fps | |
| CPU per frame (NPU + reuse) | **~133 ms** | vs ~169 ms with the CPU embed |
| Cache hit rate, person standing still | 90 - 94% | |
| Same-person cosine similarity | 0.55 - 0.69 live | 11 enrolled photos, intra-person 0.76 |
| Unrelated-person similarity | -0.13 to 0.13 | |
| Genuine-face liveness score | 0.24 - 0.99 | long low tail; see the liveness section |

Recognition dominates cost, and the SFace pass is the bulk of it. Two things
address that, and they stack:

- **The NPU** runs the embedding in ~15 ms instead of ~68, because the A55 has
  `asimddp` but not `i8mm` and has no vectorised int8 path, while the VIP9000
  has real int8 tensor cores.
- **Embedding reuse** skips SFace entirely on frames where the face has not
  changed, which is where the larger win comes from - it is what took an empty
  scene to 45 fps and keeps a recognised face at 43.

With both, a recognised face costs less wall time than an unrecognised one
latched onto an empty scene, because the liveness pass only runs on faces that
would be granted.

## About the NPU

The T527's NPU only executes Vivante `.nb` network binaries, so the SFace ONNX
model was converted with the Allwinner ACUITY toolkit and compiled for this
board's chip (`VIP9000NANOSI_PLUS_PID0X10000016`). The result agrees closely
with the CPU model:

```
cosine(NPU, CPU) = 0.9916   over 20 real camera crops
```

and in a live run the two produced the **same identity decision on 70/70
frames**. That is why switching `recognizer.use_npu` does not invalidate
`data/facedb.json` - nobody has to re-enrol.

Set `use_npu: false` to go back to the CPU float32 model at any time. If the
graph, the VIPLite library or `/dev/vipcore` is missing, facegate logs
`NPU backend unavailable (...) - using CPU` and carries on.

`tools/README-npu.md` documents how the `.nb` was produced and how to rebuild
it, since the converter only runs in an x86_64 Docker image from Allwinner.


## Embedding reuse

`recognizer.reuse_window` (default `1.5`) enables it; set it to `0` to restore
the original always-re-embed behaviour.

A face is re-embedded only when it fails either gate:

- **`reuse_min_iou`** (0.5) — the new bounding box must overlap the cached one
- **`reuse_min_appearance`** (0.9) — a 16x16 greyscale descriptor of the
  aligned crop must be cosinally close to the cached one

The appearance gate is the one that matters for security. Without it, a
different person who walks into the same bounding box would inherit the first
person's cached identity — and would be granted access. The measured
separation on this board:

```
same person, 8px shift        cos=0.945
same person, exposure -25%    cos=0.999
same person, sensor noise     cos=0.988
hard negative (mirrored face) cos=0.780
```

The default 0.9 sits in that gap. The error is deliberately asymmetric: a
false reuse grants access, while a false miss only costs one re-embed, so this
should stay biased high.

The database match is **not** cached. It re-runs on whatever embedding is in
hand every frame, so enrolling someone or changing `match_threshold` takes
effect immediately even with reuse on.

Because a cached embedding skips SFace, the HUD `emb` timing shows 0 on
reused frames, and the status API reports `embed_reused` / `embed_fresh`.
Watch `reuse` in the live view: if a genuinely still face shows `reuse 0/N`,
the gate is too strict and you are paying full price for nothing.

## How it works

```
webcam 640x480 ──► YuNet detect (CPU) ──► 5 landmarks + box
                                          │
                            similarity transform to 112x112
                                          │
                          ┌───────────────┴───────────────┐
                    reuse gate passes              fresh embed
                    (cached embedding)         SFace 128-d on the NPU
                          └───────────────┬───────────────┘
                                          │
                            cosine match vs faces/<name>/*
                                          │
                        ┌─────────────────┴──────────────────┐
                  no match                            match found
                        │                                   │
                    DENY                            MiniFASNet liveness (CPU)
                                                            │
                                              live ──► GRANT / DENY_SPOOF
```

Three models do all the work, all executed by the ONNX runtime compiled into
OpenCV itself:

- **YuNet** detects faces and returns 5 landmarks per face
- **SFace** turns an aligned crop into a 128-d embedding
- **MiniFASNet V2 SE** classifies a face crop as live or spoofed

The landmarks are what make recognition work: they drive a similarity transform
that aligns each face to a canonical 112x112 pose before embedding. Without
correct alignment, SFace embeddings are not comparable between images.

Liveness runs **only on faces that would otherwise be granted**, so a stranger at
the door never costs the extra 17 ms.

## Install

```bash
cd ~/facegate
./install.sh
```

This creates a venv, installs `opencv-contrib-python-headless` (the *contrib*
build is required, the face DNN classes do not exist in plain OpenCV), and
downloads both models, verifying their exact byte size.

## Enroll people

Put photos in one folder per person:

```
faces/
├── alice/    front.jpg  side.jpg  smile.jpg
└── bob/      front.jpg  glasses.jpg
```

Then:

```bash
./enroll.sh
```

Use **3-8 photos per person**, at least one facing forward. The script reports a
within-person similarity score per person; anything below ~0.7 means those photos
are too inconsistent and matching will be unreliable for that person.

Images with no face, or several faces, are skipped with a warning rather than
silently embedded.

## Run

```bash
./run.sh
```

Then open **http://<board-ip>:8080** — that's the operator UI. It has four tabs:

- **Live** — the annotated camera view with fps and per-stage timings
- **People** — who's enrolled, and capture-and-enrol new people from the webcam
- **Events** — the access log, filterable by decision
- **Settings** — the tunables, with the measured numbers as guidance

Adding someone: put a name in, press **Capture photo** a few times from slightly
different angles, then **Enrol**. Each capture shows a thumbnail and whether it
already matches someone, and you can discard individual shots before committing.

The same access log is written to `data/events.jsonl`, one JSON object per
decision:

```json
{"ts":"2026-09-25T19:27:16+0000","decision":"GRANT","name":"alice","similarity":0.94,"box":[373,82,235,336],"detect_score":0.94,"liveness":0.83}
```

Useful flags: `--no-web`, `--show` (needs a desktop session), `-d /dev/video1`,
`--list-people`, `--remove NAME`, `--print-config`.

### There is no login

Anyone who can reach port 8080 on your network can enrol people, delete them and
change the security thresholds. That is fine on a trusted home or office LAN and
deliberately nothing more. If the board is reachable from anywhere else, put it
behind a reverse proxy with authentication, or bind `web.host` to `127.0.0.1`
and use an SSH tunnel.

## Run at boot

```bash
sudo cp facegate.service /etc/systemd/system/
sudo systemctl enable --now facegate
journalctl -u facegate -f
```

## Configuration

Everything lives in `config.yaml`; anything omitted uses the default. The
values you are most likely to change:

- **`recognizer.match_threshold`** (default `0.5`) — cosine similarity needed to
  accept a face. Enrolled people score 0.55-0.69 live on this camera and
  strangers score near zero, so the default sits in a wide empty gap. Lower it if
  legitimate people get denied, raise it if strangers get in.
- **`recognizer.use_npu`** (default `true`) — run the embedding on the NPU. Set
  it to `false` to use the CPU float32 model instead; the two are
  interchangeable, so the database stays valid either way.
- **`recognizer.reuse_window`** (default `1.5`) — how long an embedding may be
  reused for a still face. `0` disables reuse and restores always-re-embed.
- **`detector.input_size`** (default `256`) — the biggest detection speed lever.
  Measured on this board: 224 → 18ms, 256 → 22ms, 320 → 30ms, 512 → 81ms.
  Note the cost tracks *this* value, not the camera resolution: the frame is
  resized to this size before inference, so raising it costs real time.

## Liveness detection (anti-spoofing)

A photograph held to the lens matches the face database perfectly — it *is* the
same pixels. So a third model, **MiniFASNet V2 SE**, checks each face that is
about to be granted and judges it live or spoofed. Spoofed faces are logged as
`DENY_SPOOF` and shown in amber.

```yaml
access:
  require_liveness: true      # false disables the check entirely
  liveness_threshold: 0.3
```

### Read this before relying on it

**I could not validate the liveness threshold against real print or screen
attacks.** I have no physical printed photo or a phone screen to test with, and
the public anti-spoof datasets are large and gated. What I could do is test
software-synthesised attacks, and the honest result is that they are not
representative:

| Test | Liveness score | Verdict |
|---|---|---|
| Genuine face photos | 0.50 - 0.90 | live (correct) |
| Low-contrast "print" | 0.02 | spoof (correct, but it also failed face *detection*) |
| Over-bright "screen" | 0.47 | spoof (correct) |
| Moiré screen replay | 0.86 | **live (missed the attack)** |
| Low-quality JPEG | 0.94 | **live (missed the attack)** |
| Upscaled/resampled | 0.63 | **live (missed the attack)** |

The last three are the concern: realistic digital spoofs sail through. So
`0.5` is a **starting point, not a validated setting**, and I would not treat
this as door-grade security yet.

### The threshold has no comfortable value

Measured against a real face at the camera (92 granted frames, production
config with embedding reuse on), the genuine-face liveness score is not
clustered around some clear "live" value — it has a long low tail:

```
min 0.244 | p1 0.289 | p5 0.389 | median 0.822 | max 0.985
```

That distribution collides with the spoof scores in the table above:

| Threshold | False `DENY_SPOOF` on a genuine face |
|---|---|
| 0.5 (old default) | 8.7% of frames |
| 0.4 | 5.4% |
| 0.3 (current default) | 2.2% |
| 0.25 | 1.1% |
| 0.2 | 0.0% |

So there is no threshold that is both free of false denials and still above
the one real spoof this project has measured (`0.47` for the over-bright
screen). `0.2` would give a live person a clean pass, but it sits *below* that
attack. **This is a genuine limitation of MiniFASNet on a cheap webcam, not a
tuning problem you can solve by picking a better number.**

`0.3` is the current default because a false `DENY_SPOOF` locks a real
person out of their own door, which is the worse failure for a single-person
installation. If you enrol several people, or the door is unattended, raise it
back toward 0.5 and accept the false denials.

To tune it yourself, print a photo of an enrolled person, hold it to the camera,
and read the score off the `PHOTO/SPOOF` box in the live view. If the score stays
above your threshold, raise it.

Options if you need real security:
- raise `liveness_threshold` after your own testing
- require a **challenge-response** gesture (turn head, blink) — much stronger,
  needs a tracking implementation
- add a depth sensor or a second camera at an angle; geometry beats heuristics
- add `access.webhook` to forward every decision to a system that can require a
  second factor



Built for access control, so it fails closed:

- An unrecognised face is **DENY**, not grant (`access.require_match: true`)
- Faces smaller than `min_face_size` (60px) are never matched. A small face gives
  an unreliable embedding, and that is the main way a false accept happens
- Multiple embeddings are kept per person and matched by nearest neighbour.
  Averaging them into one vector blurs across expressions and raises false rejects

## Notes and limits

- **The camera is resolved by USB vendor/product id** (`1b3f:2002`), not by
  `/dev/videoN`, because the node number usually changes across reboots. Change
  `camera.usb_id` if you swap cameras; `camera.device` forces a fixed node.
- **This camera only offers MJPG at 1280x720 and 1920x1080.** Asking for
  640x480 MJPG works, but anything smaller is silently ignored and the driver
  keeps 640x480. The code logs a warning when the negotiated size differs from
  the requested one, so this is visible rather than mysterious. YUYV 640x480 is
  also available and measured identical in cost.
- **Recognition accuracy depends entirely on the camera angle at the door.**
  A 45-degree off-axis view degrades SFace cosine similarity substantially. Mount
  the camera roughly at face height, straight on.
- Liveness detection is present but **unvalidated against real spoofs** - read
  the liveness section above before relying on it. A printed photo is not
  guaranteed to be caught.
- `setPreferableTarget` warnings from OpenCV are expected and harmless here; they
  mean the new graph engine ignores device targeting, which is fine on CPU.

## Layout

```
facegate/
├── config.py        config dataclasses + YAML loading
├── camera.py        V4L2 capture, USB-id resolution, reconnect
├── detector.py      YuNet wrapper, resizes and rescales boxes
├── recognizer.py    SFace embedding (NPU or CPU) + the face database
├── npu.py           ctypes binding for the Allwinner VIPLite runtime
├── tracking.py      embedding reuse with the box + appearance gates
├── liveness.py      MiniFASNet anti-spoofing check
├── enroll.py        shared enrolment rules (used by CLI and web)
├── service.py       job queue, app state, comment-preserving config writes
├── pipeline.py      per-frame orchestration and annotation
├── events.py        JSONL access log + optional webhook
├── web.py           JSON API + static file serving
├── static/          operator UI (index.html, app.js, style.css)
└── tools/           NPU graph rebuild notes + a standalone embed CLI
```

`bench.py` measures real timings: `./bench.sh -n 100 --sizes 224,256,320`

## HTTP API

The UI is a thin client over these; they are usable directly with `curl`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | uptime, fps, per-stage ms, camera, counts |
| GET | `/api/people` | enrolled names, embedding counts, self-similarity |
| GET | `/api/events?limit=N` | recent access events, newest first |
| DELETE | `/api/events` | clear the access log |
| POST | `/api/capture` | grab one frame, return embedding + thumbnail + liveness |
| POST | `/api/enroll` | `{"name":..., "features":[[128 floats], ...]}` |
| POST | `/api/people/remove` | `{"name":...}` |
| GET | `/api/config` | effective config |
| POST | `/api/config` | update writable settings |
| GET | `/api/snapshot` | current annotated frame as a JPEG |
| GET | `/stream` | MJPEG live view |

Only a whitelist of settings is writable, each range-checked server-side.
`detector.input_size` is accepted but flagged as needing a restart.

Captures happen on the detection loop's own thread: OpenCV DNN objects are not
thread-safe, so `/api/capture` queues work and waits for the loop to run it
rather than touching the models from the HTTP thread.

The enrolment CLI now lives at `scripts/enroll_cli.py` and is a thin wrapper
over `facegate/enroll.py`, so the CLI and the web UI cannot disagree about what
counts as a usable photo. `./enroll.sh` still works exactly as before.
