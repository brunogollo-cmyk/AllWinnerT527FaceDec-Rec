# Architecture

How a frame becomes an access decision, and why the pieces are split the way
they are.

## The pipeline

Everything happens in one thread, in `facegate/pipeline.py`. A single thread
owns the three DNN models, which sidesteps a real hazard: OpenCV DNN objects
are not documented as thread-safe, and the web layer would otherwise call them
concurrently.

```
grabber.read()            V4L2, 640x480 MJPG
  │
  ├── detector.detect()   YuNet, 256x256 input      ~17 ms  CPU
  │      └── faces[]      box, 5 landmarks, score
  │
  ├── for each face
  │     ├── too small?    → skip, no decision
  │     ├── recognizer.align()   similarity transform → 112x112
  │     ├── cache.lookup()        reuse a still face's embedding
  │     ├── recognizer.embed_aligned()   SFace     ~15 ms  NPU
  │     ├── db.match()       cosine vs every enrolled vector
  │     └── if matched → liveness.check()   MiniFASNet  ~17 ms  CPU
  │
  ├── annotate frame
  └── log the decision
```

## Why detection and liveness stay on the CPU

They are small models and they run every frame. The NPU has per-inference
overhead (buffer setup, graph submission) that is not worth paying for a 17 ms
convolution it will not speed up much, and moving them would add a second
conversion step to the build for no measured gain.

The embedding is the opposite: a 9.7M-parameter model, run constantly, and the
single most expensive thing in the pipeline. That is the one that belongs on
accelerated hardware.

## Two independent optimisations

Recognition dominates cost, and the two things that address it are separate
mechanisms that stack.

### 1. The NPU

The A55 has `asimddp` but not `i8mm`, and OpenCV's DNN backend has no
vectorised int8 path for it. Measured on this board, the int8 SFace model runs
**3.9x slower** than float32 on the CPU:

| Model | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| SFace float32 | 284 ms | 78 ms | 69 ms |
| SFace int8 | 867 ms | 301 ms | 271 ms |

The NPU has real int8 tensor cores, so the same quantised model runs there in
**~15 ms**. See [NPU.md](NPU.md).

### 2. Embedding reuse

A face standing at the door is re-embedded on every frame even though nothing
about it has changed. At ~15-68 ms per embedding, that is the larger waste.

`facegate/tracking.py` caches the last embedding per tracked face and reuses it
while two independent gates agree the face is the same one:

| Gate | Default | Guards against |
|---|---|---|
| `reuse_min_iou` | 0.5 | a different face elsewhere in the frame |
| `reuse_min_appearance` | 0.9 | **a different person in the same spot** |

The appearance gate is the one that matters for security. Without it, someone
walking into the same bounding box would inherit the previous person's cached
identity — and would be granted access. It compares a 16x16 greyscale
descriptor of the aligned crop, mean-centred so auto-exposure drift does not
matter. Measured separation:

```
same person, 8px shift        cos = 0.945
same person, exposure -25%    cos = 0.999
same person, sensor noise     cos = 0.988
hard negative (mirrored face) cos = 0.780
```

The default 0.9 sits in that gap. **The error is deliberately asymmetric:** a
false reuse grants access, while a false miss only costs one re-embed. Keep it
biased high.

What reuse does *not* cache is the database match. It re-runs every frame on
whatever embedding is in hand, so enrolling someone or changing
`match_threshold` takes effect immediately.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | dataclasses, defaults, YAML merge, path resolution |
| `camera.py` | V4L2 capture, USB-id resolution, reconnect on unplug |
| `detector.py` | YuNet wrapper; resizes to `input_size`, rescales boxes back |
| `recognizer.py` | embedding (NPU or CPU) + the `FaceDB` |
| `npu.py` | ctypes binding for Allwinner VIPLite |
| `tracking.py` | the embedding cache and its two gates |
| `liveness.py` | MiniFASNet, crop/letterbox/preprocess, softmax |
| `enroll.py` | shared enrolment rules, used by both the CLI and the web UI |
| `pipeline.py` | per-frame orchestration, annotation, HUD |
| `service.py` | job queue so the web thread never touches the models |
| `events.py` | JSONL access log + optional webhook |
| `web.py` | JSON API, static files, MJPEG stream |

## Two design decisions worth explaining

**All model work happens on one thread.** The web layer submits jobs to a
queue that the detection loop drains between frames, and waits for the result.
Otherwise a `/api/capture` from a browser would race the loop on the same
`cv2.dnn` objects. It also means the queue can back up under load, which is why
`queue_depth` is in `/api/status`.

**The database holds several embeddings per person**, matched by nearest
neighbour. Averaging them into one vector blurs across expressions and head
angles, which raises false rejects. `mean_intra_person_similarity()` reports how
consistent a person's photos are, and enrolment warns when a set is too
inconsistent to trust.

## Alignment matters more than it looks

The 5 landmarks from YuNet drive a similarity transform onto a canonical
112x112 pose before embedding. Without correct alignment, embeddings of the
same person from different angles are not comparable, and recognition quietly
degrades. This is also why the detector's own confidence is a useful signal: a
face detected at a low score tends to be poorly localised, and a poorly
localised face produces an unreliable embedding. On this board, similarity
correlates with detector score at **+0.63**.

That is why `detector.score_threshold` is 0.7. At 0.5, roughly 22% of frames
fell below the match threshold; at 0.7, zero did.
