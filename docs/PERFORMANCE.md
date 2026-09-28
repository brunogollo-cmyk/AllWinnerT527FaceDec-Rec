# Performance

Every number here was measured on the target hardware, not estimated. The
machine:

```
Orange Pi 4A  ·  Allwinner T527 (sun55iw3)  ·  8x Cortex-A55 @ 1.8 GHz
4 GB LPDDR4  ·  Debian 12  ·  kernel 5.15.147-sun55iw3
640x480 MJPG USB webcam  ·  NPU: VeriSilicon VIP9000, 2 TOPS INT8
```

## Summary

| | |
|---|---|
| Face embedding, NPU | **15 ms** (median, p95 16.8, max 18.2) |
| Face embedding, CPU | 68 ms |
| Face detection, YuNet @ 256 | 17.6 ms |
| Liveness, MiniFASNet | 17 ms |
| **Steady state, face in view** | **23.5 ms/frame = 43 fps** |
| Empty scene | 22.0 ms/frame = 45 fps |
| CPU per frame | 133 ms (NPU) / 169 ms (CPU) |
| Cache hit rate, person still | 90 - 94% |

## Where the time goes

A recognised face, per frame:

```
YuNet detect      17.6 ms  ████
SFace embed        0-15   ██      ← NPU, and usually skipped by the cache
MiniFASNet        17.0   ████
                        ────
per frame         ~42 ms          (8.8 fps without the cache)
```

Recognition dominated the profile before these two optimisations. Embedding the
same unmoving face every frame cost 68 ms of a 114 ms frame, which is what
`tracking.py` removes.

With both optimisations the per-frame cost is back down to roughly the empty
scene's, and a recognised face is not more expensive than an unrecognised one —
because the liveness pass only runs on faces that would be granted.

## Thread scaling, and why fewer threads is not free

The A55 has 8 cores. How many OpenCV uses is a real trade-off, not a "more is
better" dial:

| threads | wall/frame | fps | CPU/frame | parallelism |
|---|---|---|---|---|
| 2 | 206.7 ms | 4.8 | 429 ms | 1.95x |
| 4 | 117.5 ms | 8.5 | 489 ms | 3.71x |
| 6 | 97.5 ms | 10.3 | 589 ms | 5.20x |
| 8 | 96.0 ms | 10.4 | 636 ms | 5.74x |

Going 2 → 8 threads buys 2.15x speed for **1.48x more CPU**. Past ~4 threads
the parallelism is mostly overhead: 5.74x of parallelism across 8 cores means a
quarter of the threads are contending for cache and synchronising.

So cutting threads is not a way to save CPU — dropping 8 → 4 saves 23% of CPU
and costs 18% of throughput. That is a trade, not a win, and it is why the
project reduces *work* rather than spreading it thinner.

## Why int8 is slower on the CPU

The OpenCV Zoo ships an int8 SFace. On this board it is **3.9x slower** than
float32:

| threads | float32 | int8 | ratio |
|---|---|---|---|
| 1 | 284.4 ms | 866.8 ms | 3.05x |
| 4 | 77.8 ms | 300.8 ms | 3.86x |
| 8 | 69.4 ms | 271.4 ms | 3.91x |

The A55 has `asimddp` (dot product) but **not `i8mm`**, and OpenCV's DNN
backend has no vectorised int8 path for this ISA. The quantised graph
decomposes into scalar integer sequences, and each one still pays thread
synchronisation — which is why the penalty *grows* with thread count.

This is the clearest argument for the NPU: the same quantised model runs there
in 15 ms, because the VIP9000 has int8 tensor cores that actually exist.

## Detection size

`detector.input_size` is the main detection lever. The cost tracks this value,
not the camera resolution, because the frame is resized to it first:

| `input_size` | detect |
|---|---|
| 160 | ~13 ms |
| 224 | 18 ms |
| 256 (default) | 22 ms |
| 320 | 30 ms |
| 512 | 81 ms |

## Accuracy numbers

| Measurement | Value |
|---|---|
| Same-person cosine, live | 0.55 - 0.69 |
| Enrolled intra-person similarity (11 photos) | 0.76 |
| Unrelated-person similarity | -0.13 to 0.13 |
| NPU vs CPU embedding agreement | cosine 0.9916 (min 0.9879) |
| Same identity decision, NPU vs CPU | 70/70 frames |

## Reproducing these

```bash
# per-stage timings on the live camera
./bench.sh -n 100 --sizes 224,256,320

# on a still image instead
./bench.sh -n 50 --offline /path/to/photo.jpg
```

`bench.sh` reports detect and embed times per `input_size`. It does not
exercise the NPU path or the reuse cache — for those, run `./run.sh` and read
the HUD and `/api/status`.

## Memory

| Component | Resident |
|---|---|
| Python + OpenCV | ~280 MB |
| SFace float32 model | 37 MB |
| NPU graph | 8 MB |
| YuNet | 0.2 MB |
| MiniFASNet | 1.9 MB |

Comfortable in 4 GB, leaving room for whatever else the board is doing — which
was the point of moving the embedding to the NPU.
