# Using the T527 NPU for face embedding

The Orange Pi 4A's Allwinner T527 carries a 2 TOPS VeriSilicon VIP9000 NPU.
`facegate` uses it for the SFace embedding, which is the most expensive step in
the pipeline.

| | CPU | NPU |
|---|---|---|
| SFace embed | ~68 ms | **~15 ms** |
| Agrees with the other path | — | cosine **0.9916** |

It is on by default. Set `recognizer.use_npu: false` to turn it off; the
database stays valid either way.

## How it works

The NPU does not run ONNX. It runs Vivante **`.nb` network binaries**, so the
SFace model was converted offline:

```
face_recognition_sface_2021dec.onnx     37 MB float32, 9.67M params
        │  ACUITY: pegasus import onnx
        ▼
   sface.json + sface.data
        │  pegasus quantize   perchannel_symmetric_affine, int8, KL divergence
        ▼
   sface.quantize                          activation ranges
        │  pegasus export ovxlib --pack-nbg-unify
        ▼
   network_binary.nb                       7.9 MB, runs on VIP9000
```

The compile target is `VIP9000NANOSI_PLUS_PID0X10000016`, which is this board's
chip id — the runtime reports `cid=0x10000016`. The wrong target produces a
`.nb` the driver silently refuses to run.

At load time, `facegate/npu.py` reads the graph's own buffer descriptors
(shape, data format, quantisation parameters) rather than hardcoding them, so a
re-converted model with different dimensions works without code changes.

## Requirements

```bash
# 1. The VIPLite runtime (public SDK, no NDA)
git clone --depth 1 -b product-aiot-stable \
    https://gitlab.com/tina5.0_aiot/ai-sdk.git

# 2. A kernel driver exposing /dev/vipcore
ls -l /dev/vipcore
```

`facegate` will use the `glibc-gcc10_2_0` build of VIPLite — it targets glibc
2.17+ and works on Debian 12. The `aarch64-none-linux-gnu` build is for a
bare-metal toolchain, and vendor builds targeting Ubuntu 24.04 need glibc 2.38,
which is newer than this board has.

## Configuration

```yaml
recognizer:
  use_npu: true
  npu_model: "models/npu/sface_int8.nb"
  viplite_dir: "/path/to/ai-sdk/viplite-tina/lib/glibc-gcc10_2_0/v1.13"
```

## Verifying it is actually running

```bash
./run.sh
```

At startup the log should say:

```
INFO facegate.npu: NPU graph loaded: sface_int8.nb driver=0x00010d00 cid=0x10000016 in=75264B ...
INFO facegate.recognizer: Face embedding on NPU (sface_int8.nb), ~15 ms vs ~68 ms on CPU
```

The Live tab's HUD shows `emb` per frame. On a fresh face you should see a
non-zero value around 15; on reused frames it shows 0 by design.

There is also a standalone tool that prints one embedding, which is the easiest
way to A/B the graph against the CPU model:

```bash
gcc -O2 -I<viplite>/inc -o /tmp/npu_embed tools/npu_embed.c \
    -L<viplite> -lVIPlite -lVIPuser -lm

# float16 NCHW 1x3x112x112, RGB
LD_LIBRARY_PATH=<viplite> /tmp/npu_embed \
    models/npu/sface_int8.nb input.dat
```

## Rebuilding the graph

The converter only runs inside an x86_64 Docker image that Allwinner
distributes, so this is done on another machine and the result copied over.
The full recipe — including the exact commands and the traps — is in
[`../tools/README-npu.md`](../tools/README-npu.md).

Two things that cost real time if you do not know them:

- **Use `v1.8.x` for the T527.** The `v2.0.x` image is for the A733. Mixing
  them produces a graph the driver rejects.
- **Calibrate with real faces.** Activation ranges come from the calibration
  set; random noise quantises badly and quietly degrades the embedding.

## Accuracy

Quantising to int8 does cost some precision. Measured on 20 real camera crops:

```
cosine(NPU, CPU) = 0.9916   (min 0.9879, max 0.9946)
```

More importantly, in a live run against the enrolled database the NPU and CPU
paths produced the **same identity decision on 70/70 frames**, with mean
pairwise cosine 0.9998. That is why switching `use_npu` does not invalidate
`data/facedb.json` — **nobody has to re-enrol**.

If you re-enrol anyway (for example after changing the lighting at the door),
do it with the backend you intend to run. Both directions are safe; mixing
databases across a *model* change is not.

## When it falls back to the CPU

`facegate` never fails because the NPU is unavailable. It logs and continues:

```
NPU backend unavailable (<reason>) - using CPU
```

The reasons it will fall back:

- `use_npu: false` in the config
- the `.nb` file is missing, or `viplite_dir` is wrong
- `libVIPlite.so` / `libVIPuser.so` not found
- `/dev/vipcore` does not exist — the NPU kernel driver is not loaded
- `vip_init` or `vip_create_network` fails

If the NPU errors *during* a frame, that one face is embedded on the CPU and
the loop continues. The error counter is visible as repeated warnings.

## Troubleshooting

**`viplite doesn't support this buffer format`** — the graph was compiled for a
different chip. Check the `--optimize` target against the `cid` printed at
startup.

**`input buffer alloc failed` with a garbage shape** — a struct layout
mismatch. The VIPLite v1.13 `vip_buffer_create_params_t` puts `memory_type`
**last** and sizes `sizes[6]`; the v2.0 layout is different (56 bytes vs 32 for
the allocation struct) and the driver returns `status=-11`. Use the v1.13
runtime that ships with the T527 SDK.

**`vip_create_network ... not support this create network type=0x0`** —
`VIP_CREATE_NETWORK_FROM_FILE` is `0x01`, not `0`.

**Everything returns the previous call's result** — a missing
`vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_FLUSH)` after writing input. It
produces no error at all; the output is just one call stale.
