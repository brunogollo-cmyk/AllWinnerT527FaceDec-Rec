# Rebuilding the NPU graph for facegate

`models/npu/sface_int8.nb` is the SFace recogniser, converted to INT8 and
compiled for the T527's VeriSilicon VIP9000 NPU. This is what lets facegate
embed a face in ~15 ms instead of ~68 ms.

This document exists because regenerating the file is not a one-liner: the
converter (Vivante Acuity) only runs inside an x86_64 Docker image that Allwinner
distributes separately, so the model is built on another machine and copied here.

## What the conversion actually is

```
face_recognition_sface_2021dec.onnx   (37 MB, float32, 9.67M params)
        │  pegasus import onnx
        ▼
sface.json + sface.data                 (Acuity intermediate)
        │  pegasus quantize  (perchannel_symmetric_affine, int8, KL)
        ▼
sface.quantize                          (activation ranges)
        │  pegasus export ovxlib --pack-nbg-unify
        ▼
network_binary.nb                       (7.9 MB, runs on VIP9000)
```

The target string `VIP9000NANOSI_PLUS_PID0X10000016` is this board's chip id -
`vip_query_hardware` reports `cid=0x10000016` at runtime. Using the wrong
target produces a `.nb` the driver rejects.

Note the T527 ships Acuity **v1.8.13**; the `v2.0.x` image is for the A733.

## Measured result

| | CPU | NPU |
|---|---|---|
| SFace embed | ~68 ms | **~15 ms** (median, p95 16.8) |

Agreement between the two paths, over 20 real camera crops:

```
cosine(NPU, CPU) = 0.9916   (min 0.9879, max 0.9946)
```

In a live run against the enrolled float32 database, the NPU and CPU paths
produced the **same identity decision on 70/70 frames** with mean pairwise
cosine 0.9998. That is why `data/facedb.json` needs no re-enrolment when you
switch `recognizer.use_npu`.

## Regenerating it

### 1. On an x86_64 machine, get the ACUITY image

From Radxa's docs (which point at Allwinner's netdisk):

- T527: `https://netstorage.allwinnertech.com:5001/sharing/N6TVlZQVZ` -> `docker_images_v1.8.x.zip`
- A733: `.../sharing/Mh23BhPHq` -> `docker_images_v2.0.x.zip`

```bash
unzip docker_images_v1.8.x.zip
cd docker_images_v1.8.x
unzip ubuntu-npu_v1.8.13.tar.zip
md5sum -c ubuntu-npu_v1.8.13.tar.zip_md5sum.txt
sudo docker load -i ubuntu-npu_v1.8.13.tar
sudo docker run --ipc=host -d -v "$PWD/work:/workspace" \
    --name acuity_t527 ubuntu-npu:v1.8.13 sleep infinity
```

### 2. Import

```bash
pegasus import onnx \
  --model sface.onnx --inputs data --outputs fc1 \
  --input-size-list "3,112,112" \
  --output-model sface.json --output-data sface.data
```

Input/output tensor names come from the ONNX graph: `data` in, `fc1` out.

### 3. Calibrate

Quantisation needs real activation ranges, so use real faces - not random
noise. Capture 50-100 aligned 112x112 crops from the actual camera.

`calib_database.txt` is a bare list of image paths (no header; the shape comes
from the model), and the input meta must point at it:

```yaml
preproc_type: IMAGE_RGB
cal_database: /workspace/calib_database.txt
iterations: 60
```

Take the `pegasus generate inputmeta` output and add those two keys.

```bash
pegasus quantize \
  --model sface.json --model-data sface.data \
  --quantizer perchannel_symmetric_affine --qtype int8 \
  --with-input-meta calib_meta.yml --iterations 60 \
  --algorithm kl_divergence --device CPU --output-dir quant
```

### 4. Export

```bash
pegasus export ovxlib \
  --model sface.json --model-data sface.data \
  --model-quantize quant/sface.quantize \
  --dtype quantized --with-input-meta calib_meta.yml \
  --pack-nbg-unify \
  --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk /root/Vivante_IDE/VivanteIDE5.8.2/cmdtools \
  --output-path nbg
```

`--pack-nbg-unify` is what produces a portable `.nb`; without it you get C
source for a different runtime. Note that `--output-path` is *concatenated* onto
the directory, so `nbg` lands in `./nbg_unify/`, not `./nbg/nbg_unify/`.

### 5. Copy here

```bash
scp nbg_unify/network_binary.nb \
    facegate/models/npu/sface_int8.nb
```

## The VIPLite runtime

`recognizer.viplite_dir` points at a copy of the Allwinner VIPLite v1.13
runtime. It comes from the public SDK, no NDA needed:

```bash
git clone --depth 1 -b product-aiot-stable \
    https://gitlab.com/tina5.0_aiot/ai-sdk.git
# -> viplite-tina/lib/glibc-gcc10_2_0/v1.13/{libVIPlite.so,libVIPuser.so,inc/}
```

Use the `glibc-gcc10_2_0` build. The `aarch64-none-linux-gnu` build is for a
bare-metal toolchain, and any Ubuntu 24.04 build of the tools wants glibc 2.38
while this board runs 2.36.

## Using it

```yaml
recognizer:
  use_npu: true
  npu_model: "models/npu/sface_int8.nb"
```

Set `use_npu: false` to go back to the CPU float32 model. The two produce
interchangeable embeddings, so switching does not invalidate the database.

If anything is missing - the graph, the library, or `/dev/vipcore` - facegate
logs `NPU backend unavailable (...) - using CPU` and keeps working.

## Files

- `facegate/npu.py` - ctypes binding for VIPLite. Mirrors the vendor's
  `vpm_run.c` call sequence, and queries buffer descriptors from the graph
  rather than hardcoding them, so a differently-shaped model works unchanged.
- `tools/npu_embed.c` - standalone CLI that prints one embedding. Useful for
  A/B checking the graph against the CPU model:
  ```bash
  gcc -O2 -I<sdk>/inc -o npu_embed tools/npu_embed.c -L<sdk> -lVIPlite -lVIPuser -lm
  LD_LIBRARY_PATH=<sdk> ./npu_embed models/npu/sface_int8.nb input.dat
  ```
  `input.dat` is float16 NCHW 1x3x112x112 RGB.
