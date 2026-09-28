# Troubleshooting

## The camera

**`No camera matching usb_id '1b3f:2002'`**

The id in `config.yaml` does not match what is plugged in. Find the real one:

```bash
lsusb | grep -i camera
```

The `1b3f:2002` in the default is a Generalplus 808 camera. If you swapped
cameras, update `camera.usb_id`, or set `camera.device: /dev/videoN` to pin a
node.

**The camera is plugged in but facegate cannot see it**

Check group membership — the device is `root:video`:

```bash
ls -l /dev/video*
groups | grep video       # your user must be in it
sudo usermod -aG video "$USER"
```

Log out and back in; group changes do not apply to a running session.

**Permission denied opening /dev/videoN**

Same cause. Or, if you are on a minimal install without udev rules, create one:

```bash
echo 'SUBSYSTEM=="video", GROUP="video", MODE="0660"' | \
    sudo tee /etc/udev/rules.d/70-video.rules
sudo udevadm control --reload-rules
```

**Frame size is not what I asked for**

Many of these cameras only offer MJPG at 1280x720 and 1920x1080. Asking for
smaller is silently ignored and the driver keeps its own size. facegate logs a
warning when the negotiated size differs from the request, so check the startup
log. YUYV 640x480 is also available and measured identical in cost.

**`Corrupt JPEG data: premature end of data segment`**

A frame was truncated mid-decode — usually the camera was unplugged or
re-negotiated. facegate reconnects automatically (`camera.reconnect_attempts`).
Occasional occurrences are harmless; a flood means the cable or power is
marginal.

## Recognition

**I am denied even though I enrolled**

Check the similarity, not just the enrolment. Two things cause this:

1. **Too few or inconsistent photos.** The People tab shows a self-similarity
   score; below 0.7 the threshold cannot be trusted for that person. Re-enrol
   with 3-8 good photos. A single photo scored 0.42 and produced a denial in
   testing; 11 photos scored 0.76.
2. **Poor framing at the door.** A 45-degree off-axis view degrades cosine
   similarity substantially. Mount the camera at face height, straight on.

**Strangers get in**

Raise `recognizer.match_threshold`. Also consider that faces below
`min_face_size` are already ignored, and that raising
`access.require_liveness` is doing nothing useful — the anti-spoof model is
unvalidated (see [SECURITY.md](SECURITY.md)).

**Detection is inconsistent / people are missed**

Do **not** lower `detector.score_threshold` to compensate. Detection confidence
correlates with embedding quality at +0.63, so admitting low-confidence
detections makes recognition worse: at 0.5, 22% of frames fell below the match
threshold; at 0.7, zero did. Fix the lighting instead.

**The HUD shows `emb 0.0ms`**

That is the embedding cache working — the face did not change, so SFace was
skipped. Look at `reuse N/M` next to it. If a completely still face shows
`reuse 0/N`, the gates are too strict and you are paying full price for nothing;
lower `reuse_min_appearance` a little.

## The NPU

**`NPU backend unavailable (...) - using CPU` at startup**

Expected and harmless — facegate continues on the CPU. The reason is in the
message. The common ones:

| Message | Cause |
|---|---|
| `not found` (model) | the `.nb` was not built, see [NPU.md](NPU.md) |
| `libVIPlite.so not found` | `viplite_dir` is wrong |
| `/dev/vipcore missing` | the NPU kernel driver is not loaded — see below |
| `vip_init failed` | the driver loaded but the runtime cannot attach |

**How do I check the NPU is actually being used?**

```bash
./run.sh
# INFO facegate.npu: NPU graph loaded: sface_int8.nb driver=0x00010d00 cid=0x10000016 ...
# INFO facegate.recognizer: Face embedding on NPU (sface_int8.nb), ~15 ms vs ~68 ms on CPU
```

If you see `using CPU` instead, work through the table above.

**`viplite doesn't support this buffer format`**

The graph was compiled for a different chip. The target must be
`VIP9000NANOSI_PLUS_PID0X10000016` for the T527 — check the `cid` printed at
startup.

**`input buffer alloc failed` with a nonsensical shape**

A VIPLite struct layout mismatch. The v1.13 `vip_buffer_create_params_t` puts
`memory_type` **last** and sizes `sizes[6]`; the v2.0 layout differs (32 vs 56
bytes for the allocation struct) and the driver returns `status=-11`. Use the
`glibc-gcc10_2_0` runtime from the T527 SDK, not a v2.0 build.

**NPU results lag one frame behind**

A missing `vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_FLUSH)` after writing
input. It produces no error at all — the output is just one call stale. If you
have modified `npu.py`, check that flush is there.

**`vip_create_network ... not support this create network type=0x0`**

`VIP_CREATE_NETWORK_FROM_FILE` is `0x01`, not `0`.

## Performance

**It is slow**

Work through these in order of effect:

1. Raise `detector.input_size`? No — **lower** it. 512 costs 81 ms vs 22 ms at 256.
2. Is embedding reuse working? Check `reuse N/M` in the HUD. Below ~50% and
   the cache is not earning its keep.
3. Is the NPU active? See above.
4. More faces in frame means more embeds and more liveness passes. That is
   inherent, not a bug.

**`GIL` / CPU contention with other work**

That is what the NPU and the cache are for. With both, CPU per frame is ~133 ms
out of a possible 1000 ms, so there is room for other work. The
`/api/status` `detect_ms` and `embed_ms` fields show where time is going.

## The web UI

**Port 8080 already in use**

```bash
ss -tlnp | grep 8080
```

Change `web.port` in `config.yaml`.

**The live view is blank**

Check `camera_connected` in `/api/status`. If the stream was interrupted, the
page reconnects it every few seconds; a hard reload also helps.

**The live view is stuttering**

Lower `web.quality`. The stream is MJPEG at 640x480 and the per-frame work is
real, not a network problem.

**Anyone can reach the UI and change things**

That is by design. See
[SECURITY.md](SECURITY.md#the-web-interface-has-no-login) — bind to
`127.0.0.1` and use an SSH tunnel, or put it behind a proxy with auth.

## Install

**`This OpenCV build lacks the face DNN classes`**

You need `opencv-contrib-python-headless`, not plain `opencv`. The face DNN
classes only exist in the contrib build. Re-run `./install.sh`.

**Model download fails**

The models come from GitHub. `yunet.onnx` and the SFace model are served from
`media.githubusercontent.com` because OpenCV Zoo stores them in Git LFS; the
anti-spoof model is a plain blob on `raw.githubusercontent.com`. If you are
behind a proxy, that may need configuring. The script verifies each file's
exact byte size and will tell you if a download was truncated.
