# Installation

## Requirements

- Orange Pi 4A (Allwinner T527 / `sun55iw3`), or another aarch64 Linux board
- Debian 12 / Ubuntu 22.04 or newer, Python 3.10+
- A USB webcam
- ~500 MB free for the venv, models and the NPU graph

## Install

```bash
git clone <this-repo>
cd facegate
./install.sh
```

This creates a virtualenv, installs the dependencies, and downloads the three
ONNX models, verifying each one's exact byte size.

`opencv-contrib-python-headless` is required — **not** plain `opencv`. The face
DNN classes this project uses (`FaceDetectorYN`, `FaceRecognizerSF`) only exist
in the contrib build. The script checks for them and fails loudly if the wrong
build was installed.

The model downloads need working internet access on first run. They come from:

| Model | Host |
|---|---|
| `yunet.onnx` | `media.githubusercontent.com` (OpenCV Zoo uses Git LFS) |
| `face_recognition_sface_2021dec.onnx` | `media.githubusercontent.com` (LFS) |
| `antispoof_minifas.onnx` | `raw.githubusercontent.com` (plain blob) |

## The NPU graph (optional)

The `.nb` graph for the T527's NPU is a build artifact and is not committed,
because producing it needs an x86_64 machine with Allwinner's ACUITY Docker
image. To use the NPU path, follow [NPU.md](NPU.md).

Without it, facegate runs entirely on the CPU and everything still works.

## Camera

Any UVC webcam works. facegate resolves the camera by **USB vendor/product id**
rather than `/dev/videoN`, because the node number usually changes across
reboots. Find yours:

```bash
lsusb | grep -i camera
# Bus 001 Device 005: ID 1b3f:2002 Generalplus Technology Inc. 808 Camera
```

Then set it in `config.yaml`:

```yaml
camera:
  usb_id: "1b3f:2002"
```

Or pin a specific node with `camera.device: /dev/video1`.

**User access.** The camera device is group `video`, and the service needs to be
in that group:

```bash
sudo usermod -aG video "$USER"
```

Log out and back in for that to take effect.

**Format.** MJPG at 640x480 is the cheap path — the JPEG is decoded in hardware.
Many of these cameras only offer MJPG at 1280x720 and 1920x1080; asking for
smaller is silently ignored and the driver keeps its own size. The code logs a
warning when the negotiated size differs from the request, so this is visible
rather than mysterious.

## Enrol people

Put photos in one folder per person, then run the enrolment:

```
faces/
├── alice/    front.jpg  side.jpg  smile.jpg
└── bob/      front.jpg  glasses.jpg
```

```bash
./enroll.sh
```

Use **3-8 photos per person**, at least one facing the camera, in the light the
door will actually see. The script reports a within-person similarity per
person; anything below ~0.7 means those photos are too inconsistent and
matching will be unreliable for that person.

Images with no face, or several faces, are skipped with a warning rather than
silently embedded.

A single photo works but is not recommended — it is what the database looked
like in testing, and the same person then scored 0.42 against it, below the
0.5 match threshold, and was denied. With 6+ good photos the same person scores
0.55-0.69.

## Run

```bash
./run.sh
```

Then open `http://<board-ip>:8080`.

Useful flags:

| Flag | Effect |
|---|---|
| `--no-web` | recognition only, no web server |
| `--show` | open an OpenCV window (needs a desktop session) |
| `-d /dev/video1` | force a camera node |
| `-c PATH` | use an alternative config file |
| `-v` | verbose logging |
| `--list-people` | print the enrolled names and exit |
| `--remove NAME` | delete someone's enrolment |
| `--print-config` | show the effective configuration and exit |

## Run at boot

```bash
sudo cp facegate.service /etc/systemd/system/
sudo systemctl enable --now facegate
journalctl -u facegate -f
```

Edit `facegate.service` first if your install is not in the home directory of
the user running it — it references the venv and the working directory by path.

## Licence

The code is MIT. The models come from their own projects and keep their own
terms: YuNet and SFace from [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
(Apache-2.0), MiniFASNet from
[facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx).
The NPU graph is a build artifact derived from the SFace model.
