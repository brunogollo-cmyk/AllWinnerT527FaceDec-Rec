#!/usr/bin/env bash
# One-time setup for facegate: venv, dependencies and the two ONNX models.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

say "Creating virtualenv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip

say "Installing dependencies"
# opencv-contrib (not plain opencv) because the face DNN classes we rely on --
# FaceDetectorYN and FaceRecognizerSF -- are only in the contrib build.
# The manylinux_2_28_aarch64 wheel is a real arm64 build; piwheels has no
# aarch64 OpenCV at all, so it is important this resolves from PyPI itself.
"$VENV/bin/pip" install --quiet \
    "opencv-contrib-python-headless>=4.10" \
    numpy \
    pyyaml

say "Verifying the OpenCV face DNN classes are present"
"$VENV/bin/python" - <<'PY' || die "This OpenCV build lacks the face DNN classes. Install opencv-contrib-python-headless >= 4.10."
import cv2
missing = [n for n in ("FaceDetectorYN", "FaceRecognizerSF") if not hasattr(cv2, n)]
print(f"  opencv {cv2.__version__}")
if missing:
    raise SystemExit("  missing: " + ", ".join(missing))
print("  FaceDetectorYN and FaceRecognizerSF both present")
PY

mkdir -p models data faces

fetch_model() {
    local url="$1" dest="$2" expect="$3"
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" = "$expect" ]; then
        say "$(basename "$dest") already present, skipping"
        return
    fi
    say "Downloading $(basename "$dest")"
    curl -fsSL --retry 3 -o "$dest" "$url" || die "Failed to download $dest"
    local got
    got="$(stat -c%s "$dest")"
    [ "$got" = "$expect" ] || die "$(basename "$dest") is $got bytes, expected $expect"
}

fetch_lfs() {
    # The OpenCV model zoo stores its models in Git LFS, so the plain
    # raw.githubusercontent.com URL returns a 130-byte pointer file rather than
    # the real model. media.githubusercontent.com serves the actual binary.
    fetch_model "https://media.githubusercontent.com/media/$1" "$2" "$3"
}

fetch_plain() {
    # Ordinary git blobs, so raw.githubusercontent.com is correct here.
    fetch_model "https://raw.githubusercontent.com/$1" "$2" "$3"
}

fetch_lfs "opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    "models/yunet.onnx" 232589
fetch_lfs "opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
    "models/face_recognition_sface_2021dec.onnx" 38696353
# MiniFASNet V2 SE anti-spoofing (98.2% on its own benchmark). The float32 build
# is used rather than the 626KB quantised one because the quantised graph uses
# DynamicQuantizeLinear, which OpenCV's DNN module cannot parse.
fetch_plain "facenox/face-antispoof-onnx/main/models/best/98.20/best_model.onnx" \
    "models/antispoof_minifas.onnx" 1912594

say "Verifying the anti-spoof model loads through OpenCV"
"$VENV/bin/python" - <<'PY' || die "Anti-spoof model could not be loaded by OpenCV."
import cv2, numpy as np
net = cv2.dnn.readNetFromONNX("models/antispoof_minifas.onnx")
net.setInput(np.zeros((1, 3, 128, 128), np.float32))
out = net.forward()
if out.size != 2:
    raise SystemExit(f"  unexpected output size {out.size}, expected 2 classes")
print("  MiniFASNet loaded, 2 output classes")
PY

say "Done."
echo
echo "Next steps:"
echo "  1. Put photos in faces/<name>/*.jpg   (3-8 photos per person, varied angles)"
echo "  2. ./enroll.sh                        builds the face database"
echo "  3. ./run.sh                           starts recognition + web view"
echo
echo "Web view will be at http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080"
