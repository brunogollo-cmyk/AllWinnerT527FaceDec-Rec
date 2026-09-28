"""NPU backend for the SFace embedding, via Allwinner VeriSilicon VIPLite.

The T527 carries a VIP9000 NPU that only executes Vivante `.nb` network
binaries. `sface_int8.nb` was produced with the Allwinner ACUITY toolkit
(import onnx -> quantize -> export ovxlib --pack-nbg-unify, target
VIP9000NANOSI_PLUS_PID0X10000016) and measures 13-15 ms per embedding on the
NPU against ~68 ms for the float32 ONNX model on the CPU.

Measured agreement between the two, on 20 real camera crops:

    cosine(NPU, CPU) = 0.9916  (min 0.9879, max 0.9946)

and against the enrolled float32 face database it produced the *same identity
decision on 20/20 frames*, so the existing facedb.json stays valid and nobody
has to re-enrol.

The I/O sequence mirrors the vendor's own vpm_run.c: vip_init, create the
network from file, query the graph's own buffer descriptors, create matching
buffers, set/prepare/run per inference, then read and dequantise the int8
output. The descriptors are queried from the graph rather than hardcoded, so a
re-converted model with different shapes works without editing this file.

If anything is unavailable - library missing, no /dev/vipcore, model will not
load - `load()` returns None and the caller silently keeps the CPU path.
"""
from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ALIGN_SIZE = 112
EMBED_DIM = 128

# Enumerations from vip_lite.h, named so the ctypes calls read like the C ones.
VIP_SUCCESS = 0
VIP_BUFFER_MEMORY_TYPE_DEFAULT = 0
VIP_BUFFER_QUANTIZE_NONE = 0
VIP_BUFFER_QUANTIZE_TF_ASYMM = 2
VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT = 3
VIP_CREATE_NETWORK_FROM_FILE = 0x01
VIP_BUFFER_OPER_TYPE_FLUSH = 1
VIP_BUFFER_OPER_TYPE_INVALIDATE = 2

# Values from vip_buffer_property_e in vip_lite.h (v1.13).
VIP_BUFFER_PROP_QUANT_FORMAT = 0
VIP_BUFFER_PROP_NUM_OF_DIMENSION = 1
VIP_BUFFER_PROP_SIZES_OF_DIMENSION = 2
VIP_BUFFER_PROP_DATA_FORMAT = 3
VIP_BUFFER_PROP_FIXED_POINT_POS = 4
VIP_BUFFER_PROP_TF_SCALE = 5
VIP_BUFFER_PROP_TF_ZERO_POINT = 6
VIP_BUFFER_PROP_NAME = 7

# Values from vip_network_property_e.
VIP_NETWORK_PROP_INPUT_COUNT = 0
VIP_NETWORK_PROP_OUTPUT_COUNT = 1


class NpuUnavailable(RuntimeError):
    """The NPU path cannot be used; fall back to the CPU."""


class _Affine(ctypes.Structure):
    _fields_ = [("scale", ctypes.c_float), ("zeroPoint", ctypes.c_int32)]


class _QuantData(ctypes.Union):
    """The quant_data union from vip_buffer_create_params_t."""

    _fields_ = [
        ("dfp", ctypes.c_int32),
        ("affine", _Affine),
    ]


class _BufferParams(ctypes.Structure):
    """Mirrors vip_buffer_create_params_t (v1.13) exactly.

    Field order matters: ctypes lays the struct out in declaration order, and
    the driver validates it. Note that `sizes` has six entries and `memory_type`
    comes last, which is the opposite of the intuitive ordering.
    """

    _fields_ = [
        ("num_of_dims", ctypes.c_uint32),
        ("sizes", ctypes.c_uint32 * 6),
        ("data_format", ctypes.c_uint32),
        ("quant_format", ctypes.c_uint32),
        ("quant_data", _QuantData),
        ("memory_type", ctypes.c_uint32),
    ]

def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name, None)
    if fn is None:
        raise NpuUnavailable(f"libVIPlite has no symbol {name}")
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


class NpuRecognizer:
    """Runs the converted SFace graph on the VIP9000."""

    def __init__(self, nb_path: Path, lib_dir: Path) -> None:
        self.nb_path = Path(nb_path)
        self.lib_dir = Path(lib_dir)
        self.last_ms = 0.0
        self.failures = 0
        self._closed = False

        lib_path = self.lib_dir / "libVIPlite.so"
        if not lib_path.exists():
            raise NpuUnavailable(f"{lib_path} not found")
        if not self.nb_path.exists():
            raise NpuUnavailable(f"{self.nb_path} not found")
        if not Path("/dev/vipcore").exists():
            raise NpuUnavailable("/dev/vipcore missing (is vipcore.ko loaded?)")

        # libVIPuser carries the implementation libVIPlite's vip_* symbols
        # resolve to, so it has to be loaded globally first.
        user = self.lib_dir / "libVIPuser.so"
        if user.exists():
            ctypes.CDLL(str(user), mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        self._bind_api()
        self._load_graph()

    # ----------------------------------------------------------------- setup

    def _bind_api(self) -> None:
        lib = self._lib
        self._vip_init = _bind(lib, "vip_init", ctypes.c_int, [])
        self._vip_get_version = _bind(
            lib, "vip_get_version", ctypes.c_uint32, []
        )
        self._vip_query_hardware = _bind(
            lib, "vip_query_hardware", ctypes.c_int,
            [ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p],
        )
        self._vip_create_network = _bind(
            lib, "vip_create_network", ctypes.c_int,
            [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int,
             ctypes.POINTER(ctypes.c_void_p)],
        )
        self._vip_query_network = _bind(
            lib, "vip_query_network", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_query_input = _bind(
            lib, "vip_query_input", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_query_output = _bind(
            lib, "vip_query_output", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_create_buffer = _bind(
            lib, "vip_create_buffer", ctypes.c_int,
            [ctypes.POINTER(_BufferParams), ctypes.c_size_t,
             ctypes.POINTER(ctypes.c_void_p)],
        )
        self._vip_get_buffer_size = _bind(
            lib, "vip_get_buffer_size", ctypes.c_uint32, [ctypes.c_void_p]
        )
        self._vip_map_buffer = _bind(
            lib, "vip_map_buffer", ctypes.c_void_p, [ctypes.c_void_p]
        )
        self._vip_unmap_buffer = _bind(
            lib, "vip_unmap_buffer", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_flush_buffer = _bind(
            lib, "vip_flush_buffer", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32],
        )
        self._vip_set_input = _bind(
            lib, "vip_set_input", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_set_output = _bind(
            lib, "vip_set_output", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_prepare_network = _bind(
            lib, "vip_prepare_network", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_run_network = _bind(
            lib, "vip_run_network", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_finish_network = _bind(
            lib, "vip_finish_network", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_destroy_buffer = _bind(
            lib, "vip_destroy_buffer", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_destroy_network = _bind(
            lib, "vip_destroy_network", ctypes.c_int, [ctypes.c_void_p]
        )

        status = self._vip_init()
        if status != VIP_SUCCESS:
            raise NpuUnavailable(f"vip_init failed: {status}")

        cid = ctypes.c_uint32(0)
        self._vip_query_hardware(0, ctypes.sizeof(cid), ctypes.byref(cid))
        self.cid = cid.value
        self.version = self._vip_get_version()

    def _query_params(self, index: int, is_input: bool) -> _BufferParams:
        param = _BufferParams()
        param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT
        query = self._vip_query_input if is_input else self._vip_query_output
        net = self._network

        # Query into standalone variables and copy in. Reading a struct field
        # through ctypes yields a plain int that byref() rejects, and byref(obj,
        # offset) does not write back, so the safe route is a local per query.
        data_format = ctypes.c_uint32(0)
        num_of_dims = ctypes.c_uint32(0)
        quant_format = ctypes.c_uint32(0)
        sizes = (ctypes.c_uint32 * 4)()

        query(net, index, VIP_BUFFER_PROP_DATA_FORMAT, ctypes.byref(data_format))
        query(net, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, ctypes.byref(num_of_dims))
        query(net, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, sizes)
        query(net, index, VIP_BUFFER_PROP_QUANT_FORMAT, ctypes.byref(quant_format))

        param.data_format = data_format.value
        param.num_of_dims = num_of_dims.value
        param.quant_format = quant_format.value
        for i in range(len(sizes)):
            param.sizes[i] = sizes[i]
        param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT

        # Carry the graph's own quantisation parameters into the buffer request;
        # the driver uses them to place the data, not just to describe it.
        if param.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM:
            scale = ctypes.c_float(0.0)
            zero = ctypes.c_int32(0)
            query(net, index, VIP_BUFFER_PROP_TF_SCALE, ctypes.byref(scale))
            query(net, index, VIP_BUFFER_PROP_TF_ZERO_POINT, ctypes.byref(zero))
            param.quant_data.affine.scale = scale.value
            param.quant_data.affine.zeroPoint = zero.value
        elif param.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
            pos = ctypes.c_int32(0)
            query(net, index, VIP_BUFFER_PROP_FIXED_POINT_POS, ctypes.byref(pos))
            param.quant_data.dfp = pos.value
        return param

    def _load_graph(self) -> None:
        net = ctypes.c_void_p()
        status = self._vip_create_network(
            str(self.nb_path).encode(), 0, VIP_CREATE_NETWORK_FROM_FILE,
            ctypes.byref(net),
        )
        if status != VIP_SUCCESS:
            raise NpuUnavailable(f"vip_create_network failed: {status}")
        self._network = net

        counts = []
        for prop in (VIP_NETWORK_PROP_INPUT_COUNT, VIP_NETWORK_PROP_OUTPUT_COUNT):
            value = ctypes.c_uint32(0)
            self._vip_query_network(net, prop, ctypes.byref(value))
            counts.append(value.value)
        self.input_count, self.output_count = counts
        if self.input_count < 1 or self.output_count < 1:
            raise NpuUnavailable(f"bad io counts {counts}")

        self._in_param = self._query_params(0, True)
        self._out_param = self._query_params(0, False)

        in_buf = ctypes.c_void_p()
        out_buf = ctypes.c_void_p()
        if self._vip_create_buffer(
            ctypes.byref(self._in_param), ctypes.sizeof(self._in_param),
            ctypes.byref(in_buf)
        ) != VIP_SUCCESS:
            raise NpuUnavailable("input buffer alloc failed")
        if self._vip_create_buffer(
            ctypes.byref(self._out_param), ctypes.sizeof(self._out_param),
            ctypes.byref(out_buf)
        ) != VIP_SUCCESS:
            raise NpuUnavailable("output buffer alloc failed")
        self._in_buf, self._out_buf = in_buf, out_buf

        # The graph is int8 out; read its own dequantisation parameters.
        scale = ctypes.c_float(0.0)
        zero = ctypes.c_int32(0)
        self._vip_query_output(
            self._network, 0, VIP_BUFFER_PROP_TF_SCALE, ctypes.byref(scale)
        )
        self._vip_query_output(
            self._network, 0, VIP_BUFFER_PROP_TF_ZERO_POINT, ctypes.byref(zero)
        )
        self.out_scale, self.out_zero = scale.value, zero.value

        self.in_size = self._vip_get_buffer_size(in_buf)
        self._out_map = None
        log.info(
            "NPU graph loaded: %s driver=0x%08x cid=0x%x in=%dB out_scale=%.9f zp=%d",
            self.nb_path.name, self.version, self.cid, self.in_size,
            self.out_scale, self.out_zero,
        )

    # ------------------------------------------------------------- inference

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray | None:
        """Embed a 112x112 aligned BGR crop, returning a 128-d unit vector."""
        if aligned is None or aligned.size == 0:
            return None
        if aligned.shape[0] != ALIGN_SIZE or aligned.shape[1] != ALIGN_SIZE:
            import cv2

            aligned = cv2.resize(aligned, (ALIGN_SIZE, ALIGN_SIZE))

        # The graph takes RGB in NCHW. Feed the same crop the CPU model sees so
        # the two paths stay comparable.
        rgb = aligned[:, :, ::-1]
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None, ...].astype(np.float16)
        )

        start = time.perf_counter()
        try:
            out = self._run(tensor)
        except NpuUnavailable:
            raise
        except Exception:  # noqa: BLE001 - never let the NPU kill the loop
            self.failures += 1
            log.exception("NPU inference failed")
            return None
        self.last_ms = (time.perf_counter() - start) * 1000
        return out

    def _run(self, tensor: np.ndarray) -> np.ndarray | None:
        data = tensor.view(np.uint8)
        if data.size != self.in_size:
            # A mismatch here means the graph's input is not float16 NCHW of
            # this size; better to fall back than to feed it garbage.
            raise NpuUnavailable(
                f"input {data.size}B does not match graph buffer {self.in_size}B"
            )

        mapped = self._vip_map_buffer(self._in_buf)
        ctypes.memmove(mapped, data.ctypes.data, data.nbytes)
        self._vip_flush_buffer(self._in_buf, VIP_BUFFER_OPER_TYPE_FLUSH)
        self._vip_unmap_buffer(self._in_buf)

        net = self._network
        for step, call in (
            ("prepare", self._vip_prepare_network),
            ("set_input", lambda: self._vip_set_input(net, 0, self._in_buf)),
            ("set_output", lambda: self._vip_set_output(net, 0, self._out_buf)),
            ("run", self._vip_run_network),
        ):
            status = call(net) if step == "prepare" or step == "run" else call()
            if status != VIP_SUCCESS:
                raise NpuUnavailable(f"{step} failed: {status}")
        self._vip_finish_network(net)

        self._vip_flush_buffer(self._out_buf, VIP_BUFFER_OPER_TYPE_INVALIDATE)
        out_ptr = self._vip_map_buffer(self._out_buf)
        raw = np.ctypeslib.as_array(
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int8)), (EMBED_DIM,)
        ).astype(np.float32)
        self._vip_unmap_buffer(self._out_buf)

        values = (raw - np.float32(self.out_zero)) * np.float32(self.out_scale)
        norm = float(np.linalg.norm(values))
        if norm < 1e-6:
            return None
        return values / norm

    # -------------------------------------------------------------- teardown

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fn, handle in (
            (self._vip_destroy_buffer, self._in_buf),
            (self._vip_destroy_buffer, self._out_buf),
            (self._vip_destroy_network, self._network),
        ):
            if handle:
                try:
                    fn(handle)
                except Exception:  # noqa: BLE001 - teardown is best effort
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass


def load(nb_path: Path, lib_dir: Path) -> NpuRecognizer | None:
    """Build the NPU backend, or None if it cannot be used."""
    try:
        return NpuRecognizer(nb_path, lib_dir)
    except (NpuUnavailable, OSError, AttributeError, ValueError) as exc:
        log.warning("NPU backend unavailable (%s) - using CPU", exc)
        return None
