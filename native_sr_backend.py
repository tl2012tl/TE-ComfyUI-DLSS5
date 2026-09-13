from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path


LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


class NativeSRBackendError(RuntimeError):
    pass


class NativeSuperResolution:
    """ABI adapter for the separate D3D12/NGX DLSS Super Resolution bridge."""

    def __init__(self, dll_path: str, input_width: int, input_height: int,
                 output_width: int, output_height: int, runtime_dir: str):
        path = Path(dll_path).expanduser()
        if not path.is_file():
            raise NativeSRBackendError(
                f"Native DLSS SR bridge was not found: {path}. Rebuild native DLLs."
            )
        self._dll_dir_handle = None
        try:
            runtime = Path(runtime_dir)
            if os.name == "nt" and runtime.is_dir() and hasattr(os, "add_dll_directory"):
                self._dll_dir_handle = os.add_dll_directory(str(runtime))
            self._dll = ctypes.WinDLL(str(path))
        except (AttributeError, OSError) as exc:
            raise NativeSRBackendError(f"Could not load native DLSS SR bridge {path}: {exc}") from exc
        self._create = self._bind(
            "te_sr_create", ctypes.c_void_p,
            [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p],
        )
        self._process = self._bind(
            "te_sr_process_rgba", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32],
        )
        self._destroy = self._bind("te_sr_destroy", None, [ctypes.c_void_p])
        self._last_error = getattr(self._dll, "te_sr_last_error", None)
        if self._last_error is not None:
            self._last_error.restype = ctypes.c_char_p
            self._last_error.argtypes = []
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.input_bytes = self.input_width * self.input_height * 4
        self.output_bytes = self.output_width * self.output_height * 4
        runtime_bytes = str(runtime_dir).encode("utf-8")
        self._handle = self._create(
            self.input_width, self.input_height,
            self.output_width, self.output_height, runtime_bytes,
        )
        if not self._handle:
            detail = ""
            if self._last_error is not None:
                detail = (self._last_error() or b"").decode("utf-8", "replace")
            if self._dll_dir_handle is not None:
                self._dll_dir_handle.close()
                self._dll_dir_handle = None
            raise NativeSRBackendError(
                "Native DLSS SR backend failed to create an upscaler"
                + (f": {detail}" if detail else "")
            )
        LOGGER.info(
            "[TE DLSS5] native DLSS SR ready: dll=%s, feature=SuperSampling, size=%sx%s -> %sx%s",
            path, self.input_width, self.input_height, self.output_width, self.output_height,
        )
        self._frame_index = 0

    def _bind(self, name, restype, argtypes):
        try:
            function = getattr(self._dll, name)
        except AttributeError as exc:
            raise NativeSRBackendError(f"DLSS SR bridge export is missing: {name}") from exc
        function.restype = restype
        function.argtypes = argtypes
        return function

    def process(self, frame: bytes) -> bytes:
        if len(frame) != self.input_bytes:
            raise NativeSRBackendError(
                f"Unexpected source RGBA frame size {len(frame)}; expected {self.input_bytes}"
            )
        source = ctypes.create_string_buffer(frame)
        target = ctypes.create_string_buffer(self.output_bytes)
        result = self._process(
            self._handle, source, target,
            ctypes.c_uint32(self.input_bytes), ctypes.c_uint32(self.output_bytes),
        )
        if result != 0:
            detail = (self._last_error() or b"").decode("utf-8", "replace") if self._last_error else ""
            raise NativeSRBackendError(f"Native DLSS SR frame processing failed ({result})" + (f": {detail}" if detail else ""))
        self._frame_index += 1
        if self._frame_index == 1 or self._frame_index % 30 == 0:
            LOGGER.info("[TE DLSS5] frame=%d native DLSS SR evaluate ok", self._frame_index)
        return target.raw

    def close(self):
        if getattr(self, "_handle", None):
            self._destroy(self._handle)
            self._handle = None
        if self._dll_dir_handle is not None:
            self._dll_dir_handle.close()
            self._dll_dir_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def resolve_sr_backend(explicit: str = "") -> str | None:
    candidate = explicit or os.environ.get("TE_DLSS5_SR_BACKEND_DLL", "")
    if candidate:
        return candidate
    root = Path(__file__).resolve().parent
    for path in (root / "te_dlss_sr_native.dll", root / "native" / "build" / "Release" / "te_dlss_sr_native.dll"):
        if path.is_file():
            return str(path)
    return None
