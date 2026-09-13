"""Python adapter for the TE native DLSS Frame Generation bridge.

The bridge is deliberately a separate DLL from the working DLSSNR bridge.
It accepts two consecutive video frames through the normal ComfyUI batch
path and returns the generated intermediate frame without spawning a worker
process.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
from pathlib import Path

from .native_backend import PLUGIN_ROOT, resolve_runtime_dir

LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


class FrameGenerationError(RuntimeError):
    pass


def resolve_frame_generator(explicit: str = "") -> str | None:
    candidate = explicit or os.environ.get("TE_DLSS5_FG_DLL", "")
    if candidate:
        return str(Path(candidate).expanduser())
    for path in (
        PLUGIN_ROOT / "te_dlssg_native.dll",
        PLUGIN_ROOT / "native" / "te_dlssg_native.dll",
        PLUGIN_ROOT / "native" / "build_dlssg" / "Release" / "te_dlssg_native.dll",
    ):
        if path.is_file():
            return str(path)
    return None


class NativeFrameGenerator:
    """Own one native Feature-11 session for a fixed-size sequential stream."""

    def __init__(self, dll_path: str, width: int, height: int, settings: dict):
        path = Path(dll_path).expanduser()
        if not path.is_file():
            raise FrameGenerationError(f"TE DLSSG bridge was not found: {path}")
        self._dll_dir = None
        runtime = Path(settings.get("runtimeDir", ""))
        try:
            if os.name == "nt" and runtime.is_dir() and hasattr(os, "add_dll_directory"):
                self._dll_dir = os.add_dll_directory(str(runtime))
            self._dll = ctypes.WinDLL(str(path))
        except (AttributeError, OSError) as exc:
            raise FrameGenerationError(f"Could not load TE DLSSG bridge {path}: {exc}") from exc

        self._create = self._bind("te_fg_create", ctypes.c_void_p,
                                  [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p])
        self._process = self._bind(
            "te_fg_process_rgba_guided", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)],
        )
        self._reset = self._bind("te_fg_reset", None, [ctypes.c_void_p])
        self._destroy = self._bind("te_fg_destroy", None, [ctypes.c_void_p])
        self._info = getattr(self._dll, "te_fg_info", None)
        if self._info is not None:
            self._info.restype = ctypes.c_char_p
            self._info.argtypes = [ctypes.c_void_p]
        self._last_error = getattr(self._dll, "te_fg_last_error", None)
        if self._last_error is not None:
            self._last_error.restype = ctypes.c_char_p
            self._last_error.argtypes = []

        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * 4
        self.motion_bytes = self.width * self.height * 4
        self.depth_bytes = self.width * self.height * 4
        self._handle = self._create(
            self.width, self.height, json.dumps(settings).encode("utf-8")
        )
        if not self._handle:
            detail = ""
            if self._last_error is not None:
                detail = (self._last_error() or b"").decode("utf-8", "replace")
            self.close()
            raise FrameGenerationError(
                f"Native DLSSG backend failed to create a generator"
                f"{': ' + detail if detail else ''}"
            )
        details = ""
        if self._info is not None:
            details = (self._info(self._handle) or b"").decode("utf-8", "replace")
        LOGGER.info("[TE DLSS5] DLSSG native bridge ready: dll=%s%s", path,
                    ", " + details if details else "")

    def _bind(self, name, restype, argtypes):
        try:
            fn = getattr(self._dll, name)
        except AttributeError as exc:
            raise FrameGenerationError(f"DLSSG bridge export is missing: {name}") from exc
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def process(self, rgba: bytes, motion: bytes | None = None,
                depth: bytes | None = None) -> bytes | None:
        if len(rgba) != self.frame_bytes:
            raise FrameGenerationError(
                f"Unexpected RGBA size {len(rgba)}; expected {self.frame_bytes}"
            )
        if motion is not None and len(motion) != self.motion_bytes:
            raise FrameGenerationError("Unexpected motion guide size")
        if depth is not None and len(depth) != self.depth_bytes:
            raise FrameGenerationError("Unexpected depth guide size")
        src = ctypes.create_string_buffer(rgba)
        mv = ctypes.create_string_buffer(motion) if motion is not None else None
        dep = ctypes.create_string_buffer(depth) if depth is not None else None
        dst = ctypes.create_string_buffer(self.frame_bytes)
        generated = ctypes.c_uint32(0)
        status = self._process(
            self._handle, src, mv, dep, dst,
            ctypes.c_uint32(self.frame_bytes), ctypes.byref(generated),
        )
        if status != 0:
            detail = ""
            if self._last_error is not None:
                detail = (self._last_error() or b"").decode("utf-8", "replace")
            raise FrameGenerationError(
                f"DLSSG frame generation failed ({status})"
                f"{': ' + detail if detail else ''}"
            )
        return dst.raw if generated.value else None

    def reset(self):
        if self._handle:
            self._reset(self._handle)
            LOGGER.info("[TE DLSS5] DLSSG temporal history reset")

    def close(self):
        if getattr(self, "_handle", None):
            self._destroy(self._handle)
            self._handle = None
        if self._dll_dir is not None:
            self._dll_dir.close()
            self._dll_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

