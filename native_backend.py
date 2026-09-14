from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional


PLUGIN_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


class NativeBackendError(RuntimeError):
    pass


class _SharedResourceInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("slot_count", ctypes.c_uint32),
        ("color_format", ctypes.c_uint32),
        ("output_format", ctypes.c_uint32),
        ("motion_format", ctypes.c_uint32),
        ("depth_format", ctypes.c_uint32),
        ("color_allocation_size", ctypes.c_uint64),
        ("output_allocation_size", ctypes.c_uint64),
        ("motion_allocation_size", ctypes.c_uint64),
        ("depth_allocation_size", ctypes.c_uint64),
        ("adapter_luid", ctypes.c_uint8 * 8),
        ("adapter_node_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class NativeEnhancer:
    """Small ABI adapter for the TE D3D12/NGX bridge.

    The bridge owns all D3D and NGX objects. Python only passes one RGBA frame
    at a time, which keeps the ComfyUI side independent from NVIDIA headers.
    """

    def __init__(self, dll_path: str, width: int, height: int, settings: dict):
        path = Path(dll_path).expanduser()
        if not path.is_file():
            raise NativeBackendError(
                f"TE DLSS5 native backend was not found: {path}. "
                "Build/install the native bridge in the plugin directory or "
                "set TE_DLSS5_BACKEND_DLL for an external test build."
            )
        try:
            # Make the bundled NGX dependencies visible to the Windows loader
            # before the bridge (and its delay-loaded SDK import) is loaded.
            self._dll_dir_handle = None
            runtime = Path(settings.get("runtimeDir", ""))
            if os.name == "nt" and runtime.is_dir() and hasattr(os, "add_dll_directory"):
                self._dll_dir_handle = os.add_dll_directory(str(runtime))
            self._dll = ctypes.WinDLL(str(path))
        except (AttributeError, OSError) as exc:
            raise NativeBackendError(f"Could not load native backend {path}: {exc}") from exc

        self._create = self._bind("te_nr_create", ctypes.c_void_p, [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p])
        self._process = self._bind("te_nr_process_rgba", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32])
        self._process_guided = getattr(self._dll, "te_nr_process_rgba_guided", None)
        if self._process_guided is not None:
            self._process_guided.restype = ctypes.c_int
            self._process_guided.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ]
        self._submit = getattr(self._dll, "te_nr_submit_rgba_guided", None)
        self._collect = getattr(self._dll, "te_nr_collect_rgba", None)
        self._collect_nowait = getattr(self._dll, "te_nr_collect_rgba_nowait", None)
        self._flush = getattr(self._dll, "te_nr_flush", None)
        self._shared_query = getattr(self._dll, "te_nr_get_shared_resources", None)
        self._shared_query_v2 = getattr(self._dll, "te_nr_get_shared_resources_v2", None)
        self._shared_info_query = getattr(self._dll, "te_nr_get_shared_resource_info", None)
        self._submit_shared = getattr(self._dll, "te_nr_submit_shared_guided", None)
        self.shared_interop_ready = False
        self.shared_resource_info = None
        if self._shared_query is not None:
            self._shared_query.restype = ctypes.c_int
            self._shared_query.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
        if self._shared_info_query is not None:
            self._shared_info_query.restype = ctypes.c_int
            self._shared_info_query.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_SharedResourceInfo),
            ]
        if self._shared_query_v2 is not None:
            self._shared_query_v2.restype = ctypes.c_int
            self._shared_query_v2.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32,
                *(ctypes.POINTER(ctypes.c_void_p) for _ in range(6)),
            ]
        if self._submit_shared is not None:
            self._submit_shared.restype = ctypes.c_int
            self._submit_shared.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
            ]
        self.async_ready = all(
            fn is not None for fn in (self._submit, self._collect, self._collect_nowait, self._flush)
        )
        if self.async_ready:
            self._submit.restype = ctypes.c_int
            self._submit.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
            ]
            for fn in (self._collect, self._collect_nowait):
                fn.restype = ctypes.c_int
                fn.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32,
                ]
            self._flush.restype = ctypes.c_int
            self._flush.argtypes = [ctypes.c_void_p]
        elif any(fn is not None for fn in (self._submit, self._collect, self._collect_nowait, self._flush)):
            LOGGER.warning("[TE DLSS5] native async ABI is incomplete; using synchronous frame processing")
        self._reset = self._bind("te_nr_reset", None, [ctypes.c_void_p])
        self._destroy = self._bind("te_nr_destroy", None, [ctypes.c_void_p])
        self._info = getattr(self._dll, "te_nr_info", None)
        if self._info is not None:
            self._info.restype = ctypes.c_char_p
            self._info.argtypes = [ctypes.c_void_p]
        self._last_error = getattr(self._dll, "te_nr_last_error", None)
        if self._last_error is not None:
            self._last_error.restype = ctypes.c_char_p
            self._last_error.argtypes = []
        self._handle = self._create(width, height, json.dumps(settings).encode("utf-8"))
        if not self._handle:
            detail = ""
            if self._last_error is not None:
                try:
                    detail = (self._last_error() or b"").decode("utf-8", "replace")
                except Exception:
                    detail = ""
            suffix = f": {detail}" if detail else ""
            raise NativeBackendError(f"Native TE DLSS5 backend failed to create an enhancer{suffix}")
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 4
        self._frame_index = 0
        LOGGER.info("[TE DLSS5] DLSSNR native bridge ready: dll=%s, feature=18, size=%sx%s", path, width, height)
        LOGGER.info("[TE DLSS5] native frame queue: %s", "3-slot async" if self.async_ready else "synchronous fallback")
        if self._info is not None:
            try:
                details = (self._info(self._handle) or b"").decode("utf-8", "replace")
                if details:
                    LOGGER.info("[TE DLSS5] DLSSNR parameters accepted by native bridge: %s", details)
            except Exception:
                pass
        self._probe_shared_resources()

    def _bind(self, name, restype, argtypes):
        try:
            fn = getattr(self._dll, name)
        except AttributeError as exc:
            raise NativeBackendError(f"Backend export is missing: {name}") from exc
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    @staticmethod
    def _close_shared_handle(handle: ctypes.c_void_p) -> None:
        """Close a Windows HANDLE returned by the probe ABI."""
        value = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle or 0)
        if not value or os.name != "nt":
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle(ctypes.c_void_p(value))
        except (AttributeError, OSError):
            LOGGER.warning("[TE DLSS5] could not close a shared resource probe handle")

    def _probe_shared_resources(self) -> None:
        """Validate that the native bridge can export shared D3D12 handles.

        This first query is intentionally short-lived. The video scheduler
        later exports all three v2 slots and hands them to CUDA NVOF when both
        native DLLs expose the complete producer/consumer Fence ABI.
        """
        if self._shared_query is None:
            LOGGER.info("[TE DLSS5] shared D3D12 resources: ABI unavailable (CPU staging active)")
            return
        handles = [ctypes.c_void_p() for _ in range(5)]
        info = _SharedResourceInfo()
        info.struct_size = ctypes.sizeof(_SharedResourceInfo)
        info_result = (
            self._shared_info_query(self._handle, ctypes.byref(info))
            if self._shared_info_query is not None else 50
        )
        result = self._shared_query(
            self._handle,
            ctypes.c_uint32(0),
            *(ctypes.byref(handle) for handle in handles),
        )
        try:
            self.shared_interop_ready = result == 0 and all(handle.value for handle in handles)
            if self.shared_interop_ready:
                if info_result == 0:
                    luid = bytes(info.adapter_luid).hex()
                    self.shared_resource_info = {
                        "version": int(info.version),
                        "width": int(info.width),
                        "height": int(info.height),
                        "slot_count": int(info.slot_count),
                        "formats": (
                            int(info.color_format), int(info.output_format),
                            int(info.motion_format), int(info.depth_format),
                        ),
                        "allocation_sizes": (
                            int(info.color_allocation_size), int(info.output_allocation_size),
                            int(info.motion_allocation_size), int(info.depth_allocation_size),
                        ),
                        "adapter_luid": luid,
                        "adapter_node_mask": int(info.adapter_node_mask),
                    }
                else:
                    luid = "unknown"
                LOGGER.info(
                    "[TE DLSS5] shared D3D12 resources: export probe ready "
                    "(actual DLSSNR textures+fence, slots=%s, adapter_luid=%s; "
                    "awaiting CUDA NVOF attachment)",
                    info.slot_count if info_result == 0 else "unknown",
                    luid,
                )
            elif result == 50:  # ERROR_NOT_SUPPORTED: opt-in switch is off.
                LOGGER.info("[TE DLSS5] shared D3D12 resources: disabled (CPU staging active)")
            else:
                LOGGER.warning("[TE DLSS5] shared D3D12 resources: probe failed (%s)", result)
        finally:
            for handle in handles:
                self._close_shared_handle(handle)

    def export_shared_slot(self, slot_index: int) -> dict:
        """Return duplicated v2 handles for one slot; caller must close them."""
        if not self.shared_interop_ready or self._shared_query_v2 is None:
            raise NativeBackendError("Native bridge lacks shared D3D12 v2 ABI")
        handles = [ctypes.c_void_p() for _ in range(6)]
        result = self._shared_query_v2(
            self._handle, ctypes.c_uint32(slot_index),
            *(ctypes.byref(handle) for handle in handles),
        )
        if result != 0:
            for handle in handles:
                self._close_shared_handle(handle)
            raise NativeBackendError(
                f"Could not export shared D3D12 slot {slot_index} ({result})"
            )
        names = (
            "color", "output", "motion", "depth",
            "producer_fence", "consumer_fence",
        )
        return {name: handle for name, handle in zip(names, handles)}

    def close_shared_slot_handles(self, handles: dict) -> None:
        for handle in handles.values():
            self._close_shared_handle(handle)

    @property
    def shared_submit_ready(self) -> bool:
        return (
            self.shared_interop_ready and self._shared_query_v2 is not None and
            self._submit_shared is not None and self.shared_resource_info is not None
        )

    def submit_shared(self, slot_index: int, producer_fence_value: int,
                      depth: bytes | None = None) -> int:
        if not self.shared_submit_ready:
            raise NativeBackendError("Native bridge shared submission ABI is unavailable")
        expected = self.width * self.height * 4
        if depth is not None and len(depth) != expected:
            raise NativeBackendError(f"Unexpected depth guide size {len(depth)}; expected {expected}")
        depth_buf = ctypes.create_string_buffer(depth) if depth is not None else None
        token = ctypes.c_uint64(0)
        result = self._submit_shared(
            self._handle, ctypes.c_uint32(slot_index),
            ctypes.c_uint64(producer_fence_value), depth_buf,
            ctypes.c_uint32(len(depth) if depth is not None else 0),
            ctypes.byref(token),
        )
        if result != 0:
            raise NativeBackendError(f"Native shared DLSSNR submit failed ({result})")
        self._frame_index += 1
        if self._frame_index == 1 or self._frame_index % 30 == 0:
            LOGGER.info(
                "[TE DLSS5] frame=%d DLSSNR shared-texture submit ok slot=%d fence=%d",
                self._frame_index, slot_index, producer_fence_value,
            )
        return int(token.value)

    def process(self, frame: bytes, motion: bytes | None = None, depth: bytes | None = None) -> bytes:
        if len(frame) != self.frame_bytes:
            raise NativeBackendError(
                f"Unexpected RGBA frame size {len(frame)}; expected {self.frame_bytes}"
            )
        src = ctypes.create_string_buffer(frame)
        dst = ctypes.create_string_buffer(self.frame_bytes)
        if motion is not None or depth is not None:
            if self._process_guided is None:
                raise NativeBackendError("Native bridge lacks guided-frame support; rebuild the DLL")
            expected = self.width * self.height * 4
            if motion is not None and len(motion) != expected:
                raise NativeBackendError(f"Unexpected motion guide size {len(motion)}; expected {expected}")
            if depth is not None and len(depth) != expected:
                raise NativeBackendError(f"Unexpected depth guide size {len(depth)}; expected {expected}")
            motion_buf = ctypes.create_string_buffer(motion) if motion is not None else None
            depth_buf = ctypes.create_string_buffer(depth) if depth is not None else None
            result = self._process_guided(
                self._handle, src,
                motion_buf, depth_buf, dst, self.frame_bytes,
            )
        else:
            result = self._process(self._handle, src, dst, self.frame_bytes)
        if result != 0:
            raise NativeBackendError(f"Native TE DLSS5 frame processing failed ({result})")
        self._frame_index += 1
        if self._frame_index == 1 or self._frame_index % 30 == 0:
            LOGGER.info(
                "[TE DLSS5] frame=%d DLSSNR Feature 18 evaluate ok (guided=%s motion=%s depth=%s)",
                self._frame_index,
                motion is not None or depth is not None,
                motion is not None,
                depth is not None,
            )
        return dst.raw

    def submit(self, frame: bytes, motion: bytes | None = None, depth: bytes | None = None) -> int:
        """Queue one guided frame and return its native opaque token."""
        if not self.async_ready:
            raise NativeBackendError("Native bridge lacks asynchronous ABI; rebuild the DLL")
        if len(frame) != self.frame_bytes:
            raise NativeBackendError(
                f"Unexpected RGBA frame size {len(frame)}; expected {self.frame_bytes}"
            )
        expected = self.width * self.height * 4
        if motion is not None and len(motion) != expected:
            raise NativeBackendError(f"Unexpected motion guide size {len(motion)}; expected {expected}")
        if depth is not None and len(depth) != expected:
            raise NativeBackendError(f"Unexpected depth guide size {len(depth)}; expected {expected}")
        src = ctypes.create_string_buffer(frame)
        motion_buf = ctypes.create_string_buffer(motion) if motion is not None else None
        depth_buf = ctypes.create_string_buffer(depth) if depth is not None else None
        token = ctypes.c_uint64(0)
        result = self._submit(
            self._handle, src, motion_buf, depth_buf,
            ctypes.c_uint32(self.frame_bytes), ctypes.byref(token),
        )
        if result != 0:
            raise NativeBackendError(f"Native TE DLSS5 frame submit failed ({result})")
        self._frame_index += 1
        if self._frame_index == 1 or self._frame_index % 30 == 0:
            LOGGER.info("[TE DLSS5] frame=%d DLSSNR Feature 18 submit ok (async)", self._frame_index)
        return int(token.value)

    def collect(self, token: int, *, wait: bool = True) -> bytes:
        """Collect one queued frame, waiting unless ``wait`` is false."""
        if not self.async_ready:
            raise NativeBackendError("Native bridge lacks asynchronous ABI; rebuild the DLL")
        dst = ctypes.create_string_buffer(self.frame_bytes)
        fn = self._collect if wait else self._collect_nowait
        result = fn(self._handle, ctypes.c_uint64(token), dst, ctypes.c_uint32(self.frame_bytes))
        if result != 0:
            raise NativeBackendError(f"Native TE DLSS5 frame collect failed ({result})")
        return dst.raw

    def flush(self):
        """Wait for all native in-flight slots and release their tokens."""
        if self.async_ready and getattr(self, "_handle", None):
            result = self._flush(self._handle)
            if result != 0:
                raise NativeBackendError(f"Native TE DLSS5 queue flush failed ({result})")

    def reset(self):
        self._reset(self._handle)
        LOGGER.info("[TE DLSS5] DLSSNR temporal history reset")

    def close(self):
        if getattr(self, "_handle", None):
            if self.async_ready:
                try:
                    self.flush()
                except NativeBackendError as exc:
                    LOGGER.warning("[TE DLSS5] native queue flush during close failed: %s", exc)
            self._destroy(self._handle)
            self._handle = None
            LOGGER.info("[TE DLSS5] DLSSNR native bridge closed after %d frame(s)", getattr(self, "_frame_index", 0))
        handle = getattr(self, "_dll_dir_handle", None)
        if handle is not None:
            handle.close()
            self._dll_dir_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Creating the bridge is expensive: it opens a D3D12 device, initialises the NGX
# runtime and creates DLSSNR Feature 18. Measured on an RTX 5070 Ti Laptop this
# costs 2.2-2.6 s per call, which dominates every still-image prompt. Resetting
# is cheap and `te_nr_reset` restores exactly the state a freshly created
# instance starts in (verified byte-identical output across a new instance, a
# reset instance, and an instance that had already processed a different frame
# before being reset), so instances are kept alive and reused.
_ENHANCER_SLOTS = 4
_ENHANCER_CACHE: dict[tuple, NativeEnhancer] = {}
_ENHANCER_ORDER: list[tuple] = []
_ENHANCER_LOCK = threading.Lock()


def acquire_enhancer(dll_path: str, width: int, height: int, settings: dict,
                     slots: int = _ENHANCER_SLOTS) -> NativeEnhancer:
    """Return a reusable bridge instance for this configuration.

    The returned instance is owned by the cache and must NOT be closed by the
    caller. Callers are responsible for calling ``reset()`` before use, because
    a cached instance still carries the temporal history of whatever ran before
    it.
    """
    key = (str(Path(dll_path).expanduser()), int(width), int(height),
           json.dumps(settings, sort_keys=True))
    with _ENHANCER_LOCK:
        cached = _ENHANCER_CACHE.get(key)
        if cached is not None and getattr(cached, "_handle", None):
            _ENHANCER_ORDER.remove(key)
            _ENHANCER_ORDER.append(key)
            LOGGER.info("[TE DLSS5] reusing cached DLSSNR bridge for %sx%s", width, height)
            return cached
        # A cached entry without a handle was closed underneath us; rebuild it.
        _ENHANCER_CACHE.pop(key, None)
        if key in _ENHANCER_ORDER:
            _ENHANCER_ORDER.remove(key)
        while len(_ENHANCER_ORDER) >= max(1, int(slots)):
            evicted = _ENHANCER_ORDER.pop(0)
            victim = _ENHANCER_CACHE.pop(evicted, None)
            if victim is not None:
                victim.close()
        enhancer = NativeEnhancer(dll_path, width, height, settings)
        _ENHANCER_CACHE[key] = enhancer
        _ENHANCER_ORDER.append(key)
        return enhancer


def release_all() -> None:
    """Close every cached bridge and drop it. Safe to call at any time."""
    with _ENHANCER_LOCK:
        for enhancer in _ENHANCER_CACHE.values():
            enhancer.close()
        _ENHANCER_CACHE.clear()
        _ENHANCER_ORDER.clear()


def resolve_backend(explicit: str = "") -> Optional[str]:
    candidate = explicit or os.environ.get("TE_DLSS5_BACKEND_DLL", "")
    if candidate:
        return candidate
    # Self-contained test layout: a user-built bridge can live next to this
    # package, so no environment variable is required after compilation.
    candidates = (
        PLUGIN_ROOT / "te_dlss5_native.dll",
        PLUGIN_ROOT / "native" / "te_dlss5_native.dll",
        PLUGIN_ROOT / "native" / "build" / "Release" / "te_dlss5_native.dll",
        PLUGIN_ROOT / "native" / "build" / "te_dlss5_native.dll",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def resolve_runtime_dir(explicit: str = "") -> Path:
    candidate = explicit or os.environ.get("TE_DLSS5_RUNTIME_DIR", "")
    if candidate:
        return Path(candidate).expanduser()
    return PLUGIN_ROOT / "runtime" / "dlssnr"


def describe_backend(explicit: str = "") -> str:
    path = resolve_backend(explicit)
    runtime = resolve_runtime_dir()
    runtime_file = runtime / "nvngx_dlssnr.dll"
    if not path:
        return (
            "TE DLSS5 backend: bridge DLL not built; "
            f"NGX runtime={'found' if runtime_file.is_file() else 'missing'} at {runtime_file}"
        )
    target = Path(path).expanduser()
    if not target.is_file():
        return f"TE DLSS5 backend: missing file: {target}"
    if os.name != "nt":
        return f"TE DLSS5 backend: {target} (Windows DLL; current OS is unsupported)"
    guided = False
    async_abi = False
    load_error = ""
    dll_dir_handle = None
    try:
        runtime_dir = resolve_runtime_dir()
        if hasattr(os, "add_dll_directory") and runtime_dir.is_dir():
            dll_dir_handle = os.add_dll_directory(str(runtime_dir))
        loaded = ctypes.WinDLL(str(target))
        guided = getattr(loaded, "te_nr_process_rgba_guided", None) is not None
        async_abi = all(
            getattr(loaded, name, None) is not None
            for name in (
                "te_nr_submit_rgba_guided",
                "te_nr_collect_rgba",
                "te_nr_collect_rgba_nowait",
                "te_nr_flush",
            )
        )
    except (AttributeError, OSError) as exc:
        load_error = f"; load error={exc}"
    finally:
        if dll_dir_handle is not None:
            dll_dir_handle.close()
    if not guided:
        return (
            f"TE DLSS5 backend: {target}; guided ABI=missing; rebuild native DLL"
            f"; NGX runtime={'found' if runtime_file.is_file() else 'missing'} at {runtime_file}"
            f"{load_error}"
        )
    return (
        f"TE DLSS5 backend: configured: {target}; guided ABI=ready; "
        f"async ABI={'ready' if async_abi else 'missing (sync fallback)'}; "
        f"NGX runtime={'found' if runtime_file.is_file() else 'missing'} at {runtime_file}"
    )
