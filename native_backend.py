from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional


PLUGIN_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


_NGX_CODE_NOTES = {
    "0xBAD00001": "NGX runtime rejected this feature on the current GPU/driver/runtime combination (feature unsupported)",
    "0xBAD00002": "NGX runtime rejected the caller/application identity (caller validation failed)",
}


def explain_ngx_codes(text: str) -> str:
    """Append human-readable notes when an NGX status code appears in text."""
    lowered = (text or "").lower()
    notes = [
        f"{code}: {note}"
        for code, note in _NGX_CODE_NOTES.items()
        if code[2:].lower() in lowered
    ]
    if not notes:
        return text
    return f"{text} [{' ; '.join(notes)}]"


class NativeBackendError(RuntimeError):
    pass


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
        self._reset = self._bind("te_nr_reset", None, [ctypes.c_void_p])
        self._destroy = self._bind("te_nr_destroy", None, [ctypes.c_void_p])
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
            suffix = ""
            detail = explain_ngx_codes(detail)
            if detail:
                suffix = f": {detail}"
            raise NativeBackendError(f"Native TE DLSS5 backend failed to create an enhancer{suffix}")
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 4
        self._frame_index = 0
        LOGGER.info("[TE DLSS5] DLSSNR native bridge ready: dll=%s, feature=18, size=%sx%s", path, width, height)

    def _bind(self, name, restype, argtypes):
        try:
            fn = getattr(self._dll, name)
        except AttributeError as exc:
            raise NativeBackendError(f"Backend export is missing: {name}") from exc
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

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

    def reset(self):
        self._reset(self._handle)
        LOGGER.info("[TE DLSS5] DLSSNR temporal history reset")

    def close(self):
        if getattr(self, "_handle", None):
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
    load_error = ""
    dll_dir_handle = None
    try:
        runtime_dir = resolve_runtime_dir()
        if hasattr(os, "add_dll_directory") and runtime_dir.is_dir():
            dll_dir_handle = os.add_dll_directory(str(runtime_dir))
        loaded = ctypes.WinDLL(str(target))
        guided = getattr(loaded, "te_nr_process_rgba_guided", None) is not None
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
        f"NGX runtime={'found' if runtime_file.is_file() else 'missing'} at {runtime_file}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_section(lines: list[str]) -> None:
    import sys

    torch_note = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = ", ".join(
                f"{i}:{torch.cuda.get_device_name(i)}" for i in range(count)
            )
            torch_note = f"torch {torch.__version__} (cuda {torch.version.cuda}); {count} CUDA device(s): {names}"
        else:
            torch_note = f"torch {torch.__version__} installed but CUDA is NOT available"
    except Exception as exc:
        torch_note = f"torch import failed: {exc}"
    lines.append(f"python: {sys.version.split()[0]} | {torch_note}")
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", "replace")
        lines.append(f"driver: {driver}")
    except Exception:
        lines.append("driver: unknown (pynvml unavailable)")


def _ngx_section(lines: list[str]) -> None:
    runtime_dir = resolve_runtime_dir()
    entries = sorted(runtime_dir.glob("*.dll")) if runtime_dir.is_dir() else []
    if not entries:
        lines.append(f"NGX runtime: none found at {runtime_dir}")
        return
    for entry in entries:
        lines.append(f"  {entry.name}: {entry.stat().st_size:,} bytes")
    dlssnr = runtime_dir / "nvngx_dlssnr.dll"
    if dlssnr.is_file():
        lines.append(f"  nvngx_dlssnr.dll sha256={_sha256_file(dlssnr)}")


def _nvof_section(lines: list[str]) -> None:
    from .frame_guidance import cuda_bin_candidates, nvof_dll_candidates

    path = next((item for item in nvof_dll_candidates() if item.is_file()), None)
    if path is None:
        lines.append("NVOF bridge DLL: not found (guidance modes nvof/nvof_depth unavailable)")
    else:
        exports = "unknown"
        try:
            loaded = ctypes.WinDLL(str(path))
            exports = "ok" if hasattr(loaded, "te_nvof_create") else "missing te_nvof_create"
        except OSError as exc:
            exports = f"load failed: {exc}"
        lines.append(
            f"NVOF bridge DLL: {path} ({path.stat().st_size:,} bytes, exports={exports})"
        )
        if exports == "ok":
            lines.append(f"  sha256={_sha256_file(path)}")
    try:
        ctypes.WinDLL("nvofapi64.dll")
        lines.append("nvofapi64.dll (driver): visible")
    except OSError:
        lines.append("nvofapi64.dll (driver): NOT loadable; temporal guidance needs a recent NVIDIA driver")
    bins = cuda_bin_candidates()
    lines.append(f"CUDA toolkit bin: {bins[0] if bins else 'not found (bridge may still load its deps from the driver)'}")


def _deep_nvof_test(lines: list[str]) -> None:
    if os.environ.get("TE_DLSS5_RUNTIME_INFO_DEEP", "").strip().lower() not in {"1", "true", "yes", "on"}:
        lines.append("deep NVOF test: skipped (set TE_DLSS5_RUNTIME_INFO_DEEP=1 to create NVOF, process one frame, and verify torch CUDA survives)")
        return
    try:
        from .frame_guidance import _CudaNvof

        size = 64
        nvof = _CudaNvof(size, size)
        try:
            nvof.next(os.urandom(size * size * 4))
        finally:
            nvof.close()
        import torch

        if torch.cuda.is_available():
            (torch.zeros(8, device="cuda") + 1).sum().item()
        lines.append("deep NVOF test: ok (created, processed one frame, closed; torch CUDA intact)")
    except Exception as exc:
        lines.append(f"deep NVOF test: FAILED: {type(exc).__name__}: {exc}")


def _depth_section(lines: list[str]) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            checkpoint = (
                PLUGIN_ROOT / "runtime" / "frame_guidance" / "DepthAnythingV2"
                / "model" / "depth_anything_v2_vits.pth"
            )
            state = "found" if checkpoint.is_file() else f"missing ({checkpoint})"
            lines.append(f"depth (torch): CUDA available ({device}); checkpoint {state}")
        else:
            lines.append("depth (torch): CUDA unavailable; ONNX fallback would be used")
    except Exception as exc:
        lines.append(f"depth (torch): check failed: {exc}")
    try:
        import onnxruntime as ort

        lines.append(f"depth (onnxruntime): providers={ort.get_available_providers()}")
    except Exception:
        lines.append("depth (onnxruntime): not installed")


def _media_section(lines: list[str]) -> None:
    try:
        from .video_pipeline import _encoder_args, _resolve_media_tool

        ffmpeg = _resolve_media_tool("ffmpeg")
        encoder = _encoder_args(ffmpeg)
        lines.append(f"ffmpeg: {ffmpeg} | encoder: {encoder[1]}")
    except Exception as exc:
        lines.append(f"ffmpeg: not usable ({exc}); video output mode requires it, frames mode does not")


def build_runtime_report() -> str:
    """One-shot diagnostic text for the TE DLSS5 Runtime Info node."""
    lines: list[str] = ["TE DLSS5 Runtime Info", "== host =="]
    _host_section(lines)
    lines.append("== backend (DLSSNR D3D12/NGX) ==")
    lines.append(describe_backend())
    bridge = PLUGIN_ROOT / "te_dlss5_native.dll"
    if bridge.is_file():
        lines.append(f"bridge sha256={_sha256_file(bridge)}")
    lines.append("== NGX runtime ==")
    _ngx_section(lines)
    lines.append("== NVOF (CUDA optical flow) ==")
    _nvof_section(lines)
    _deep_nvof_test(lines)
    lines.append("== depth guidance ==")
    _depth_section(lines)
    lines.append("== media ==")
    _media_section(lines)
    return "\n".join(lines)
