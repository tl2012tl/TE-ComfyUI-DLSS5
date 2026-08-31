from __future__ import annotations

import json
import logging
import math
import numbers
import os
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from .native_backend import NativeEnhancer, NativeBackendError, resolve_backend, resolve_runtime_dir
from .frame_guidance import FrameGuidance, GuidanceError


LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


def _resolve_media_tool(name: str) -> str:
    """Resolve FFmpeg tools consistently with ComfyUI/VHS."""
    env_name = "TE_DLSS5_FFMPEG_PATH" if name == "ffmpeg" else "TE_DLSS5_FFPROBE_PATH"
    configured = os.environ.get(env_name, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        raise RuntimeError(f"{env_name} points to a missing file: {path}")

    if name == "ffmpeg":
        # VideoHelperSuite exposes the path it selected, including its
        # imageio-ffmpeg fallback, even when ffmpeg is absent from PATH.
        try:
            from videohelpersuite.utils import ffmpeg_path  # type: ignore
            if ffmpeg_path and Path(ffmpeg_path).is_file():
                return str(ffmpeg_path)
        except Exception:
            pass
        try:
            from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
            path = get_ffmpeg_exe()
            if path and Path(path).is_file():
                return str(path)
        except Exception:
            pass
    else:
        # Standard FFmpeg distributions ship ffprobe beside ffmpeg. Reuse a
        # configured ffmpeg location when only the probe tool is absent from
        # PATH.
        ffmpeg_configured = os.environ.get("TE_DLSS5_FFMPEG_PATH", "").strip()
        if ffmpeg_configured:
            sibling = Path(ffmpeg_configured).expanduser().with_name(
                "ffprobe.exe" if os.name == "nt" else "ffprobe"
            )
            if sibling.is_file():
                return str(sibling)

    path = shutil.which(name)
    if path:
        return path
    for candidate in (Path.cwd() / f"{name}.exe", Path.cwd() / name):
        if candidate.is_file():
            return str(candidate.resolve())
    hint = (
        "Install VideoHelperSuite/imageio-ffmpeg, add FFmpeg to PATH, or set "
        "TE_DLSS5_FFMPEG_PATH to the full path of ffmpeg.exe"
        if name == "ffmpeg" else
        "Install ffprobe.exe with FFmpeg or set TE_DLSS5_FFPROBE_PATH to its full path"
    )
    raise RuntimeError(f"{name} was not found. {hint}")


@dataclass
class VideoJob:
    source: Path | None
    output_prefix: str
    style: str
    intensity: float
    local_tone: float
    local_structure: float
    guidance: str
    depth_interval: int
    output_fps: float
    backend_dll: str
    runtime_dir: str
    overwrite: bool


def _run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    # Always decode child diagnostics as text with replacement. A malformed
    # byte from an external codec must never be allowed to enter ComfyUI's
    # log stream as a raw frame dump.
    return subprocess.run(
        command,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )


def _ffprobe(source: Path) -> dict:
    try:
        ffprobe = _resolve_media_tool("ffprobe")
        result = _run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,pix_fmt,color_space,color_transfer,color_primaries,color_range",
                "-of", "json", str(source),
            ],
            capture=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe could not be started. Set TE_DLSS5_FFPROBE_PATH to the full path of ffprobe.exe"
        ) from exc
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {source}")
    stream = streams[0]

    def ratio(value: str) -> float:
        if not value or value == "0/0":
            return 0.0
        num, den = value.split("/", 1)
        return float(num) / float(den)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": ratio(stream.get("avg_frame_rate", "0/0")),
        "pix_fmt": stream.get("pix_fmt", ""),
        "color_space": stream.get("color_space", ""),
        "color_transfer": stream.get("color_transfer", ""),
        "color_primaries": stream.get("color_primaries", ""),
        "color_range": stream.get("color_range", ""),
    }


def _encoder_args(ffmpeg: str | None = None) -> list[str]:
    """Select NVENC when present, with an explicit CPU fallback."""
    requested = os.environ.get("TE_DLSS5_ENCODER", "auto").strip().lower()
    if requested not in {"auto", "nvenc", "x264", "libx264"}:
        raise RuntimeError("TE_DLSS5_ENCODER must be auto, nvenc, or x264")
    nvenc_available = False
    ffmpeg = ffmpeg or _resolve_media_tool("ffmpeg")
    if requested in {"auto", "nvenc"}:
        try:
            probe = _run([ffmpeg, "-hide_banner", "-encoders"], capture=True)
            nvenc_available = "h264_nvenc" in ((probe.stdout or "") + (probe.stderr or ""))
        except (FileNotFoundError, subprocess.CalledProcessError):
            nvenc_available = False
    if requested == "nvenc" and not nvenc_available:
        raise RuntimeError("TE_DLSS5_ENCODER=nvenc requested but ffmpeg h264_nvenc is unavailable")
    if nvenc_available:
        # p5/hq is a good quality-speed point for a GPU encode. QP keeps the
        # output behavior close to the previous CRF 16 x264 path.
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "constqp", "-qp", "16"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "16"]


def _color_args(metadata: dict | None) -> list[str]:
    """Preserve known source transfer metadata instead of silently tagging BT.601."""
    metadata = metadata or {}
    args = []
    mapping = (
        ("color_space", "-colorspace"),
        ("color_transfer", "-color_trc"),
        ("color_primaries", "-color_primaries"),
        ("color_range", "-color_range"),
    )
    for key, option in mapping:
        value = metadata.get(key)
        if value and value not in {"unknown", "unspecified"}:
            args.extend([option, str(value)])
    return args


def _output_pix_fmt() -> str:
    value = os.environ.get("TE_DLSS5_OUTPUT_PIX_FMT", "yuv420p").strip().lower()
    allowed = {"yuv420p", "yuv444p", "yuv420p10le", "p010le"}
    if value not in allowed:
        raise RuntimeError(
            "TE_DLSS5_OUTPUT_PIX_FMT must be yuv420p, yuv444p, yuv420p10le, or p010le"
        )
    return value


def _output_path(job: VideoJob) -> Path:
    base = Path(job.output_prefix).expanduser()
    if not base.is_absolute():
        try:
            import folder_paths  # type: ignore

            output_root = Path(folder_paths.get_output_directory())
        except Exception:
            # Makes the module usable in a small standalone smoke test too.
            output_root = Path.cwd() / "output"
        base = output_root / base
    if base.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm"}:
        base = base.with_suffix(".mp4")
    base.parent.mkdir(parents=True, exist_ok=True)
    if job.overwrite:
        return base
    candidate = base
    index = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.stem}_{index:03d}{base.suffix}")
        index += 1
    return candidate


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _stop_process(process) -> None:
    """Close pipes and fully reap a child before its temporary files are removed."""
    if process is None:
        return
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        # The handle may already have been reaped by Windows after kill().
        pass


def _settings(job: VideoJob) -> dict:
    return {
        "style": job.style,
        # DLSSNR's runtime accepts the same -1..2 tuning range as the
        # reference UI; keeping the clamp here prevents malformed workflows
        # from sending arbitrary values into the native bridge.
        "intensity": max(-1.0, min(2.0, job.intensity)),
        "localToneStrength": max(-1.0, min(2.0, job.local_tone)),
        "localStructureStrength": max(-1.0, min(2.0, job.local_structure)),
        "guidance": job.guidance,
        "depthInterval": max(1, int(job.depth_interval)),
        "runtimeDir": str(resolve_runtime_dir(job.runtime_dir)),
    }


def _write_audio_wav(audio, path: Path) -> bool:
    """Write ComfyUI's AUDIO dict to a temporary PCM WAV for muxing."""
    try:
        import numpy as np

        waveform = audio["waveform"].detach().float().cpu().numpy()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim != 2:
            return False
        channels, samples = waveform.shape
        pcm = np.clip(waveform.T, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(pcm.tobytes())
        return True
    except Exception:
        return False


def _safe_positive_float(value) -> float:
    """Convert only scalar FPS-like values; never parse frame/binary payloads."""
    if isinstance(value, (bytes, bytearray, memoryview, bool)):
        return 0.0
    if isinstance(value, numbers.Real):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or len(text) > 32:
            return 0.0
        try:
            result = float(text)
        except (TypeError, ValueError, OverflowError):
            return 0.0
    else:
        # Tensors, lists, mappings, and arbitrary objects are not FPS values.
        return 0.0
    return result if math.isfinite(result) and result > 0.0 else 0.0


def _video_info_fps(video_info) -> float:
    """Read VHS metadata across dict/object variants without touching payload bytes."""
    if video_info is None or isinstance(video_info, (bytes, bytearray, memoryview)):
        return 0.0
    keys = ("loaded_fps", "fps", "source_fps", "frame_rate")

    def candidates():
        if isinstance(video_info, Mapping):
            for key in keys:
                if key in video_info:
                    yield key, video_info.get(key)
            return
        for key in keys:
            try:
                yield key, getattr(video_info, key)
            except AttributeError:
                continue

    for key, raw_value in candidates():
        value = _safe_positive_float(raw_value)
        if value > 0.0:
            LOGGER.info("[TE DLSS5] fps selected from video_info.%s: %.3f", key, value)
            return value
        if isinstance(raw_value, (bytes, bytearray, memoryview)):
            LOGGER.info(
                "[TE DLSS5] video_info fps ignored: key=%s, type=%s",
                key, type(raw_value).__name__,
            )
    LOGGER.info("[TE DLSS5] no usable FPS found in video_info")
    return 0.0


def run_video_job(job: VideoJob, *, images=None, frame_count: int = 0, audio=None, video_info=None) -> Path:
    if os.name != "nt":
        raise RuntimeError("TE DLSS5 currently requires Windows and an NVIDIA NGX backend")
    backend_path = resolve_backend(job.backend_dll)
    if not backend_path:
        raise NativeBackendError(
            "No TE DLSS5 native backend configured. This node does not silently "
            "fall back to a non-DLSS filter."
        )

    # Resolve once up front so a missing executable is reported before native
    # initialization or temporary files are created.
    ffmpeg = _resolve_media_tool("ffmpeg")
    LOGGER.info("[TE DLSS5] FFmpeg executable: %s", ffmpeg)

    if images is None:
        if job.source is None or not job.source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {job.source}")
        info = _ffprobe(job.source)
        width, height = info["width"], info["height"]
        source_fps = info["fps"]
    else:
        height, width = int(images.shape[1]), int(images.shape[2])
        source_fps = 0.0
        source_fps = _video_info_fps(video_info)
        info = {}
    fps = job.output_fps if job.output_fps > 0 else source_fps
    if fps <= 0:
        # IMAGE batches do not carry timing metadata.  Keep the node usable
        # when VHS_VIDEOINFO is omitted, while making the fallback explicit.
        fps = 24.0

    settings = _settings(job)
    output = _output_path(job)
    frame_size = width * height * 4
    LOGGER.info(
        "[TE DLSS5] job start: guidance=%s, size=%sx%s, fps=%.3f, depth_interval=%d, style=%s, intensity=%.3f, local_tone=%.3f, local_structure=%.3f",
        job.guidance, width, height, fps, max(1, int(job.depth_interval)), job.style, settings["intensity"],
        settings["localToneStrength"], settings["localStructureStrength"],
    )
    LOGGER.info("[TE DLSS5] backend=%s, runtime=%s, output=%s", backend_path, settings["runtimeDir"], output)

    with tempfile.TemporaryDirectory(prefix="te_dlss5_") as temp_dir:
        silent_video = Path(temp_dir) / "video_only.mp4"
        encode = None
        decode = None
        ab_enhancer = None
        ab_enabled = os.environ.get("TE_DLSS5_AB_TEST", "").strip().lower() in {"1", "true", "yes", "on"}
        ab_mae = []
        ab_psnr = []
        async_executor = None
        pending_native = deque()
        frame_count = 0
        try:
            with FrameGuidance(width, height, job.guidance, settings["runtimeDir"], job.depth_interval) as guidance, NativeEnhancer(
                backend_path, width, height, settings
            ) as enhancer:
                if ab_enabled:
                    # The second instance receives the exact same source
                    # frames but no motion/depth. It is diagnostic only and
                    # never changes the encoded output.
                    try:
                        ab_enhancer = NativeEnhancer(
                            backend_path, width, height,
                            {**settings, "guidance": "zero"},
                        )
                        LOGGER.info(
                            "[TE DLSS5] A/B diagnostic enabled: guided=%s vs zero-guidance DLSSNR on identical frames",
                            job.guidance,
                        )
                    except Exception as exc:
                        LOGGER.warning("[TE DLSS5] A/B diagnostic unavailable; continuing guided run: %s", exc)
                        ab_enhancer = None
                        ab_enabled = False
                async_enabled = (
                    not ab_enabled and
                    os.environ.get("TE_DLSS5_ASYNC_NATIVE", "1").strip().lower()
                    in {"1", "true", "yes", "on"}
                )
                if async_enabled:
                    # The DLL remains single-threaded, but a one-worker queue
                    # lets guidance for frame N+1 overlap the native GPU work
                    # for frame N. Ordering is preserved by the executor.
                    async_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="te-dlss5-native")
                    LOGGER.info("[TE DLSS5] native submit queue enabled: depth=2")
                LOGGER.info("[TE DLSS5] native DLSSNR backend ready; frame pipeline=RGBA -> guidance -> DLSSNR Feature 18 -> H.264")
                encode = subprocess.Popen(
                    [
                        ffmpeg, "-y", "-v", "error", "-f", "rawvideo",
                        "-pix_fmt", "rgba", "-s:v", f"{width}x{height}", "-r", str(fps),
                        "-i", "-", "-an", *_encoder_args(ffmpeg),
                        "-pix_fmt", _output_pix_fmt(),
                        *_color_args(info),
                        str(silent_video),
                    ],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if encode.stdin is None:
                    raise RuntimeError("Failed to create FFmpeg output pipe")

                def process_frame(frame: bytes):
                    import numpy as np

                    rgba = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 4))
                    motion, depth = guidance.next(rgba)
                    reset = guidance.consume_reset()
                    if reset:
                        if async_executor is not None:
                            async_executor.submit(enhancer.reset)
                        else:
                            enhancer.reset()
                        if ab_enhancer is not None:
                            ab_enhancer.reset()
                    if async_executor is not None:
                        pending_native.append(async_executor.submit(
                            enhancer.process, frame, motion, depth))
                        if len(pending_native) <= 1:
                            return None
                        return pending_native.popleft().result()
                    guided = enhancer.process(frame, motion, depth)
                    if ab_enhancer is not None:
                        baseline = ab_enhancer.process(frame)
                        a = np.frombuffer(guided, dtype=np.uint8).astype(np.float32)
                        b = np.frombuffer(baseline, dtype=np.uint8).astype(np.float32)
                        diff = a - b
                        mae = float(np.mean(np.abs(diff)))
                        mse = float(np.mean(diff * diff))
                        psnr = 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))
                        ab_mae.append(mae)
                        ab_psnr.append(psnr)
                        index = len(ab_mae)
                        if index == 1 or index % 30 == 0:
                            LOGGER.info(
                                "[TE DLSS5] A/B frame=%d guided-vs-zero MAE=%.3f PSNR=%.2f dB",
                                index, mae, psnr,
                            )
                    return guided

                if images is None:
                    decode = subprocess.Popen(
                        [ffmpeg, "-v", "error", "-i", str(job.source), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    if decode.stdout is None:
                        raise RuntimeError("Failed to create FFmpeg input pipe")
                    while True:
                        frame = _read_exact(decode.stdout, frame_size)
                        if not frame:
                            break
                        if len(frame) != frame_size:
                            raise RuntimeError("FFmpeg ended with a partial RGBA frame")
                        processed = process_frame(frame)
                        if processed is not None:
                            encode.stdin.write(processed)
                        frame_count += 1
                        if frame_count == 1 or frame_count % 30 == 0:
                            LOGGER.info("[TE DLSS5] processed frame=%d", frame_count)
                else:
                    import numpy as np

                    for frame_tensor in images:
                        frame = frame_tensor.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
                        if frame.shape[-1] == 3:
                            alpha = np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)
                            frame = np.concatenate((frame, alpha), axis=-1)
                        if frame.shape != (height, width, 4):
                            raise RuntimeError("IMAGE frames must have 3 or 4 channels")
                        processed = process_frame(frame.tobytes())
                        if processed is not None:
                            encode.stdin.write(processed)
                        frame_count += 1
                        if frame_count == 1 or frame_count % 30 == 0:
                            LOGGER.info("[TE DLSS5] processed frame=%d", frame_count)
                while pending_native:
                    encode.stdin.write(pending_native.popleft().result())
                encode.stdin.close()
                decode_ok = images is not None or decode.wait() == 0
                encode_code = encode.wait()
                if not decode_ok or encode_code != 0:
                    details = []
                    for process in (decode, encode):
                        if process is not None and process.stderr is not None:
                            message = process.stderr.read(8192).decode("utf-8", "replace").strip()
                            if message:
                                details.append(message)
                    suffix = f": {' | '.join(details)}" if details else ""
                    raise RuntimeError(f"FFmpeg decode/encode failed{suffix}")
                LOGGER.info("[TE DLSS5] video encode complete: frames=%d, file=%s", frame_count, silent_video)
        finally:
            _stop_process(decode)
            _stop_process(encode)
            if async_executor is not None:
                async_executor.shutdown(wait=True, cancel_futures=True)
            if ab_enhancer is not None:
                ab_enhancer.close()

        if frame_count == 0:
            raise RuntimeError("The input video contained no frames")

        if images is not None and audio is not None:
            audio_wav = Path(temp_dir) / "audio.wav"
            if _write_audio_wav(audio, audio_wav):
                _run([ffmpeg, "-y", "-v", "error", "-i", str(audio_wav), "-i", str(silent_video), "-map", "1:v:0", "-map", "0:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output)])
            else:
                _run([ffmpeg, "-y", "-v", "error", "-i", str(silent_video), "-c", "copy", str(output)])
            LOGGER.info("[TE DLSS5] audio mux complete: source=AUDIO input, output=%s", output)
        elif images is None and audio is not None:
            _run(
                [
                    ffmpeg, "-y", "-v", "error", "-i", str(job.source),
                    "-i", str(silent_video), "-map", "1:v:0", "-map", "0:a?",
                    "-c:v", "copy", "-c:a", "copy", "-shortest", str(output),
                ]
            )
            LOGGER.info("[TE DLSS5] audio mux complete: source video audio, output=%s", output)
        else:
            _run([ffmpeg, "-y", "-v", "error", "-i", str(silent_video), "-c", "copy", str(output)])
            LOGGER.info("[TE DLSS5] output finalize complete: no audio, output=%s", output)

    LOGGER.info("[TE DLSS5] job complete: frames=%d, output=%s", frame_count, output)
    if ab_mae:
        LOGGER.info(
            "[TE DLSS5] A/B summary: frames=%d mean_MAE=%.3f mean_PSNR=%.2f dB",
            len(ab_mae), sum(ab_mae) / len(ab_mae), sum(ab_psnr) / len(ab_psnr),
        )
    return output
