"""ComfyUI IMAGE-batch pipeline for the in-process TE DLSSG bridge."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .dlssg_backend import FrameGenerationError, NativeFrameGenerator, resolve_frame_generator
from .frame_guidance import FrameGuidance
from .video_pipeline import (
    _color_args, _encoder_args, _output_path, _output_pix_fmt,
    _resolve_media_tool, _run, _stop_process, _video_info_fps,
    _write_audio_wav,
)
from .native_backend import resolve_runtime_dir

LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


@dataclass
class FrameGenerationJob:
    output_prefix: str
    # DLSSG performs its own motion estimation.  External NVOF vectors are
    # intentionally not passed to the frame-generation bridge because they
    # can violate the runtime's motion contract and corrupt generated frames.
    guidance: str = "zero"
    output_fps: float = 0.0


def run_frame_generation_job(job: FrameGenerationJob, *, images, frame_count: int = 0,
                             audio=None, video_info=None) -> tuple[Path, dict]:
    if os.name != "nt":
        raise RuntimeError("TE DLSS5 frame generation currently requires Windows and NVIDIA DLSSG")
    import numpy as np

    if images is None or len(images.shape) != 4:
        raise TypeError("images must be an IMAGE batch")
    total = int(images.shape[0])
    if total < 2:
        raise ValueError("at least two real frames are required")
    height, width = int(images.shape[1]), int(images.shape[2])
    channels = int(images.shape[3])
    if channels not in (3, 4):
        raise ValueError(f"IMAGE frames must have 3 or 4 channels, got {channels}")
    selected = int(frame_count) if int(frame_count) > 0 else total
    selected = min(selected, total)
    if selected < 2:
        raise ValueError("frame_count must select at least two frames")
    fps = float(job.output_fps) if float(job.output_fps) > 0 else _video_info_fps(video_info)
    if fps <= 0:
        fps = 24.0
    output_fps = fps * 2.0
    ffmpeg = _resolve_media_tool("ffmpeg")
    dll = resolve_frame_generator()
    if not dll:
        raise FrameGenerationError(
            "te_dlssg_native.dll is not installed; build it with native\\build_dlssg.bat "
            "and the NVIDIA DLSS SDK path"
        )
    runtime = resolve_runtime_dir()
    output = _output_path(type("OutputJob", (), {"output_prefix": job.output_prefix, "overwrite": False})())
    frame_bytes = width * height * 4
    # Keep the field for ABI/log compatibility, but force the only supported
    # frame-generation mode.  This also makes old workflows containing
    # nvof/nvof_depth safe after the node UI was reduced to `zero`.
    guidance_mode = "zero"
    settings = {"runtimeDir": str(runtime), "guidance": guidance_mode}
    LOGGER.info(
        "[TE DLSS5] DLSSG job start: guidance=%s, size=%sx%s, fps=%.3f->%.3f, input_frames=%d",
        guidance_mode, width, height, fps, output_fps, selected,
    )
    encode = None
    frame_written = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="te_dlssg_") as temp_dir:
        silent = Path(temp_dir) / "video_only.mp4"
        try:
            with NativeFrameGenerator(dll, width, height, settings) as generator:
                encode_args = _encoder_args(ffmpeg)
                encode = subprocess.Popen(
                    [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                     "-s:v", f"{width}x{height}", "-r", str(output_fps), "-i", "-",
                     "-an", *encode_args, "-pix_fmt", _output_pix_fmt(), str(silent)],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if encode.stdin is None:
                    raise RuntimeError("Failed to create FFmpeg output pipe")
                previous_rgba = None
                for index, tensor in enumerate(images[:selected]):
                    frame = tensor.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
                    if frame.shape[-1] == 3:
                        frame = np.concatenate(
                            (frame, np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)), axis=-1
                        )
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
                    rgba = frame.tobytes(order="C")
                    # The DLSSG runtime estimates motion internally.  Passing
                    # nullptr guides is equivalent to the working
                    # ComfyUI-DLSS-NR-2 path, which supplies a zero motion
                    # plane and avoids the incompatible external NVOF contract.
                    motion = depth = None
                    if index == 0:
                        # Prime DLSSG with the first real frame.  The bridge
                        # intentionally returns no generated frame while its
                        # temporal history is initialized.  Doing this before
                        # writing the first frame lets the next call produce
                        # the midpoint between frame 0 and frame 1 instead of
                        # duplicating frame 1 at the start of the 2x stream.
                        generator.process(rgba, motion, depth)
                        encode.stdin.write(rgba)
                        frame_written += 1
                    else:
                        # A detected scene cut invalidates the pair crossing
                        # the cut. Prime the new DLSSG history with the
                        # current real frame, then hold the previous frame for
                        # the midpoint so the CFR slot count remains exact.
                        reset = False
                        generated = generator.process(rgba, motion, depth)
                        if reset:
                            if previous_rgba is not None:
                                encode.stdin.write(previous_rgba)
                            encode.stdin.write(rgba)
                            frame_written += 2 if previous_rgba is not None else 1
                        elif generated is not None:
                            encode.stdin.write(generated)
                            frame_written += 1
                            encode.stdin.write(rgba)
                            frame_written += 1
                        else:
                            # The first frame after a reset/history prime has
                            # no generated output. Keep the timeline valid by
                            # duplicating the current frame for its midpoint.
                            encode.stdin.write(rgba)
                            encode.stdin.write(rgba)
                            frame_written += 2
                    previous_rgba = rgba
                    if index == 0 or (index + 1) % 30 == 0:
                        LOGGER.info(
                            "[TE DLSS5] DLSSG processed real frame=%d, generated=%s, output_frames=%d",
                            index + 1, "yes" if index > 0 and generated is not None else "no", frame_written,
                        )
                if previous_rgba is not None and frame_written < selected * 2:
                    # Match the source duration exactly. A stream of N real
                    # frames has N*2 output slots at 2x, so the final slot is a
                    # held copy of the last real frame.
                    encode.stdin.write(previous_rgba)
                    frame_written += 1
                encode.stdin.close()
                rc = encode.wait()
                if rc != 0:
                    detail = encode.stderr.read().decode("utf-8", "replace")[-2000:] if encode.stderr else ""
                    raise RuntimeError(f"FFmpeg encode failed ({rc}): {detail}")
                encode = None
        finally:
            _stop_process(encode)
        if audio is not None:
            wav = Path(temp_dir) / "audio.wav"
            if _write_audio_wav(audio, wav):
                _run([ffmpeg, "-y", "-v", "error", "-i", str(wav), "-i", str(silent),
                      "-map", "1:v:0", "-map", "0:a:0", "-c:v", "copy", "-c:a", "aac",
                      "-shortest", str(output)])
            else:
                _run([ffmpeg, "-y", "-v", "error", "-i", str(silent), "-c", "copy", str(output)])
        else:
            _run([ffmpeg, "-y", "-v", "error", "-i", str(silent), "-c", "copy", str(output)])
    elapsed = time.perf_counter() - started
    stats = {"input_frames": selected, "output_frames": frame_written, "output_fps": output_fps, "elapsed": elapsed}
    LOGGER.info(
        "[TE DLSS5] DLSSG job complete: %d -> %d frames, %.2fs, output=%s",
        selected, frame_written, elapsed, output,
    )
    return output, stats
