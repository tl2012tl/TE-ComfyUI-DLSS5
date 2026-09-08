from __future__ import annotations

from .native_backend import NativeBackendError, NativeEnhancer, resolve_backend, resolve_runtime_dir
from .frame_guidance import FrameGuidance
from .video_pipeline import VideoJob, run_video_job, video_info_fps


def _bounded_error(exc: BaseException, limit: int = 1600) -> str:
    """Keep malformed/binary third-party diagnostics from flooding ComfyUI logs."""
    text = str(exc)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [diagnostic truncated; {len(text)} characters]"


class TE_DLSS5_VideoEnhancer:
    """Stream a video through the TE native same-resolution enhancer.

    The node returns ComfyUI's VIDEO object backed by the encoded output file.
    Frames are still processed as an IMAGE batch internally, but downstream
    video nodes receive a normal VIDEO connection rather than a bare path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "style": (["default", "natural", "cinematic"],),
                "intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "guidance": (["nvof_depth", "nvof"],),
                "depth_interval": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "output_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "output_prefix": ("STRING", {"default": "TE_DLSS5/enhanced"}),
                "output_mode": (["video", "frames"],),
            },
            "optional": {
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "status")
    FUNCTION = "execute"
    CATEGORY = "TE/DLSS5"

    def execute(
        self,
        images,
        style: str,
        intensity: float,
        local_tone: float,
        local_structure: float,
        guidance: str,
        depth_interval: int,
        output_fps: float,
        output_prefix: str,
        output_mode: str = "video",
        audio=None,
        video_info=None,
        **_legacy_options,
    ):
        if images is None or not hasattr(images, "shape") or len(images.shape) != 4:
            raise TypeError("images must be a ComfyUI IMAGE batch [frames,height,width,channels]")
        mode = str(output_mode).strip().lower()
        if mode not in {"video", "frames"}:
            raise ValueError(f"output_mode must be 'video' or 'frames', got {output_mode!r}")
        job = VideoJob(
            source=None,
            output_prefix=output_prefix,
            style=style,
            intensity=float(intensity),
            local_tone=float(local_tone),
            local_structure=float(local_structure),
            guidance=guidance,
            depth_interval=max(1, min(8, int(depth_interval))),
            output_fps=float(output_fps),
            backend_dll="",
            runtime_dir="",
            overwrite=False,
            output_mode=mode,
        )
        try:
            output, frames, processed_count = run_video_job(
                job, images=images,
                audio=audio if mode == "video" else None,
                video_info=video_info,
            )
        except Exception as exc:
            # ComfyUI prints the exception and traceback. Re-raise a bounded
            # text-only error so a codec/provider cannot dump an entire frame
            # as an escaped byte string and hide the actionable log lines.
            raise RuntimeError(f"TE DLSS5 processing failed: {_bounded_error(exc)}") from None
        # Surface an assumed FPS in the UI, not only in the console log, so a
        # silent 24fps fallback cannot masquerade as the source frame rate.
        fps_assumed = float(output_fps) <= 0.0 and video_info_fps(video_info) <= 0.0
        fps_note = " | fps=24 assumed (no video_info connected and output_fps=0)" if fps_assumed and mode == "video" else ""
        if mode == "frames":
            return (
                None,
                frames,
                f"TE DLSS5 NR complete: {processed_count} frame(s) | guidance={guidance} | "
                f"depth_interval={max(1, int(depth_interval))} | output=IMAGE batch",
            )
        try:
            from comfy_api.input_impl import VideoFromFile
        except ImportError as exc:
            raise RuntimeError(
                "This ComfyUI version does not provide the VIDEO input type; "
                "update ComfyUI or use the legacy video_path output."
            ) from exc
        return (
            VideoFromFile(str(output)),
            None,
            f"TE DLSS5 NR complete: {output.name} | guidance={guidance} | "
            f"depth_interval={max(1, int(depth_interval))}{fps_note}",
        )


class TE_DLSS5_PictureEnhancer:
    """Apply one-frame DLSSNR processing to a ComfyUI IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "style": (["cinematic", "default", "natural"],),
                "intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "guidance": (["nvof_depth", "depth", "zero"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "execute"
    CATEGORY = "TE/DLSS5"

    def execute(
        self,
        image,
        style: str,
        intensity: float,
        local_tone: float,
        local_structure: float,
        guidance: str,
    ):
        if image is None or not hasattr(image, "shape") or len(image.shape) != 4:
            raise TypeError("image must be a ComfyUI IMAGE batch [1,height,width,channels]")
        if int(image.shape[0]) != 1:
            raise ValueError(
                "TE DLSS5 Picture Enhancer accepts one image only; "
                "use TE DLSS5 Video Enhancer for an IMAGE batch"
            )
        channels = int(image.shape[-1])
        if channels not in (3, 4):
            raise ValueError(f"image must have 3 or 4 channels, got {channels}")

        try:
            import numpy as np
            import torch

            # Match the video pipeline's float-to-RGBA8 conversion exactly.
            frame = image[0].detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
            if channels == 3:
                alpha = np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)
                frame = np.concatenate((frame, alpha), axis=-1)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            backend = resolve_backend()
            if not backend:
                raise NativeBackendError(
                    "TE DLSS5 native backend is not built; build te_dlss5_native.dll first"
                )
            settings = {
                "style": str(style),
                "intensity": max(-1.0, min(2.0, float(intensity))),
                "localToneStrength": max(-1.0, min(2.0, float(local_tone))),
                "localStructureStrength": max(-1.0, min(2.0, float(local_structure))),
                "guidance": str(guidance),
                "runtimeDir": str(resolve_runtime_dir()),
            }
            frame_bytes = frame.tobytes(order="C")
            with NativeEnhancer(backend, width, height, settings) as enhancer:
                if guidance == "zero":
                    result = enhancer.process(frame_bytes)
                else:
                    # A still image has no temporal pair. NVOF therefore
                    # returns a zero bootstrap vector; depth mode avoids
                    # starting an optical-flow session when only depth is
                    # needed. Both paths use the same guided native call as
                    # the first frame of the video node.
                    with FrameGuidance(
                        width, height, str(guidance), settings["runtimeDir"], depth_interval=1
                    ) as guide:
                        motion, depth = guide.next(frame)
                        if guide.consume_reset():
                            enhancer.reset()
                        result = enhancer.process(frame_bytes, motion, depth)
            output = np.frombuffer(result, dtype=np.uint8).reshape((height, width, 4)).copy()
            if channels == 3:
                output = output[..., :3]
            output = torch.from_numpy(output).float().div(255.0).unsqueeze(0)
            return (
                output,
                f"TE DLSS5 NR picture complete: {width}x{height} | guidance={guidance} | style={style} | "
                f"intensity={settings['intensity']:.2f} | local_tone={settings['localToneStrength']:.2f} | "
                f"local_structure={settings['localStructureStrength']:.2f}",
            )
        except Exception as exc:
            raise RuntimeError(f"TE DLSS5 picture processing failed: {_bounded_error(exc)}") from None


class TE_DLSS5_BackendStatus:
    """Expose backend availability without starting a video job."""

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "probe"
    CATEGORY = "TE/DLSS5"

    def probe(self):
        from .native_backend import describe_backend

        return (describe_backend(),)


class TE_DLSS5_RuntimeInfo:
    """Full runtime diagnostics: host, NGX bridge, NVOF, depth, media."""

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("runtime_info",)
    FUNCTION = "probe"
    CATEGORY = "TE/DLSS5"

    def probe(self):
        from .native_backend import build_runtime_report

        return (build_runtime_report(),)


NODE_CLASS_MAPPINGS = {
    "TE_DLSS5_VideoEnhancer": TE_DLSS5_VideoEnhancer,
    "TE_DLSS5_PictureEnhancer": TE_DLSS5_PictureEnhancer,
    "TE_DLSS5_BackendStatus": TE_DLSS5_BackendStatus,
    "TE_DLSS5_RuntimeInfo": TE_DLSS5_RuntimeInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TE_DLSS5_VideoEnhancer": "TE DLSS5 Video Enhancer (NR)",
    "TE_DLSS5_PictureEnhancer": "TE DLSS5 Picture Enhancer (NR)",
    "TE_DLSS5_BackendStatus": "TE DLSS5 Backend Status",
    "TE_DLSS5_RuntimeInfo": "TE DLSS5 Runtime Info",
}
