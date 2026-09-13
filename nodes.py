from __future__ import annotations

from .native_backend import NativeBackendError, NativeEnhancer, resolve_backend, resolve_runtime_dir
from .frame_guidance import FrameGuidance
from .video_pipeline import VideoJob, run_video_job
from .frame_generation_pipeline import FrameGenerationJob, run_frame_generation_job


def _bounded_error(exc: BaseException, limit: int = 1600) -> str:
    """Keep malformed/binary third-party diagnostics from flooding ComfyUI logs."""
    text = str(exc)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [diagnostic truncated; {len(text)} characters]"


class TE_DLSS5_VideoEnhancer:
    """Stream a video through DLSSNR, optionally preceded by native DLSS SR.

    The node returns ComfyUI's VIDEO object backed by the encoded output file.
    Frames are still processed as an IMAGE batch internally, but downstream
    video nodes receive a normal VIDEO connection rather than a bare path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_count": ("INT", {"default": 0, "min": 0, "tooltip": "0 = auto：处理全部输入帧；大于 0 时只处理指定数量的帧。"}),
                "style": (["default", "natural", "cinematic"],),
                "intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "guidance": (["nvof_depth", "nvof"],),
                "depth_interval": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "output_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01, "tooltip": "0 = auto：使用视频信息中的帧率；大于 0 时指定输出帧率。自动模式需要连接 video_info。"}),
                "output_prefix": ("STRING", {"default": "TE_DLSS5/enhanced"}),
                # Append new controls after the original workflow fields so
                # loading older nodes does not shift existing widget values.
                "nr_preset": (["0 Default", "1 Preset #1", "2 Preset #2", "3 Preset #3"],),
                "skin_structure_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                # Keep this new control last so existing workflow widget
                # positions and values remain unchanged.
                "upscale_factor": (["1x (原分辨率)", "2x", "4x"], {"tooltip": "1x：同分辨率 NR；2x/4x：原生 DLSS Super Resolution 后接 NR，需要 te_dlss_sr_native.dll。"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "status")
    FUNCTION = "execute"
    CATEGORY = "TE/DLSS5"

    def execute(
        self,
        images,
        frame_count: int,
        style: str,
        intensity: float,
        local_tone: float,
        local_structure: float,
        nr_preset: str,
        skin_structure_strength: float,
        automatic_mask: bool,
        ui_correction: bool,
        guidance: str,
        depth_interval: int,
        output_fps: float,
        output_prefix: str,
        upscale_factor: str,
        audio=None,
        video_info=None,
        **_legacy_options,
    ):
        if images is None or not hasattr(images, "shape") or len(images.shape) != 4:
            raise TypeError("images must be a ComfyUI IMAGE batch [frames,height,width,channels]")
        job = VideoJob(
            source=None,
            output_prefix=output_prefix,
            style=style,
            intensity=float(intensity),
            local_tone=float(local_tone),
            local_structure=float(local_structure),
            nr_preset=int(str(nr_preset).split()[0]),
            skin_structure_strength=float(skin_structure_strength),
            automatic_mask=bool(automatic_mask),
            ui_correction=bool(ui_correction),
            guidance=guidance,
            depth_interval=max(1, min(8, int(depth_interval))),
            output_fps=float(output_fps),
            backend_dll="",
            runtime_dir="",
            overwrite=False,
            scale=int(str(upscale_factor).split("x")[0]),
        )
        try:
            output = run_video_job(job, images=images, frame_count=int(frame_count), audio=audio, video_info=video_info)
        except Exception as exc:
            # ComfyUI prints the exception and traceback. Re-raise a bounded
            # text-only error so a codec/provider cannot dump an entire frame
            # as an escaped byte string and hide the actionable log lines.
            raise RuntimeError(f"TE DLSS5 processing failed: {_bounded_error(exc)}") from None
        try:
            from comfy_api.input_impl import VideoFromFile
        except ImportError as exc:
            raise RuntimeError(
                "This ComfyUI version does not provide the VIDEO input type; "
                "update ComfyUI or use the legacy video_path output."
            ) from exc
        return (
            VideoFromFile(str(output)),
            f"TE DLSS5 NR complete: {output.name} | scale={job.scale}x | guidance={guidance} | depth_interval={max(1, int(depth_interval))} | "
            f"nr_preset={job.nr_preset} | skin_structure={job.skin_structure_strength:.2f} | "
            f"auto_mask={int(job.automatic_mask)} | ui_correction={int(job.ui_correction)}",
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
                # Keep the original picture-node widgets in place; new
                # controls are appended for workflow compatibility.
                "nr_preset": (["0 Default", "1 Preset #1", "2 Preset #2", "3 Preset #3"],),
                "skin_structure_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "ui_correction": ("BOOLEAN", {"default": False}),
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
        nr_preset: str,
        skin_structure_strength: float,
        automatic_mask: bool,
        ui_correction: bool,
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
                "nrPreset": max(0, min(3, int(str(nr_preset).split()[0]))),
                "skinStructureStrength": max(-1.0, min(2.0, float(skin_structure_strength))),
                "automaticMask": bool(automatic_mask),
                "uiCorrection": bool(ui_correction),
                "guidance": str(guidance),
                "runtimeDir": str(resolve_runtime_dir()),
                # The still-image path deliberately keeps the established
                # synchronous staging ABI. Shared slots are a temporal NVOF
                # video optimization and require the video scheduler.
                "sharedResources": False,
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
                f"local_structure={settings['localStructureStrength']:.2f} | nr_preset={settings['nrPreset']} | "
                f"skin_structure={settings['skinStructureStrength']:.2f} | auto_mask={int(settings['automaticMask'])} | "
                f"ui_correction={int(settings['uiCorrection'])}",
            )
        except Exception as exc:
            raise RuntimeError(f"TE DLSS5 picture processing failed: {_bounded_error(exc)}") from None


class TE_DLSS5_FrameInterpolator:
    """Generate one DLSSG frame between each pair in an IMAGE video batch.

    This node uses the in-process ``te_dlssg_native.dll`` bridge.  It does not
    launch ``dlssg-worker.exe`` or depend on another ComfyUI extension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_count": ("INT", {"default": 0, "min": 0, "tooltip": "0 = auto：处理全部输入帧；大于 0 时只处理指定数量的帧。"}),
                # DLSSG estimates motion internally. External NVOF guidance is
                # intentionally not exposed for frame generation.
                "guidance": (["zero"],),
                "output_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01, "tooltip": "0 = auto：从 video_info 读取源帧率，缺失时使用 24 FPS。手动值也是源帧率；2x 插帧最终输出帧率为该值的两倍。"}),
                "output_prefix": ("STRING", {"default": "TE_DLSS5/interpolated"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "status")
    FUNCTION = "execute"
    CATEGORY = "TE/DLSS5"

    def execute(self, images, frame_count: int, guidance: str,
                output_fps: float, output_prefix: str,
                audio=None, video_info=None, **_legacy_options):
        if images is None or not hasattr(images, "shape") or len(images.shape) != 4:
            raise TypeError("images must be a ComfyUI IMAGE batch [frames,height,width,channels]")
        if int(images.shape[0]) < 2:
            raise ValueError("DLSSG frame interpolation needs at least two real frames")
        job = FrameGenerationJob(
            output_prefix=output_prefix,
            guidance="zero",
            output_fps=float(output_fps),
        )
        try:
            output, stats = run_frame_generation_job(
                job, images=images, frame_count=int(frame_count),
                audio=audio, video_info=video_info,
            )
        except Exception as exc:
            raise RuntimeError(f"TE DLSS5 frame generation failed: {_bounded_error(exc)}") from None
        try:
            from comfy_api.input_impl import VideoFromFile
        except ImportError as exc:
            raise RuntimeError("This ComfyUI version does not provide the VIDEO input type") from exc
        return (
            VideoFromFile(str(output)),
            f"TE DLSS5 DLSSG 2x complete: {output.name} | guidance=zero (internal DLSSG motion) | "
            f"frames={stats['input_frames']}->{stats['output_frames']} | fps={stats['output_fps']:.3f}",
        )


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


NODE_CLASS_MAPPINGS = {
    "TE_DLSS5_VideoEnhancer": TE_DLSS5_VideoEnhancer,
    "TE_DLSS5_PictureEnhancer": TE_DLSS5_PictureEnhancer,
    "TE_DLSS5_FrameInterpolator": TE_DLSS5_FrameInterpolator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TE_DLSS5_VideoEnhancer": "TE DLSS5 Video Enhancer (NR)",
    "TE_DLSS5_PictureEnhancer": "TE DLSS5 Picture Enhancer (NR)",
    "TE_DLSS5_FrameInterpolator": "TE DLSS5 Frame Interpolation (DLSSG)",
}
