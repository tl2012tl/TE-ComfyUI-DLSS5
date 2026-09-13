# TE-ComfyUI-DLSS5

## 版本

当前版本：**2.0**

TE-ComfyUI-DLSS5 是面向 ComfyUI 视频工作流的 NVIDIA DLSS 画质处理插件。它将原生 DLSSNR、DLSS Super Resolution、DLSS Frame Generation、CUDA Optical Flow 和 ComfyUI 深度模型组合成完整的视频处理流程。

2.0 增加了原生 DLSS 放大、原生 DLSSG 插帧，以及 CUDA/D3D12 GPU 资源互操作和异步帧队列。

## 2.0 更新内容

### 原生 DLSS Super Resolution 放大

`TE DLSS5 Video Enhancer (NR)` 节点新增放大倍率：

- `1x (原分辨率)`：保持原分辨率，仅执行 DLSSNR 画质处理。
- `2x`：先使用原生 DLSS Super Resolution 放大，再执行 DLSSNR。
- `4x`：使用原生 DLSS Super Resolution 进行更高倍率放大，再执行 DLSSNR。

2x/4x 调用 NVIDIA DLSS Super Sampling 神经网络。输出分辨率会按照源视频尺寸计算，并在日志中显示 SR 阶段的执行状态和耗时。

### 原生 DLSSG 插帧

`TE DLSS5 Frame Interpolation (DLSSG)` 节点使用 NVIDIA DLSS Frame Generation，在相邻真实帧之间生成中间帧，实现 2x 帧率输出。

- 输入 24 FPS 视频，输出约 48 FPS。
- 输入 30 FPS 视频，输出约 60 FPS。


### CUDA/D3D12 GPU 资源互操作

2.0 支持三帧资源环和异步 Fence 调度,减少重复的 CPU 上传和同步等待。



### 1.0

在 ComfyUI 中使用 NVIDIA DLSSNR 的视频画质处理节点，对视频逐帧执行同分辨率神经画质处理。利用 DLSSNR 对画面的纹理、局部结构、色调和时域稳定性进行处理。

支持两种模式：
nvof：官方CUDA NVIDIA Optical Flow 运动引导。
nvof_depth：官方CUDA NVIDIA Optical Flow 加 ComfyUI原生pytorch的Depth Anything V2 深度引导。



## 节点

### TE DLSS5 Video Enhancer (NR)

对 IMAGE 帧序列执行视频级 DLSSNR 处理，并输出 VIDEO。

主要选项：

- `frame_count`：处理帧数。`0` 表示自动处理全部输入帧。
- `style`：画面风格，可选 `default`、`natural`、`cinematic`。
- `intensity`：整体处理强度。
- `local_tone`：局部色调和光照变化强度。
- `local_structure`：局部纹理和结构增强强度。
- `guidance`：使用 `nvof` 或 `nvof_depth` 作为时域引导。
- `depth_interval`：深度模型运行间隔，默认 `4`。数值越小深度更新越频繁，但耗时越高。
- `output_fps`：输出帧率。`0` 表示自动读取视频信息。
- `nr_preset`：DLSSNR 预设。
- `skin_structure_strength`：皮肤和人物表面结构强度。
- `automatic_mask`：启用自动区域掩码。
- `ui_correction`：启用界面或平面文字区域修正。
- `upscale_factor`：输出放大倍率，支持 1x、2x、4x。

### TE DLSS5 Picture Enhancer (NR)

对单张 IMAGE 执行一次 DLSSNR 处理，输出 IMAGE。适合静态图片或视频中的单帧测试。

图片节点没有连续视频历史，因此 NVOF 只能提供首帧引导；如果需要完整的时域效果，应使用 Video Enhancer 节点处理 IMAGE 批次。

### TE DLSS5 Frame Interpolation (DLSSG)

对 IMAGE 帧序列执行 DLSSG 2x 插帧并输出 VIDEO。该节点只负责生成中间帧，不执行 DLSSNR 放大或画质增强。需要同时放大和插帧时，可先使用 Video Enhancer，再连接 Frame Interpolation。

## 引导数据与深度模型

### NVOF 运动引导

插件使用 NVIDIA Optical Flow 估计相邻帧的像素运动，经过双线性稠密化和前后向一致性检查后，转换为 DLSSNR 可使用的运动纹理和 confidence。

### Depth Anything V2 深度引导

`nvof_depth` 模式使用插件内置的 Depth Anything V2 模型计算相对深度。默认每 4 帧运行一次深度推理，中间帧使用运动向量变形上一张深度图，并结合 residual 和 confidence 做时域融合。

深度图用于帮助 DLSSNR 区分前景、背景、运动边缘和遮挡区域。它是单目估计的相对深度，不是游戏引擎提供的精确世界坐标深度。

## 处理流程

普通 1x 视频：

```text
RGBA 帧 → NVOF/Depth 引导 → DLSSNR Feature 18 → 视频编码
```

2x/4x 视频：

```text
RGBA 帧 → 原生 DLSS Super Resolution → NVOF/Depth 引导 → DLSSNR Feature 18 → 视频编码
```

插帧视频：

```text
真实帧序列 → NVOF/时域数据 → DLSSG Frame Generation → 2x 帧率视频
```

## 日志说明

成功运行时，可以关注以下日志：

- `native DLSS SR ready`：原生 DLSS 放大已初始化。
- `native DLSS SR evaluate ok`：当前 SR 帧处理成功。
- `NVOF ready`：CUDA Optical Flow 已初始化。
- `Depth model ready`：Depth Anything V2 已加载并开始工作。
- `DLSSNR native bridge ready`：DLSSNR Feature 18 已创建。
- `CUDA-D3D12 interop active`：共享 GPU 纹理和 Fence 已启用。
- `encoder=h264_nvenc`：使用 NVIDIA 硬件编码。
- `timing avg/frame`：显示 SR、引导、DLSSNR 和编码各阶段平均耗时。

## 注意事项

- DLSS 模型主要针对实时渲染画面训练。普通视频缺少游戏引擎的真实运动矢量、相机参数、曝光和材质数据，因此效果会因视频内容而异。
- 真人视频中，提升可能主要表现为降噪、边缘稳定、局部结构和色调改善，不一定像 3D 游戏那样明显改变光照和材质。
- DLSS 放大 DLSSG 插帧都是针对实时渲染游戏画面训练的神经网络,仅供尝鲜,效果自测。


