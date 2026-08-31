注意:因模型文件较大,此github链接内文件还缺少模型文件夹请去网盘下载:
https://pan.quark.cn/s/2d4816b6cd1f
下载压缩包里的runtime文件移动至TE-ComfyUI-DLSS5/runtime即可.或者直接用网盘里的即可.

# TE-ComfyUI-DLSS5

## 版本

当前版本：**1.0**

一个在 ComfyUI 中使用 NVIDIA DLSSNR 的视频画质处理节点，对视频逐帧执行同分辨率神经画质处理。利用 DLSSNR 对画面的纹理、局部结构、色调和时域稳定性进行处理。

- 支持两种模式：
  - `nvof`：官方CUDA NVIDIA Optical Flow 运动引导。
  - `nvof_depth`：官方CUDA NVIDIA Optical Flow 加 ComfyUI原生pytorch的Depth Anything V2 深度引导。


## 自编译组件

项目通过 Python 调用外部命令，但包含两个本项目自行编译的 C++ 原生 DLL，
分别负责 DLSSNR 和 NVIDIA Optical Flow 的底层 GPU 接口。ComfyUI 节点负责工作流和模型调度，原生 DLL 负责显卡资源、设备同步和 NVIDIA SDK 调用。

### `te_dlss5_native.dll`

 D3D12/NGX 桥接层，负责：

- 创建和选择 D3D12 GPU 设备、命令队列、纹理和 Fence。
- 初始化 NVIDIA NGX Runtime，并创建 DLSSNR Feature 18。
- 传入颜色、运动、深度、历史重置和局部处理参数。
- 执行 GPU 处理并将结果读回给 ComfyUI 。


### `te_nvof_cuda.dll`

 CUDA Optical Flow 桥接层负责：

- 创建 CUDA 上下文和异步 CUDA Stream。
- 调用官方 NVIDIA Optical Flow SDK 的 CUDA API。
- 将 RGBA 视频帧上传到 NVOF，并取得 S10.5 运动向量和 cost。
- 对网格结果进行双线性稠密化。
- 通过反向光流一致性和 cost 计算逐像素 confidence，降低遮挡区域的错误引导。
- 向 Python 返回 DLSSNR 所需的 motion 和 confidence 数据。

 DLL 是插件的适配层，不是 NVIDIA 系统 DLL 运行时依赖显卡驱动提供的
`nvofapi64.dll`，



## ComfyUI 深度模型适配

`nvof_depth` 模式使用插件目录内置的 Depth Anything V2  模型计算深度，插件不会对视频一次性批量推理深度，而是按帧顺序处理，以保证深度历史、运动向量和
DLSSNR 历史状态保持连续。默认 `depth_interval=4`：首帧和每第 4 帧运行一次深度模型，
中间帧使用 NVOF 运动向量变形上一张深度图。这样可以在保持时域连续性的同时减少深度
模型调用；需要逐帧深度时可以将 `depth_interval` 设为 `1`。

深度融合会结合 NVOF confidence、运动边界、深度 residual 和场景切换检测，降低遮挡、快速运动或镜头切换时错误历史对画面的影响。Depth Anything V2 输出的是相对深度，不是游戏引擎提供的精确世界坐标深度。



###  注意: 视频没有游戏引擎的运动和深度数据

游戏可以直接提供物体运动矢量、相机参数、深度和曝光信息。普通视频只有颜色帧，
因此插件必须先用 NVOF 估计运动、用 Depth Anything V2 估计相对深度，再将它们转换成DLSSNR 能接受的格式。估计误差会直接影响纹理稳定性和时域细节。
因此，1.0 可以实际使用 DLSSNR 处理视频，但在快速运动、细小纹理、遮挡边缘和光照
变化上，效果不一定达到实时游戏的效果。看到的差异还会受到源视频压缩、RGBA8 转换、
重新编码和原版是否同时启用了其他渲染处理的影响。

###  注意: DLSS5天生对3D游戏画面较敏感,效果较为明显,真人效果不太明显.

DLSSNR 的训练和输入假设更接近实时渲染画面。3D 游戏通常能直接提供准确的运动矢量、
场景深度、相机抖动、曝光和高精度渲染纹理；普通视频只有经过压缩的颜色帧，插件只能
使用 NVOF 和 Depth Anything V2 估计这些引导数据。因此真人视频中的提升可能更多表现为降噪、局部色调变化和运动稳定，未必像 3D 游戏那样明显地产生细小纹理。



