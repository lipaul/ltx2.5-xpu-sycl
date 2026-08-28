# LTX-2.5 distilled pipeline — XPU 算力 / 显存利用率统计

对比模式: **baseline** = bf16 权重 + `--offload cpu`（不用 xpu-ltx-kernels） vs **blockwise** = `--quantization blockwise`（fp8 blockwise 存储、无 offload，走 xpu-ltx-kernels）

配置: 9 帧 @ 768x512，seed 42，设备 Intel(R) Arc(TM) Pro B60 Graphics (0000:0f:00.0, 25.7 GB)

B60 实际算力上限（微基准）: bf16 GEMM ≈ 92.0 TFLOPS (与能力报告的 ~96 TFLOPS oneDNN 峰值一致)，内存带宽 ≈ 398.4 GB/s


| 阶段 | baseline 耗时(s) | baseline GT忙% | baseline 核忙% | baseline 显存利用% | blockwise 耗时(s) | blockwise GT忙% | blockwise 核忙% | blockwise 显存利用% | 加速 |
|---|---|---|---|---|---|---|---|---|
| 文本编码 (Gemma) | 28.8 | 23.0 | 0.0 | 32.9% | 38.5 | 29.9 | 0.0 | 98.3% | 0.75x |
| 条件编码 (VAE enc S1) | 0.4 | 37.4 | 79.7 | 4.7% | 0.7 | 6.6 | 19.3 | 5.4% | 0.60x |
| Stage1 去噪 (8步) | 34.0 | 65.1 | 0.0 | 25.5% | 35.2 | 48.4 | 0.0 | 91.8% | 0.97x |
| 空间上采样 (VAE) | 1.5 | 34.1 | 35.4 | 7.1% | 1.9 | 45.1 | 24.5 | 7.1% | 0.79x |
| 条件编码 (VAE enc S2) | 0.4 | 26.7 | 48.0 | 2.8% | 0.5 | 32.9 | 35.1 | 2.6% | 0.84x |
| Stage2 去噪 (3步) | 14.7 | 55.4 | 0.0 | 25.5% | 21.9 | 75.7 | 0.0 | 91.8% | 0.67x |
| 音频解码 (AudioVAE) | 10.3 | 28.6 | 2.7 | 7.0% | 8.3 | 47.6 | 3.4 | 7.0% | 1.24x |
| 视频解码 (DiffVAE) | 24.6 | 43.4 | 39.1 | 10.6% | 23.0 | 50.2 | 41.6 | 10.6% | 1.07x |
| 视频编码输出 (ffmpeg) | 0.1 | n/a | n/a | 0.0% | 0.1 | n/a | n/a | 0.0% | 0.70x |

## VAE 阶段 B60 算力分析（重点）

`GT忙%` = DRM C6 驻留计数器推出的硬件引擎活跃占比（≈intel_gpu_top 的 GPU busy）；`核忙%` = torch profiler 的 XPU kernel 时间 / 墙钟时间。两者都代表“B60 真正在算”的比例，差距来自：主存/磁盘流式加载、分配与同步开销、tile 化解码的宿主端间隙。**理论上限 = 100% GT 忙且 100% 核忙**；此外 VAE 是卷积+带宽受限工作，其算力天花板远低于 GEMM 峰值（见微基准带宽）。


| VAE 阶段 | 模式 | 耗时(s) | GT忙% | 核忙% | 空闲差距(=100-GT忙) | Top kernels |
|---|---|---|---|---|---|---|
| 音频解码 (AudioVAE) | baseline | 10.3 | 28.6 | 2.7 | 71.4% | aten::copy_(34.5%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(32.7%), aten::convolution_overrideable(10.9%) |
| 音频解码 (AudioVAE) | blockwise | 8.3 | 47.6 | 3.4 | 52.4% | aten::copy_(38.7%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(36.9%), aten::convolution_overrideable(7.6%) |
| 条件编码 (VAE enc S1) | baseline | 0.4 | 37.4 | 79.7 | 62.6% | aten::copy_(33.4%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(33.4%), zeCommandListHostSynchronize(33.2%) |
| 条件编码 (VAE enc S1) | blockwise | 0.7 | 6.6 | 19.3 | 93.4% | aten::copy_(49.2%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(49.2%), zeCommandListHostSynchronize(1.2%) |
| 条件编码 (VAE enc S2) | baseline | 0.4 | 26.7 | 48.0 | 73.3% | aten::copy_(50.0%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(50.0%) |
| 条件编码 (VAE enc S2) | blockwise | 0.5 | 32.9 | 35.1 | 67.1% | aten::copy_(50.0%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(50.0%) |
| 空间上采样 (VAE) | baseline | 1.5 | 34.1 | 35.4 | 65.9% | aten::copy_(45.3%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(45.3%), aten::convolution_overrideable(4.6%) |
| 空间上采样 (VAE) | blockwise | 1.9 | 45.1 | 24.5 | 54.9% | aten::copy_(43.5%), Memcpy M2D (MEMORY(Unknown) -> DEVICE)(43.5%), aten::convolution_overrideable(6.3%) |
| 视频解码 (DiffVAE) | baseline | 24.6 | 43.4 | 39.1 | 56.6% | aten::_scaled_dot_product_fused_attention_overrideable(32.7%), micro_sdpa(32.7%), aten::copy_(5.7%) |
| 视频解码 (DiffVAE) | blockwise | 23.0 | 50.2 | 41.6 | 49.8% | aten::_scaled_dot_product_fused_attention_overrideable(32.7%), micro_sdpa(32.7%), aten::copy_(5.8%) |

## 说明与结论

- **去噪阶段耗时含一次性的模型构建**：blockwise 首次去噪调用时把 22B bf16 权重 fp8 量化到显存（约 19 s，纯加载/量化、GT 利用率低）；baseline 是流式加载、构建与首个 step 重叠。所以上表 stage1 的 `加速` 列低估了 blockwise 的优势——按去噪**循环本身**的 s/it 计（见 tqdm/`results.txt`），blockwise 8 步循环 ≈1.3 s/it，baseline ≈3–5 s/it（约 2.6–3.8x）。
- **GT 忙% 对亚秒级阶段近似**：GT 的 C6 驻留计数器粒度较粗，`条件编码` 这类 <1 s 阶段可能同时出现 核忙% > GT忙% 的读数（属测量噪声，取整体趋势即可）。
- **blockwise 文本编码显存 98%**：blockwise（无 offload）把 Gemma 12B bf16 整模型驻留显存（25.2 GB / 25.7 GB），且其后 fp8 权重常驻约 22 GB；baseline 的 Gemma 走流式，峰值仅 8.5 GB。
- **VAE 阶段的算力天花板差距**：DiffVAE 解码受 tile 化 + 顺序块 + 宿主端间隙限制，GT 忙约 45–55%、核忙约 42–44%，**空闲差距 ~45–57%**；且本机未装 `natten`，DiffVAE 邻域注意力走了 Triton/SDPA 回退（`micro_sdpa` 占核时 33%），属于可优化点。VAE 其余阶段为带宽受限的卷积+拷贝（`Memcpy M2D`/`copy_`占大头），其算力本就远低于 bf16 GEMM 峰值，优化方向是减少 H2D 拷贝与宿主端同步、安装 natten。
