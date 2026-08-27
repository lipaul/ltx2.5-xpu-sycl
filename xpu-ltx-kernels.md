# xpu-ltx-kernels：LTX 推理的 XPU（Intel GPU）移植与加速全过程

> 本文件记录 `packages/xpu-ltx-kernels` 的整个移植、构建与加速过程：从 CUDA 内核到
> SYCL/ESIMD，在 Intel Arc Pro B60（Xe2/Battlemage）上落地，并把实测基准一并写入。
> 配套文件：`packages/xpu-ltx-kernels/docs/capability-report.md`（工具链/硬件探测结论）、
> `AGENTS.md`（agent 速查）。

---

## 1. 目标与背景

参照 `packages/ltx-kernels` 在 CUDA 上的优化（all2all、blockwise FP8/FP6 GEMM、融合元素算子、
VAE 邻域注意力），在 XPU 上用 **ESIMD + SYCL** 移植出一套同等作用的原生 Intel 内核，用于
优化 LTX-2.5 22B 音频视频扩散模型的推理。

机器：20× Intel Arc Pro B60（Xe2/Battlemage，每卡 160 XMX、~24 GB GDDR6）。
工具链：oneAPI **2026.1**（唯一编译器 `icpx`），torch **2.13.0+xpu**（捆绑运行时
`intel-sycl-rt` 2026.0.0，`libsycl.so.9`）。

## 2. 关键硬件发现（决定加速路线的依据）

| 发现 | 影响 |
|---|---|
| **Xe2 XMX 无 FP8 张量核**（官方规格 INT2/4/8、FP16、BF16、TF32） | FP8 只能做*存储*，计算回落 bf16 DPAS；`torch._scaled_mm` 在 XPU 上是 `NotImplementedError` |
| **bf16 GEMM 经 oneDNN 已达 ~95 TFLOPS**（≈XMX 峰值 96%） | 手写 ESIMD DPAS GEMM 无法超越 oneDNN，**不写裸 GEMM** |
| 22B bf16 权重 = 44 GB > 24 GB | 基线必须 `--offload cpu` 逐层流式 → PCIe 瓶颈；fp8 块状存储压到 ~22 GB 可**免 offload 常驻** |
| INT8 XMX ≈ 2× bf16（~200 TFLOPS） | 未来 >bf16 计算的唯一路径（W8A8），本次未做 |

**结论**：XPU 上 blockwise FP8 的价值是**内存**（2× 权重压缩 → 免 offload），不是计算。
计算核心继续用 oneDNN bf16（已到硬件峰值）。

## 3. 工具链配方（最难啃的部分）

torch 的 `cpp_extension` SYCL 路径在本机**不可用**（强制 `-fsycl-host-compiler=c++` 且
头文件顺序与运行时布局冲突）。正确配方：

```bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=level_zero:gpu

# 1) 编译（-x c++ 必须有，否则 .sycl 被当链接输入、静默不编译）
icpx -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen,spir64 -sycl-std=2020 \
  -isystem <torch>/include -isystem <torch>/include/torch/csrc/api/include \
  -isystem <env>/include -isystem <env>/include/sycl \
  -isystem <python-dev>/include/python3.12 \
  -DTORCH_EXTENSION_NAME=_C -x c++ -c ops.sycl -o ops.o

# 2) 设备链接（关键！缺它内核提交即段错误——设备镜像没打包）
icpx -shared -fPIC -fsycl -o _C.so ops.o gemm.o vae.o \
  -L<torch>/lib -ltorch_python -ltorch -ltorch_cpu -lc10 -ltorch_xpu -lc10_xpu \
  -L<oneAPI>/lib -lsycl -Wl,-rpath,<torch>/lib -Wl,-rpath,<oneAPI>/lib
```

**环境版本匹配（血泪教训）**：
- torch 2.12.0+xpu 捆绑运行时 **2025.3.2**（`libsycl.so.8`）；唯一编译器 2026.1 产物要求
  `libsycl.so.9`（2026.x ABI，soname bump = ABI break）。2026.1 编译产物对 2025.3.2 运行时是
  **向后不兼容**方向 → 内核提交段错误。把 torch 指向 2026.1 运行时 → torch 自身 matmul 崩溃。
- 解法：把 uv 的 XPU torch 钉升到 **2.13.0+xpu**（运行时 2026.0.0 = `libsycl.so.9`），与
  icpx 2026.1 一致。这是 fork 的 XPU 钉（`packages/ltx-core/pyproject.toml`），本就有意为之。
- 2026.1 的 ESIMD 头有 N=1 标量数学编译 bug（`fmod/sin_emu/atan2` 的 `reinterpret_cast`/
  `__builtin_convertvector`）；用 env 自带的 2026.0.0 头即可规避，无需打补丁。
- `uv pip install -e packages/xpu-ltx-kernels --python .venv/bin/python --no-build-isolation`
  （uv venv 不带 Python.h，setup.py 用 `sysconfig.get_path('include')` 解析到 base Python）。

## 4. 移植过程（5 个阶段）

### Phase 0 — 能力探针（决定后续路线）
- 实测 bf16 GEMM 吞吐、fp8 matmul 可用性、oneDNN 路径、triton-xpu 可用性、分布式/IPC。
- 产出 `docs/capability-report.md`，据此定下 GEMM 路线（见 §2）。

### Phase 1 — 融合元素算子（`csrc/ops.sycl`，纯 SYCL）
移植 `ltx-kernels/csrc/ops` 的四个算子，`xpu_ltx_kernels.ops`：
- `rms_norm_rope`、`rms_norm_split_rope`：RMSNorm + RoPE 单趟融合（XPU 原为 eager 两趟 op）；
- `fp6_pack`、`fp6_unpack`：FP8→FP6 位压缩（丢 e1/e2，省 25% 权重内存）。

过程中修的 3 个经典 bug（写入 capability-report）：
1. **torch bf16 ≠ `sycl::half`**（`sycl::half` 是 fp16！）——用 `sycl::ext::oneapi::bfloat16`。
2. **网格必须是"每行一个 workgroup"**（`global = rows*wg`），不能把 `global_id` 当行号。
3. **归约用 barrier-tree**：`reduce_over_group` 和局部内存 atomic 在本运行时不可靠；
   且**任何组归约前不得提前 return**（会破坏 subgroup 汇聚，用 clamp 行号 + 掩码 store）。

### Phase 2 — GEMM（`csrc/gemm.sycl` + `xpu_ltx_kernels.gemm`）
- `BlockwiseFP8Linear`：权重 FP8 块状存储（per-128×128 scale，`[out,in]` 平面布局），
  前向先 ESIMD 解量化到 bf16，再走 oneDNN bf16 GEMM。
- 权重内存减半（44→22 GB），计算在 bf16 峰值（见 §6 基准）。
- 设计取舍：Q8 路径传入的 `(fp8, scales)` 激活元组在 forward 里解量化；门控输出保持 bf16
  （不额外量化），与"bf16 GEMM"设计一致。

### Phase 3 — VAE 邻域注意力（`csrc/vae.sycl`，`xpu_ltx_kernels.vae.na3d`）
- 3D neighborhood attention，语义对齐 `natten.na3d` / ltx-core `fallback_na/triton_na`：
  窗口 clamp/内移、`[B,T,H,W,NH,HD]` 布局、scale 在核内应用。
- 现状：DiffVAE 解码默认仍走 eager tiled SDPA 回退（natten 未装）；`na3d` 可作为
  `attention_function` 的 XPU 后端接入（未接入默认路径）。

### Phase 4 — all2all 多卡（`xpu_ltx_kernels.all2all`）
- 头重分配 `send_recv_heads`/`gather_heads` 的功能移植（`[B,S,H,D]→[B,S_total,H/ws,D]`），
  基于 `torch.distributed.all_to_all_single`（CUDA 版用 IPC 零拷贝；XPU 版为 collectives）。

### Phase 5 — ltx-core 集成（让流水线真正用上）
- `ltx_core/quantization/blockwise/_impl.py` 按设备分发内核：CUDA→`ltx_kernels`，
  XPU→`xpu_ltx_kernels` + `_xpu_kernels.py`（Triton-only 算子的 torch 兜底：
  adanorm/rms_fma 量化、门控注意力按 head 的 `2*sigmoid`、stride 敏感的 blockwise 解量化）。
- 移除 `.cuda()` 硬编码（`_blockwise_quantize_weight_helper`、`_replace_linear_modules`、
  fp8 fuse 里的 `merged.cuda()`），改用 `torch.xpu`/设备无关写法。
- CLI 新增 **`--quantization blockwise`**（`quantization_factory.py` + `args.py`）。

## 5. e2e 验证（uv 环境）

- 单元测试：**20/20 通过**（ops 12 + gemm 4 + vae 2 + all2all 2），`uv run ruff` 全绿。
- 集成验证：`build_fp8_policy()` 构建、fp8 块状权重量化/解量化 rel err ≈ 0.06、
  融合 `rms_norm_rope` 端到端。
- 完整流水线：`python -m ltx_pipelines.distilled --quantization blockwise --num-frames 33 ...`
  在 B60 跑通，产出 **h264 1536×1024@24fps + 音频** 视频。
  e2e 中再修两处：`gated_attention` 的按 head 门控广播、`BlockwiseFP8Linear` 的元组激活输入。

## 6. 基准（B60，33 帧 / 768×512 输入，同 prompt/seed）

| 阶段 | 基线 bf16 `--offload cpu` | blockwise fp8（免 offload） | 加速 |
|---|---|---|---|
| Stage 1（8 步，distilled） | ~6.4–7.3 s/it → ~51 s | ~2.4–2.9 s/it → ~19 s | **≈2.6×** |
| Stage 2（3 步） | ~9.9–10.1 s/it → ~30 s | ~7.8–7.9 s/it → ~24 s | **≈1.25×** |
| 去噪合计 | ~81 s | ~43 s | **≈1.9×** |

单线性层对照（M=4096,N=K=5120）：bf16 Linear 2.72 ms vs fp8-Linear 3.09 ms（**0.88×**，
解量化有 ~0.4 ms 开销）——**计算侧 blockwise 并不更快**。

**归因**：主因是内存——44 GB bf16 放不进 24 GB，基线每步 `--offload cpu` 流式取权重
（PCIe-bound）；fp8 块状压到 ~22 GB 免 offload，stage 1 直接省掉流式开销 → 2.6×。
次因是融合 `rms_norm_rope`。注意力后端两条路径相同（SDPA FLASH bf16），不是变量。

**一句话结论**：XPU 上 blockwise 的价值不是计算（无 FP8 张量核、bf16 GEMM 已峰值），
而是**让 22B 在 24 GB B60 免 offload 常驻推理**，端到端约 **1.9×**（stage 1 达 2.6×），
同时省掉 ~36 GB 主机 RAM。

## 7. 局限与后续

- **block streaming 不支持 blockwise policy**（`blocks.py` 只放行 bf16/fp8_cast）——
  要 `--offload` + blockwise 需扩展 `WeightsProvider` 的 fp8 块状布局。
- **INT8 W8A8** 是 B60 上唯一 >bf16 计算的路径（~2×），需自研量化格式（偏离 ltx-kernels 的
  fp8 布局），未做。
- `na3d` 与 all2all 为**功能正确版**，未做性能优化；接入默认路径需改 DiffVAE 的
  `attention_function` 分发与多卡启动脚本。
- fp8→bf16 解量化每前向都有开销（~0.88×），融合进 DPAS GEMM 才能消掉——但 oneDNN bf16 已
  在峰值，无空间。

## 8. 仓库改动记录

- `f62fa47` Add xpu-ltx-kernels: SYCL/ESIMD XPU kernels for LTX inference（全部 5 阶段）
- `4446160` Reproduce under uv: bump XPU torch pins to 2.13.0+xpu（含 gated_attention /
  BlockwiseFP8Linear 两处 e2e 修复 + 基准）
- 远程：`xpu-sycl` → https://github.com/lipaul/ltx2.5-xpu-sycl（main）