# Ascend A5：`num_blocks` 怎么传 —— blockDim 波次模型与调优案例

[← 返回 Ascend 性能优化案例合集](../README.md)

`T.Kernel(num_blocks)` 里的 `num_blocks` 到底是什么？该传多少？和硬件 AIV 核数有什么关系？

本案例用四个可复现实验回答这三个问题，结论是：

1. `num_blocks` 就是下发给 NPU 的 **blockDim**，即这次启动要跑多少个 block（每个 block 是一份 SPMD 程序，`bx` 是它的编号）。
2. **超过核数不会报错**：block 会按"波"（wave）排队上核，波宽 = 该 kernel 类型对应的物理核数。
3. **最优取值 = 波宽**（纯向量 kernel 取 AIV 核数，含 `T.gemm` 的取 AIC 核数）；比核数多出一点点（"零头波"）是最差的选择。

在本机 Ascend950DT（28 AIC / 56 AIV）上，TileLang 官方 Ascend 例子硬编码的常量正好都落在零头波上：
`example_vecadd.py` 的 `num_blocks=64` 比 `56` 慢 **1.45×**，`example_gemm.py` 的 `NUM_BLOCKS=32` 比 `28` 慢 **1.61×**。

## 0. 结论速览

| kernel 内容 | 生成码声明 | 波宽（一波最多几个 block） | 最优 `num_blocks` | 本机实测 |
| --- | --- | --- | --- | --- |
| 只有搬运 / 向量计算（无 `T.gemm`） | `__global__ __vector__` | AIV 核数 = 56 | 56 | vecadd 2906 GB/s |
| 含 `T.gemm`（Cube + Vector） | `__global__ __mix__(1, 2)` | AIC 核数 = 28 | 28 | GEMM 329.5 TFLOPS |
| 只有 Cube | `__global__ __cube__` | AIC 核数 = 28 | 28 | — |

零头波代价（同样总工作量下的相对吞吐，括号内是实际 `num_blocks`）：

| 配置 | vecadd SIMT（波宽 56） | GEMM（波宽 28） |
| --- | --- | --- |
| 1 × 波宽 | 1.000（56） | 1.000（28） |
| 波宽 + 少量 | 0.688（64） | 0.621（32） |
| 2 × 波宽 | 0.984（112） | 0.985（56） |
| 4 × 波宽 | 0.984（224） | 0.886（112） |
| 8 × 波宽 | 0.978（448） | 未测 |

## 1. 运行方式

环境要求和计时约定见[合集索引](../README.md)。本文数据测于 Ascend950DT_9574（AIC=28 / AIV=56）。

所有命令从仓库根目录执行：

```bash
python 04_tilelang_ascend/04_perf/01_num_blocks/01_device_cores.py    # 查核数（约 10 秒）
python 04_tilelang_ascend/04_perf/01_num_blocks/02_blockdim_limit.py  # 超配是否报错（约 2 分钟）
python 04_tilelang_ascend/04_perf/01_num_blocks/03_wave_model.py      # 波次模型，向量 + Cube（约 8 分钟）
python 04_tilelang_ascend/04_perf/01_num_blocks/04_vecadd_sweep.py    # vecadd 带宽扫描（每 mode 约 4 分钟）
python 04_tilelang_ascend/04_perf/01_num_blocks/05_gemm_sweep.py      # GEMM 吞吐扫描（约 6 分钟）
```

| 文件 | 作用 |
| --- | --- |
| `01_device_cores.py` | 三条路径交叉验证 AIC/AIV 核数 |
| `02_blockdim_limit.py` | blockDim 上限与"是否所有 block 都执行" |
| `03_wave_model.py` | 固定每 block 工作量，验证波次模型（`--kernel vector\|gemm\|both`） |
| `04_vecadd_sweep.py` | 真实算子：流水化 vector add 的有效带宽（`--mode simt\|simd\|both`） |
| `05_gemm_sweep.py` | 真实算子：persistent GEMM 的 TFLOPS（`--mnk`、`--blocks`） |
| `../perf_util.py` | 上一级的共用工具：核数查询、中文对齐表格、交错轮次计时（各案例共享一份） |

脚本本身不落盘，只有 TileLang 的 JIT 缓存（`~/.tilelang`）会写入。**每个 `num_blocks` 都是一次独立编译**，
首次运行较慢；上面括号里的时间已包含编译。所有扫描列表默认按本机核数派生，换机器无需改代码，
也可以用 `--blocks 28,56,64` 自定义。

## 2. `T.Kernel` 在 Ascend 上的语义

Ascend 有自己的 launch 方言（`tilelang/ascend/language/kernel.py`），和 CUDA 侧的 `T.Kernel(grid, threads=...)` 不同：

```python
with T.Kernel(num_blocks) as bx:          # 正确：一维 core grid
    ...

with T.Kernel(num_blocks, threads=128):   # 错误：Ascend 方言没有 threads 参数，Python 层直接 TypeError
```

- 只接受**一维** grid，多维在 `tilelang/ascend/language/kernel.py:171` 直接抛 `ValueError`。
- `bx` 是 block 编号，codegen 映射到 AscendC 的 `block_idx`（`src/ascend/codegen/codegen_ascend.cc:3041`）。
- 线程域在 kernel 体内单独声明：`T.SimtVF(threads=...)`（真实 threadIdx 域）或 `T.SimdVF()`（寄存器级向量，无线程概念）。
  子核号 `sid` 需要 `T.MixedKernel(n, sids=2) as (bx, sid)` 才能拿到。
- `T.alloc_shared` 是**每核各自的 UB**：`num_blocks` 不会成倍占用片上内存，但 `tile × 缓冲版本数` 必须放得进单核 UB。

`num_blocks` 一路传到 launch：

| 阶段 | 位置 | 做了什么 |
| --- | --- | --- |
| 前端 | `tilelang/ascend/language/kernel.py:139` | 校验一维 grid，构造 launch frame |
| pipeline | `tilelang/ascend/pipeline.py:44` | `MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False)`：grid 变成 `blockIdx.x` 的 `thread_extent`，kernel 层不生成 threadIdx |
| codegen | `src/ascend/codegen/codegen_ascend.cc:644` | 按体内算子推断 kernel 类型：有 Vector + (Cube 或子核) → `__mix__(1,2)`；只有 Cube → `__cube__`；其余 → `__vector__` |
| runtime | `src/ascend/runtime/ascend_module.cc:455,491` | grid 各维相乘得到 `num_blocks`，作为 `numBlocks` 传给 `aclrtLaunchKernelWithHostArgs` |

也就是说 `num_blocks` **就是 AscendC/ACL 语义下的 blockDim**。TileLang 侧只校验 `> 0` 和 uint32 范围，
不和核数比较；核数相关的行为完全由运行时决定 —— 这正是下面要测的。

## 3. 第一步：确认本机核数（`01_device_cores.py`）

```text
device            : Ascend950DT_9574
tilelang          : 0.1.15+cuda.git51bdf546
torch / torch_npu : 2.12.0+cpu / 2.12.0.post2

查询方式                         AIC(cube)  AIV(vector)
-------------------------------------------------------
torch.npu.get_device_properties         28           56
tilelang carver ascend_driver           28           56
ctypes aclGetDeviceCapability           28           56

L2 cache          : 112 MB (ACL 读数 112 MB)
HBM               : 80 GB
```

三条路径读数一致：`torch_npu` 内部就是用 `aclGetDeviceCapability(deviceid, ACL_DEVICE_INFO_AI_CORE_NUM / _VECTOR_CORE_NUM, ...)`
填的字段（枚举定义见 CANN 头文件 `include/acl/acl_base_rt.h`），也就是运行时做核数校验时用的同一个来源。
仓库里另有 notebook 版本：[00_basic/02_device_info_npu.ipynb](../../../00_basic/02_device_info_npu.ipynb)。

注意 AIV = 2 × AIC：`__mix__(1, 2)` 的一个 block 对应 1 个 AIC + 2 个 AIV 子核。
**不要把 AIV 数当成 GEMM kernel 的波宽。**

## 4. 实验一：blockDim 超过核数会报错吗（`02_blockdim_limit.py`）

每个 block 只写自己那一格 `C[bx] = A[bx] + B[bx]`，任何 block 没执行都会留下缺项，
因此"结果全对"就等价于"所有 block 都跑了"：

```text
生成码声明：extern "C" __global__ __vector__ void kernel_kernel(__gm__ float* A, __gm__ float* B, __gm__ float* C) {

blockDim  结果  详情              结论
---------------------------------------------------
       1  成功  错误元素 0/1      全部 block 均执行
      28  成功  错误元素 0/28     全部 block 均执行
      56  成功  错误元素 0/56     全部 block 均执行
      57  成功  错误元素 0/57     全部 block 均执行
     112  成功  错误元素 0/112    全部 block 均执行
     560  成功  错误元素 0/560    全部 block 均执行
   65535  成功  错误元素 0/65535  全部 block 均执行
```

**结论：不报错。** `T.Kernel(560)` 在 56 AIV 的机器上完全合法，560 个 block 都会执行完。
真正的硬上界是 uint16：`--overflow` 会额外测 65536，它在设备侧同步时失败
（`ACL stream synchronize failed, error code:507035` + `vector core execution is abnormal`），
并让当前进程的 NPU 上下文不可用，所以默认不测、且必须放在最后。

## 5. 实验二：波次模型（`03_wave_model.py`）

让**每个 block 做固定的一份工作**（工作量与 blockDim 无关），于是延迟直接反映并发度。

向量 kernel（每 block 在 16 KB UB 上做 200 轮 SIMT 计算）：

```text
=== vector kernel（生成码 __vector__，波宽 = 56）===

blockDim  延迟中位数(us)    min~max(us)  预测波数  实测/单波
------------------------------------------------------------
      14          110.91  110.26~113.22         1       0.99
      28          113.00  112.33~113.63         1       1.01
      56          111.58  111.36~113.11         1       1.00
      84          210.78  210.38~211.23         2       1.89
     112          213.33  212.84~213.50         2       1.91
     168          315.60  314.09~316.23         3       2.83
     224          416.13  415.72~419.15         4       3.73
```

Cube kernel（每 block 做 200 次 256³ `T.gemm` + `dual_copy` 写出）：

```text
=== gemm kernel（生成码 __mix__(1, 2)，波宽 = 28）===

blockDim  延迟中位数(us)      min~max(us)  预测波数  实测/单波
--------------------------------------------------------------
       7          511.13    510.45~534.06         1       1.00
      14          511.69    508.91~511.82         1       1.00
      28          510.19    510.09~510.37         1       1.00
      42         1032.42  1023.86~1037.42         2       2.02
      56         1025.84  1017.63~1036.30         2       2.01
      84         1523.13  1516.84~1524.02         3       2.99
     112         2035.00  2020.04~2038.64         4       3.99
```

两组数据都严格贴合：

```text
延迟 ≈ ceil(num_blocks / 波宽) × 单波延迟
```

- blockDim ≤ 波宽：延迟完全不变（向量 kernel 的 14/28/56 都是 ~111 us，Cube kernel 的 7/14/28 都是 ~510 us）
  → 一核一 block，真并发。
- 超过波宽：按整数倍增长，`ceil(nb / 波宽)` 波，前一波做完才轮到下一波。
- 向量 kernel 的波宽是 **AIV = 56**，含 `T.gemm` 的是 **AIC = 28**（56 个 block 的 Cube kernel 要跑 2 波，
  正好是 28 的 2 倍时间；反过来向量 kernel 跑 56 个 block 只要 1 波）。

所以 `bx` 在超配时**不等于物理核号**，只是 block 编号；同一波内的 block 才真正同时在核上跑。
由此可以推断（本案例未实测）：**跨 block 的自旋等待在超配时会死锁** —— 后一波的 block 还没上核，
不可能去释放前一波在等的标志位。需要全局协作的写法必须保证 `num_blocks ≤ 波宽`。

## 6. 实验三：真实算子上的带宽（`04_vecadd_sweep.py`）

kernel 与官方 `examples/ascend/example_vecadd.py` 一致（TILE=8192、`num_stages=2`、`SimtVF(threads=2048)`），
只把 `num_blocks` 参数化；每个配置都跑约 1 GiB/张量，保证总字节数相同：

```text
=== mode=simt（纯向量 kernel，波宽 = AIV = 56）===

num_blocks  每核迭代  单张量(MB)  波数  末波核数  带宽中位数(GB/s)    min~max  /最优
------------------------------------------------------------------------------------
        14      2341        1024     1     14/56           1024.15  1023~1025  0.352
        28      1170        1024     1     28/56           2031.22  2021~2039  0.699
        56       585        1024     1     56/56           2905.97  2905~2930  1.000
        64       512        1024     2      8/56           1999.54  1997~2000  0.688
       112       293        1026     2     56/56           2859.57  2859~2867  0.984
       224       146        1022     4     56/56           2860.08  2853~2867  0.984
       448        73        1022     8     56/56           2840.86  2828~2842  0.978
```

`--mode simd`（同一份数据划分，计算换成显式 `T.simd.*` 指令）：

```text
num_blocks  每核迭代  单张量(MB)  波数  末波核数  带宽中位数(GB/s)    min~max  /最优
------------------------------------------------------------------------------------
        14      2341        1024     1     14/56           1691.39  1473~1698  0.571
        28      1170        1024     1     28/56           2643.57  2639~2655  0.892
        56       585        1024     1     56/56           2962.48  2952~2963  1.000
        64       512        1024     2      8/56           2508.10  2508~2551  0.847
        112       293        1026     2     56/56           2916.37  2914~2917  0.984
       224       146        1022     4     56/56           2905.17  2898~2910  0.981
       448        73        1022     8     56/56           2923.66  2919~2925  0.987
```

三点观察：

1. **单波内带宽随核数近似线性，然后被 HBM 卡住**：14 → 1024，28 → 2031，56 → 2906 GB/s（SIMT），
   再加 block 也不再涨（112/224/448 都在 2840~2920）。这台机器的访存上限约 2.9 TB/s。
2. **零头波是纯浪费**：`num_blocks=64` 的波次构成是"56 个核的一整波 + 8 个核的一波"。
   第二波仍然要跑完整的每核迭代数（512 次），却只有 8 个核在干活，
   于是总时间 ≈ 2 波，而有效并行度只有 (56+8)/2 = 32 核 → 掉到 0.688。
   SIMD 模式掉得少一些（0.847），因为它的第二波更快撞上带宽上限。
3. **整数倍超配基本无损**（112/224/448 → 0.98），但也不比 `num_blocks=56` 更好，
   只是多了编译期 tile 划分和波次切换的开销。想要更多逻辑分块时用整数倍，或者干脆用核内循环。

## 7. 实验四：Cube kernel 的吞吐（`05_gemm_sweep.py`）

persistent GEMM（`C = X @ W.T`，bf16 → fp32，tile 256³，`T.Persistent` 分 tile，`dual_copy` 写回），
M=N=K=4096 → 256 个输出 tile：

```text
num_blocks  每核tile  波数  末波核数  延迟(ms)  TFLOPS  /最优  模型估算  max_err
--------------------------------------------------------------------------------
         7        37     1      7/28     1.516    90.7  0.275     0.247     0.00
        14        19     1     14/28     0.786   174.8  0.530     0.481     0.00
        28        10     1     28/28     0.417   329.5  1.000     0.914     0.00
        32         8     2      4/28     0.672   204.5  0.621     0.571     0.00
        56         5     2     28/28     0.423   324.6  0.985     0.914     0.00
       112         3     4     28/28     0.471   292.0  0.886     0.762     0.00
```

`模型估算 = (out_tiles / AIC) / (波数 × 每核 tile 数)`，只按波次和 tile 轮数推算，忽略 L2 复用、
流水充排和末波带宽差异。它和实测同趋势（32 → 0.571 预测 / 0.621 实测；7 → 0.247 / 0.275），
说明"波数 × 每核 tile 轮数"是主要的决定因素，适合做首轮配置的纸面估算。

同样是零头波最坑：`num_blocks=32` 只比 AIC 多 4 个，却把 256 个 tile 拆成
"28 个 block × 8 tile" + "4 个 block × 8 tile"两波 → 16 个 tile 轮次，而满载只需 256/28 ≈ 9.14 轮。
`num_blocks=56`（2 整波，每 block 5 tile → 10 轮）反而几乎无损。

## 8. 调优规则：`num_blocks` 该怎么传

1. **先判断 kernel 类型，再取核数**：体内没有 `T.gemm` → 波宽是 AIV 核数；有 `T.gemm` → 波宽是 AIC 核数。
   拿不准就打印 `tilelang.compile(...).get_kernel_source()`，看声明是 `__vector__` 还是 `__mix__(1, 2)`。
2. **默认取波宽**：`num_blocks = 核数`，一波打满，没有零头。
3. **tile 数少于核数时取小值**：`num_blocks = min(核数, tile 数)`，多出来的 block 只会空转
   （官方 deepgemm 就是这么做的：`examples/ascend/deepgemm/config.py:112`）。
4. **不要传"核数 + 一点点"**：`波宽 + k`（k 小）会多出一整波却只用 k 个核。
   实测损失：SIMT vecadd（56+8）31%、GEMM（28+4）38%、SIMD vecadd（56+8）15%。
   要么正好一波，要么取整数倍。
5. **工作量比核数多，用核内循环，不是加大 `num_blocks`**：
   `T.Persistent([tile 数], num_blocks, bx)` 或自己写 `for step in T.Pipelined(iterations, ...)`，
   让每个 block 顺序吃掉多个 tile。超配虽然不报错，但没有任何吞吐收益。
6. **需要跨 block 协作（自旋等待、全局 barrier）时必须 `num_blocks ≤ 波宽`**：
   后一波的 block 尚未上核，等待永远不会被满足（由波次模型推断，见 §5）。
7. **动态取核数**，别把常量写死在算子里：

```python
import tilelang.carver.arch.driver.ascend_driver as ad

def pick_num_blocks(has_gemm: bool, num_tiles: int) -> int:
    cores = ad.get_num_cube_cores() if has_gemm else ad.get_num_vector_cores()
    return max(1, min(cores, num_tiles))
```

## 9. 对官方 Ascend 例子的影响（tilelang `51bdf546`）

例子里的常量是按 **32 AIC / 64 AIV** 的机器调的，在 28/56 的 Ascend950DT 上全部落在零头波：

| 位置 | 现值 | 本机应为 | 影响 |
| --- | --- | --- | --- |
| `examples/ascend/example_vecadd.py:12` | `num_blocks = 64` | 56 | **实测** 2000 → 2906 GB/s（1.45×） |
| `examples/ascend/example_gemm.py:20` | `NUM_BLOCKS = 32` | 28 | **实测** 204.5 → 329.5 TFLOPS（1.61×，同结构 persistent GEMM） |
| `examples/ascend/example_rmsnorm.py:14` | `N_CORES = 64` | 56 | 未实测；同为纯向量 kernel，预期同量级 |
| `examples/ascend/example_per_token_cast_to_fp8.py:33,93` | `64` / `min(64, num_tiles)` | 56 | 未实测；同上 |
| `examples/ascend/example_gemm_splitk.py:17` | `NUM_BLOCKS = 16` | 28（需 `split_k` 整除） | 未实测；16 < 28，单波占不满 AIC |
| `examples/ascend/deepgemm/config.py:112` | `min(num_cores, num_tiles)` | — | 已经是动态取值，无需改 |

改的时候注意耦合约束：`example_vecadd.py:15` 要求 `N % (tile * num_blocks) == 0`，
`example_rmsnorm.py:21` 用 `N_CORES` 反推每核行数。把 `num_blocks` 改成按设备取值后，
整除条件也变成设备相关，需要同时调整 tile 大小或补尾块处理，否则会直接抛 `ValueError`。

## 10. 本案例的复现注意点

通用测量约定（正确性优先、交错轮次取中位数、总工作量对齐、记录环境）见[合集索引](../README.md)。
本案例特有的几点：

- **离群轮次真实存在**：采集时出现过约 330 GB/s 的轮次（共享设备被别的负载干扰），比中位数低一个量级。
  所以表里一律给中位数并附 `min~max`，只看单次或最小值会得出错误结论。
- **总工作量对齐**：扫描 `num_blocks` 时按 `N = TILE × num_blocks × iterations` 反推规模，
  各配置搬运/计算同样的字节数或 FLOPs，带宽和 TFLOPS 才可比。
- **模型只做趋势判断**：`05_gemm_sweep.py` 的"模型估算"列忽略 L2 复用和流水充排，性能结论一律以实测为准。
- JIT 缓存命中会跳过编译；对比不同 TileLang 版本前先清 `~/.tilelang`。
- `04_vecadd_sweep.py` 默认约 3 GiB 显存峰值（`--target` 可调小）。
- `02_blockdim_limit.py --overflow` 会让设备报 vector core exception，本进程的 NPU 上下文随后失效，需要重启进程。

## 11. 参考

- TileLang Ascend 后端说明：`tilelang/ascend/README.md`（`T.Kernel` / `T.SimtVF` / `T.dual_copy` / 自动调度分工）
- Ascend 编程技能文档：`.agents/skills/tilelang-ascend/SKILL.md`（含常见坑：`threads=` 不能传给 `T.Kernel`、只支持一维 grid）
- 官方示例：`examples/ascend/example_vecadd.py`、`example_gemm.py`、`deepgemm/`
- 本仓库相关教程：[03_lower_trace](../../03_lower_trace/README.md)（看编译过程）、
  [00_basic/02_device_info_npu.ipynb](../../../00_basic/02_device_info_npu.ipynb)（设备信息）
- CANN 接口：`aclrtLaunchKernelWithHostArgs`、`aclGetDeviceCapability`（`include/acl/acl_rt.h`、`include/acl/acl_base_rt.h`）
