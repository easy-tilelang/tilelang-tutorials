# Ascend A5：性能优化案例合集

每个案例一个子目录，自带 `README.md`（结论 + 实测数据）和可复现脚本。
所有案例都遵循同一套测量约定，脚本共用本目录的 `perf_util.py`。

## 案例索引

| 编号 | 主题 | 一句话结论 | 脚本 | 全跑耗时 |
| --- | --- | --- | --- | --- |
| [01_num_blocks](01_num_blocks/README.md) | `T.Kernel(num_blocks)` 的物理含义与取值 | `num_blocks` 就是 blockDim；超过核数不报错而是**分波执行**，波宽 = 该 kernel 类型的核数（纯向量 → AIV，含 `T.gemm` → AIC）；最优取波宽，"核数 + 一点点"的零头波最差（实测损失 15%~38%） | 5 个 | 约 20 分钟 |

## 环境要求

- Ascend A5 NPU 与匹配的 CANN 工具链，`torch.npu.is_available()` 返回 `True`。
- 可运行的 TileLang Ascend 后端（`import tilelang.ascend.language as T` 不报错）。
- 从**仓库根目录**运行脚本，例如：

```bash
python 04_tilelang_ascend/04_perf/01_num_blocks/01_device_cores.py
```

本合集现有数据采集环境（换设备或换 build 都需要重测）：

```text
Ascend950DT_9574 | AIC=28 AIV=56 | L2 112 MB | HBM 80 GB
CANN toolkit 9.3.0 (V100R001C12B136) | torch 2.12.0 | torch_npu 2.12.0.post2
tilelang 0.1.15+cuda.git51bdf546
```

先用 `01_num_blocks/01_device_cores.py` 确认本机核数：它会用 torch_npu、TileLang carver、
直接 ctypes 调 ACL 三条路径交叉验证 AIC/AIV 读数。各案例脚本的默认扫描列表都按本机核数派生，
换机器无需改代码，也可以用 `--blocks` 自定义。

## 共用工具 `perf_util.py`

| 函数 | 作用 |
| --- | --- |
| `require_npu()` | 无 NPU 时直接退出，给出明确提示 |
| `core_counts()` | 返回 `(AIC 核数, AIV 核数)` |
| `device_line()` | 一行环境标识：设备名、核数、tilelang / torch_npu 版本 |
| `print_table(...)` | 按显示宽度对齐的表格打印（中英文混排不错位） |
| `bench(call, reps, rounds)` | 交错轮次计时，返回 `(中位数, 最小值, 最大值)` |

它位于案例目录的上一级，脚本这样导入：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 共用工具在上一级 04_perf/
import perf_util  # noqa: E402
```

## 测量约定

各案例共用以下几条，案例 README 只补充自己特有的注意点：

1. **先正确性，再性能**：每个配置都先和 PyTorch 结果比对，不通过直接退出，绝不把错误结果当基线。
2. **交错轮次 + 中位数**：`perf_util.bench` 跑 `rounds × reps` 取中位数，并打印 `min~max`。
   共享设备上偶发的其他负载会把个别轮次拖慢一个量级，只看单次或最小值会得出错误结论。
3. **总工作量对齐**：扫描配置时让每个点搬运同样的字节数 / 计算同样的 FLOPs，带宽和 TFLOPS 才可比。
4. **区分模型与实测**：纸面估算只用于判断趋势，性能结论一律以实测为准。
5. **记录环境**：每个脚本第一行打印 `device_line()`；数据必须带上设备名、核数和版本才能复现或反驳。

脚本本身不落盘，只有 TileLang 的 JIT 缓存（`~/.tilelang`）会写入。首次运行较慢，因为每个配置都是一次独立编译；
缓存命中后重跑只剩执行时间。对比不同 TileLang 版本前先清缓存。

## 新增案例的约定

1. 目录命名 `NN_主题/`（`NN` 两位递增，主题用英文小写下划线），例如 `02_tile_size/`。
2. 目录内自带 `README.md`：结论速览 → 原理/源码路径 → 实验与实测表格 → 调优规则 → 复现注意点。
   顶部加一行返回本文件的链接：

   ```markdown
   [← 返回 Ascend 性能优化案例合集](../README.md)
   ```

3. 脚本命名 `0M_用途.py`，可独立运行、带 `--help`，默认参数在本机几分钟内跑完；共用逻辑放 `../perf_util.py`，
   不要在案例目录里复制一份。
4. 数据必须实测，标注设备与版本；推断而非实测的结论要显式写明"未实测"。
5. 更新本文件的案例索引表；仓库根 `README.md` 只链接本合集，不需要逐案例追加。

## 相关教程

- [03_lower_trace](../03_lower_trace/README.md)：用 lower-trace 观察编译过程（调度、同步、多缓冲在哪个 pass 落地）
- [00_basic/02_device_info_npu.ipynb](../../00_basic/02_device_info_npu.ipynb)：设备信息查询
- TileLang 仓库：`tilelang/ascend/README.md`、`.agents/skills/tilelang-ascend/SKILL.md`、`examples/ascend/`
