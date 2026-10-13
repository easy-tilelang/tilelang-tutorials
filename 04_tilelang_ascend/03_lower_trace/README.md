# Ascend A5：用 lower-trace 观察 vector add 的编译过程

本例计算一维向量 `C = A + B`，在真实的 NPU 编译和执行过程中记录每个 lowering pass 前后的 IR，最后与 PyTorch NPU 的计算结果比较。

交互式版本见 [lower_trace_demo.ipynb](lower_trace_demo.ipynb)。三个代码单元格分别演示 `html`、`terminal`、`both`，每格包含完整配置、算子定义和验证，重启内核后可任选一格运行，也支持乱序、重复执行。请选择具备下述 A5 环境的 Python 内核。

Notebook 使用 `enable()` 显式配置 tracing，并在 `finally` 中关闭；每格禁用缓存并新建 JIT 工厂。输出位于当前工作目录的 `tmp/lower_trace_notebook/<模式>`，单元格会打印实际路径。在本教程目录启动 Notebook 时，运行产物会被本目录的忽略规则覆盖。Notebook 通过 `codegen_output=None` 关闭可编辑的顶层源码工作副本，仍保留报告中的 codegen 记录。执行后缓存保持禁用，完成调试可调用 `tilelang.enable_cache()` 恢复。

## 1. 准备环境与运行

需要 Ascend A5 NPU、匹配的 CANN 工具链，以及支持 A5 接口和 `TL_LOWER_TRACE` 的 TileLang 版本。Python 环境中需要安装 `torch` 和兼容的 `torch_npu`。请先加载本机 CANN 的环境配置，并确认 NPU 可用。

从仓库根目录运行：

```bash
python 04_tilelang_ascend/03_lower_trace/01_vector_add.py
```

脚本在 `import tilelang` 前默认设置：

```python
os.environ.setdefault("TL_LOWER_TRACE", "html")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
```

`html` 模式生成报告，也会打印每个 pass 的状态摘要。禁用缓存确保本次调用实际触发编译，否则缓存命中可能跳过 lowering。环境变量采用 `setdefault`，因此命令行传入的配置优先；复现教程时请保持缓存关闭。

运行成功后，会打印 `Test passed: C = A + B ...` 和报告路径。默认报告位于本教程目录下：

```text
tmp/lower_trace_dir/vector_add/
├── report.html                  # 指向最近一次编译的报告
├── codegen.cpp                  # 最终生成的源码
├── codegen.cpp.original         # 源码基线
├── codegen.cpp.latest           # 最近一次生成的源码
└── .run_records/run_.../
    ├── report.html              # 该次编译的独立报告
    ├── pipeline_ascend/         # 各 pass 前后的 IR
    ├── unscoped/                # 已标记 pipeline 范围之外的 pass
    └── codegen/                 # codegen 前的 TIR 和生成的源码
```

默认输出目录由脚本位置确定，不随工作目录变化。直接用浏览器打开 `report.html`；它是自包含的 HTML 文件，不需要启动服务。需要保留某次报告时，打开对应的 `.run_records/run_.../report.html`。

## 2. vector add 做了什么

输入为两个长度 `2**20` 的 `float32` 向量。64 个 block 各自处理大小为 8192 的 tile，每个 block 执行两次迭代：

1. `T.copy` 把 A、B 的当前 tile 从全局内存搬到共享缓冲区。
2. `T.SimdVF()` 内的 `T.Parallel` 描述逐元素加法。
3. `T.copy` 把结果搬回 C。

`T.Pipelined(..., num_stages=2)` 与 `T.annotate_buffer_versions` 描述两级流水和缓冲区版本。编译器需要将这些高层表达转换为设备上的调度、同步、内存访问和向量计算。

对应的地址是 `begin = (step * 64 + bx) * 8192`。示例要求长度为 `64 * 8192` 的正整数倍，避免引入尾块处理；这也是 `vector_add(n)` 的参数检查条件。

## 3. 怎样阅读报告

先找到 `pipeline_ascend` 阶段，再沿 pass 导航查看前后的 IR：

| 观察点                           | 可以关注的变化              |
| -------------------------------- | --------------------------- |
| `AscendLayoutInference`        | 缓冲区和计算的布局推导      |
| `AutoSchedule`、`AssignCore` | 搬运与计算的调度、core 分配 |
| `InsertSync`                   | 搬运与计算之间的同步        |
| `MaterializeMultiBuffer`       | 流水缓冲区版本的具体表示    |
| `AscendLowerTileOp`            | 高层操作向底层调用的转换    |
| `FlattenBuffer`                | 缓冲区索引的展平            |
| `codegen`                      | 最终 TIR 向生成源码的转换   |

这些名称用于定位观察点，具体 pass 顺序、编号及是否改变 IR，以当前 TileLang 版本的报告为准。

- `CHANGED`：前后 IR 文本不同；可以展开 diff 查看增删内容。
- `NO-OP`：前后 IR 文本相同，表示本例没有可见文本变化，pass 仍然执行过。
- `FAILED`：pass 未正常完成；检查异常和已经记录的前置步骤。
- `unscoped`：该 pass 不在已标记的 pipeline 阶段内。

HTML 以两侧 IR 展示差异，也能导航到未改变 IR 的步骤。原始文件按阶段保存，例如 `XX_Simplify_before.tir` 和 `XX_Simplify_after.tir`；`XX` 是本次编译中的序号。嵌套 pass 的终端完成日志可能不按序号排列，报告按进入顺序组织。

报告会随着 pass 完成逐步更新，编译异常时也可查看已记录的部分。这里追踪的是编译过程；脚本最后的 `assert_close` 则验证算子运行结果。

一次编译可能包含设备端和 host 端的多个 codegen 事件。查看 vector add 的设备源码时，优先使用报告中对应的 codegen 记录或 `.run_records/.../codegen/*_after.cpp`；顶层 `codegen.cpp` 可能被后续 host codegen 更新。

## 4. 切换输出模式

以下命令都从仓库根目录执行：

```bash
# 终端状态摘要和 IR diff，不生成本次 HTML 报告
TL_LOWER_TRACE=terminal python 04_tilelang_ascend/03_lower_trace/01_vector_add.py

# 终端 diff 和 HTML 报告
TL_LOWER_TRACE=both python 04_tilelang_ascend/03_lower_trace/01_vector_add.py

# 关闭追踪，仍运行并验证算子
TL_LOWER_TRACE=0 python 04_tilelang_ascend/03_lower_trace/01_vector_add.py

# 自定义 trace 输出位置
TL_LOWER_TRACE_DIR=/tmp/a5-add-trace python 04_tilelang_ascend/03_lower_trace/01_vector_add.py
```

`1`、`on`、`true`、`yes` 等价于 `html`。`terminal` 模式仍会保存原始 IR 文件；已有 HTML 报告不会因为切换模式而删除。自定义的相对输出路径相对于当前工作目录解析。

若 TileLang 已经导入，例如在 Notebook 中，应通过 API 开启，而不是只修改环境变量：

```python
from tilelang.tools import lower_trace as lt

lt.enable(mode="html", trace_dir="./tmp/lower_trace_dir")
try:
    compiled = vector_add(N)  # 需要实际触发编译，避免已有 JIT/磁盘缓存命中
finally:
    lt.disable()
```

`enable()` 对后续编译生效，每次实际编译拥有独立 session；`disable()` 停止追踪后续编译，已开始的 session 会完成其报告。

本教程聚焦查看生成结果。`codegen.cpp` 同时是 lower-trace 的可编辑工作副本，支持的后端可能在下次编译中注入其中的修改。阅读报告时保持它不变；源码编辑重编译不属于本教程的步骤。
