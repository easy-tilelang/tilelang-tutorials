"""本目录（04_perf）各性能案例共用的小工具：设备核数、中文对齐表格、计时统计。

案例脚本位于 ``04_perf/NN_xxx/`` 子目录，以
``python 04_tilelang_ascend/04_perf/NN_xxx/0M_yyy.py`` 的方式从仓库根目录运行。
本模块在案例目录的上一级，脚本用如下引导导入（仓库无 lint 配置，E402 只是习惯标注）::

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import perf_util  # noqa: E402
"""

from __future__ import annotations

import statistics
import time
import unicodedata

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

import tilelang
import tilelang.carver.arch.driver.ascend_driver as ad


def require_npu() -> None:
    if not torch.npu.is_available():
        raise SystemExit("未发现可用的 NPU，请在 Ascend 环境中运行本教程。")


def core_counts() -> tuple[int, int]:
    """返回 (AIC 核数, AIV 核数)。"""
    return ad.get_num_cube_cores(), ad.get_num_vector_cores()


def device_line() -> str:
    props = torch.npu.get_device_properties(0)
    cube, vector = core_counts()
    return f"{props.name} | AIC={cube} AIV={vector} | tilelang {tilelang.__version__} | torch_npu {torch_npu.__version__}"


def width(text: str) -> int:
    """终端显示宽度：东亚全角字符按 2 列计。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text: str, columns: int, align: str = ">") -> str:
    text = str(text)
    space = " " * max(0, columns - width(text))
    return space + text if align == ">" else text + space


def print_table(headers: list[str], rows: list[list], aligns: list[str] | None = None, gap: str = "  ") -> None:
    """按显示宽度对齐打印表格，避免中英文混排错位。"""
    aligns = aligns or ["<"] + [">"] * (len(headers) - 1)
    cells = [[str(c) for c in row] for row in rows]
    columns = [max(width(h), *(width(r[i]) for r in cells)) if cells else width(h) for i, h in enumerate(headers)]
    line = gap.join(pad(h, columns[i], a) for i, (h, a) in enumerate(zip(headers, aligns)))
    print(line)
    print("-" * width(line))
    for row in cells:
        print(gap.join(pad(c, columns[i], a) for i, (c, a) in enumerate(zip(row, aligns))))


def bench(call, reps: int = 5, rounds: int = 3, warmup: int = 2) -> tuple[float, float, float]:
    """交错轮次计时，返回 (中位数, 最小值, 最大值) 秒。

    共享设备上偶发其他负载会拖慢个别轮次，用中位数而不是单次结果。
    """
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            call()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) / reps)
    return statistics.median(samples), min(samples), max(samples)
