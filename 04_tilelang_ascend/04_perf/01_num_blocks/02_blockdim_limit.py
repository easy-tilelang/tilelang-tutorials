"""实验一：blockDim（num_blocks）超过核数会怎样？

每个 block 只写自己那一格 ``C[bx] = A[bx] + B[bx]``，所以只要有 block 没执行，
结果就会缺项 —— 用这个性质验证“所有 block 是否真的都跑了”。

从仓库根目录运行：

    python 04_tilelang_ascend/04_perf/01_num_blocks/02_blockdim_limit.py
    python 04_tilelang_ascend/04_perf/01_num_blocks/02_blockdim_limit.py --blocks 1,56,57,65535,65536
"""

import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 共用工具在上一级 04_perf/
import perf_util  # noqa: E402


def scalar_add(num_blocks):
    @T.prim_func
    def kernel(
        A: T.Tensor((num_blocks,), T.float32),  # type: ignore
        B: T.Tensor((num_blocks,), T.float32),  # type: ignore
        C: T.Tensor((num_blocks,), T.float32),  # type: ignore
    ):
        with T.Kernel(num_blocks) as bx:
            C[bx] = A[bx] + B[bx]

    return kernel


def probe(num_blocks, show_source=False):
    torch.manual_seed(0)
    a = torch.randn(num_blocks, dtype=torch.float32)
    b = torch.randn(num_blocks, dtype=torch.float32)
    try:
        kernel = tilelang.compile(scalar_add(num_blocks), out_idx=[-1])
    except Exception as e:  # 编译期失败
        return f"{num_blocks}", "编译失败", f"{type(e).__name__}: {str(e).splitlines()[0][:80]}", ""
    if show_source:
        source = kernel.get_kernel_source()
        decl = next(line for line in source.splitlines() if "__global__" in line)
        print(f"生成码声明：{decl.strip()}")
        print()
    try:
        c = kernel(a.npu(), b.npu()).cpu()
    except Exception as e:  # 启动 / 运行期失败
        return f"{num_blocks}", "启动失败", f"{type(e).__name__}: {str(e).splitlines()[0][:80]}", ""
    bad = int((~torch.isclose(c, a + b)).sum())
    return f"{num_blocks}", "成功", f"错误元素 {bad}/{num_blocks}", "全部 block 均执行" if bad == 0 else "有 block 未执行"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blocks", default=None, help="逗号分隔的 blockDim 列表，默认覆盖核数上下与 uint16 上界")
    parser.add_argument(
        "--overflow",
        action="store_true",
        help="额外测 65536（uint16 越界）。会让设备报 vector core exception，本进程后续不可用，放在最后测",
    )
    args = parser.parse_args()

    perf_util.require_npu()
    _, aiv = perf_util.core_counts()
    print(perf_util.device_line())
    print()

    if args.blocks:
        blocks = [int(x) for x in args.blocks.split(",")]
    else:
        blocks = sorted({1, aiv // 2, aiv, aiv + 1, 2 * aiv, 10 * aiv, 560, 65535})
        if args.overflow:
            blocks.append(65536)

    rows = [list(probe(nb, show_source=(i == 0))) for i, nb in enumerate(blocks)]
    perf_util.print_table(["blockDim", "结果", "详情", "结论"], rows, aligns=[">", "<", "<", "<"])
    print()
    print(f"说明：本机 AIV={aiv}。超过核数的 blockDim 不报错，block 会分波（wave）执行，见 03_wave_model.py；")
    print("      驱动里 blockDim 是 uint16，65535 是最后一个合法值，65536 越界（--overflow 才测）。")


if __name__ == "__main__":
    main()
