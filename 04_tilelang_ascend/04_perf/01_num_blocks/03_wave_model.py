"""实验二：验证“波次（wave）模型”—— blockDim 超过核数后是分批执行的。

每个 block 做**固定**的一份工作（工作量与 blockDim 无关），于是：

- 如果所有 block 真并发，延迟应与 blockDim 无关；
- 如果按波执行，延迟 ≈ ceil(blockDim / 波宽) × 单波延迟。

波宽取决于 kernel 类型（由体内算子推断，见 TileLang 的 Ascend codegen）：

- 纯向量 kernel（无 ``T.gemm``，生成 ``__global__ __vector__``）      -> 波宽 = AIV 核数
- 含 Cube 的 kernel（有 ``T.gemm``，生成 ``__global__ __mix__(1, 2)``）-> 波宽 = AIC 核数

从仓库根目录运行：

    python 04_tilelang_ascend/04_perf/01_num_blocks/03_wave_model.py                    # 两种 kernel 都测
    python 04_tilelang_ascend/04_perf/01_num_blocks/03_wave_model.py --kernel vector
    python 04_tilelang_ascend/04_perf/01_num_blocks/03_wave_model.py --kernel gemm --iter 100
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 共用工具在上一级 04_perf/
import perf_util  # noqa: E402

UB_SIZE = 4096  # 向量 kernel 每个 block 的 UB 工作集（float32，16 KB）
TILE_M, TILE_N, TILE_K = 256, 256, 256  # Cube kernel 每个 block 的 GEMM tile


def vector_kernel(num_blocks, iters):
    """纯向量：每个 block 在 UB 上做 iters 轮 SIMT 计算，最后写回一格。"""

    @T.prim_func
    def kernel(C: T.Tensor((num_blocks,), T.float32)):  # type: ignore
        with T.Kernel(num_blocks) as bx:
            x = T.alloc_shared((UB_SIZE,), "float32")
            with T.SimtVF(threads=256):
                for i in T.Parallel(UB_SIZE):
                    x[i] = T.Cast("float32", i) + T.Cast("float32", bx)
            for _ in T.serial(iters):
                with T.SimtVF(threads=256):
                    for i in T.Parallel(UB_SIZE):
                        x[i] = x[i] * 1.000001 + 0.5
            C[bx] = x[bx % UB_SIZE]

    return kernel


def gemm_kernel(num_blocks, iters, dtype="bfloat16"):
    """Cube + Vector 混合：每个 block 做 iters 次 256^3 GEMM，再用 dual_copy 写出。"""

    @T.prim_func
    def kernel(
        X: T.Tensor((TILE_M, TILE_K), dtype),  # type: ignore
        W: T.Tensor((TILE_N, TILE_K), dtype),  # type: ignore
        C: T.Tensor((num_blocks * TILE_M, TILE_N), "float32"),  # type: ignore
    ):
        with T.Kernel(num_blocks) as bx:
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            for _ in T.serial(iters):
                T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=True)
            T.dual_copy(res, temp)
            T.dual_copy(temp, C[bx * TILE_M : (bx + 1) * TILE_M, :])

    return kernel


def run_case(kind, blocks, iters):
    aic, aiv = perf_util.core_counts()
    wave = aiv if kind == "vector" else aic
    build = vector_kernel if kind == "vector" else gemm_kernel
    decl = "__vector__" if kind == "vector" else "__mix__(1, 2)"
    print(f"=== {kind} kernel（生成码 {decl}，波宽 = {wave}）===")

    if kind == "vector":
        inputs = ()
    else:
        torch.manual_seed(0)
        inputs = (
            torch.randn((TILE_M, TILE_K), device="npu", dtype=torch.bfloat16),
            torch.randn((TILE_N, TILE_K), device="npu", dtype=torch.bfloat16),
        )

    rows = []
    measured = {}
    for nb in blocks:
        kernel = tilelang.compile(build(nb, iters), out_idx=[-1])
        out = kernel(*inputs)
        finite = bool(torch.isfinite(out.float()).all())
        if not finite:
            raise SystemExit(f"blockDim={nb} 的结果含非有限值，说明有 block 未正常执行。")
        del out
        median, lo, hi = perf_util.bench(lambda: kernel(*inputs), reps=5, rounds=3)
        measured[nb] = (median, lo, hi)
        print(f"  已测 blockDim={nb}")

    # 单波基准：优先取正好等于波宽的配置，否则取不超过波宽的最大配置
    candidates = [nb for nb in blocks if nb <= wave]
    base_nb = wave if wave in measured else (max(candidates) if candidates else min(blocks))
    base = measured[base_nb][0]
    for nb in blocks:
        median, lo, hi = measured[nb]
        rows.append(
            [
                nb,
                f"{median * 1e6:.2f}",
                f"{lo * 1e6:.2f}~{hi * 1e6:.2f}",
                math.ceil(nb / wave),
                f"{median / base:.2f}",
            ]
        )

    print()
    perf_util.print_table(
        ["blockDim", "延迟中位数(us)", "min~max(us)", "预测波数", "实测/单波"],
        rows,
        aligns=[">", ">", ">", ">", ">"],
    )
    print()
    print(f"单波基准 = blockDim {base_nb}（不超过波宽 {wave}，所有 block 并发）。")
    print("若“实测/单波”≈“预测波数”，说明超出核数的 block 不是并发执行，而是排队分波 ——")
    print("波宽就是该 kernel 类型对应的核数。")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", choices=["vector", "gemm", "both"], default="both")
    parser.add_argument("--iter", type=int, default=200, help="每个 block 的固定工作量（迭代次数）")
    parser.add_argument("--blocks", default=None, help="逗号分隔的 blockDim 列表，默认按核数派生")
    args = parser.parse_args()

    perf_util.require_npu()
    aic, aiv = perf_util.core_counts()
    print(perf_util.device_line())
    print(f"每个 block 的固定工作量：iter={args.iter}")
    print()

    kinds = ["vector", "gemm"] if args.kernel == "both" else [args.kernel]
    for kind in kinds:
        wave = aiv if kind == "vector" else aic
        if args.blocks:
            blocks = [int(x) for x in args.blocks.split(",")]
        else:
            blocks = sorted({max(1, wave // 4), max(1, wave // 2), wave, wave + wave // 2, 2 * wave, 3 * wave, 4 * wave})
        run_case(kind, blocks, args.iter)


if __name__ == "__main__":
    main()
