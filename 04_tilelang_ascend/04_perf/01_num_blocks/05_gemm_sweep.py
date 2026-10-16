"""实验四：含 Cube 的 kernel —— num_blocks 对 GEMM 吞吐的影响。

persistent GEMM（``C = X @ W.T``，bf16 输入 / fp32 累加与输出），结构与 TileLang
官方示例 ``examples/ascend/example_gemm.py``、``tilelang/ascend/README.md`` 的
Quick Start 一致：``T.Persistent`` 把输出 tile 分给各 block，``T.gemm`` 跑在 Cube 上，
``T.dual_copy`` 把 L0C 经 UB 写回 GM（两个 AIV 子核各拿一半）。

体内含 ``T.gemm``，codegen 出 ``__global__ __mix__(1, 2)``，所以**波宽是 AIC 核数**，
不是 AIV 核数 —— 这是最容易搞错的一点。

从仓库根目录运行：

    python 04_tilelang_ascend/04_perf/01_num_blocks/05_gemm_sweep.py
    python 04_tilelang_ascend/04_perf/01_num_blocks/05_gemm_sweep.py --blocks 14,28,32,56
    python 04_tilelang_ascend/04_perf/01_num_blocks/05_gemm_sweep.py --mnk 2048
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

TILE_M, TILE_N, TILE_K = 256, 256, 256
STAGES = 2


def gemm(M, N, K, num_blocks, dtype="bfloat16", out_dtype="float32"):
    if M % TILE_M or N % TILE_N or K % TILE_K:
        raise ValueError(f"M/N/K 必须分别是 {TILE_M}/{TILE_N}/{TILE_K} 的整数倍")
    m_tiles, n_tiles, k_tiles = M // TILE_M, N // TILE_N, K // TILE_K
    out_tiles = m_tiles * n_tiles

    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),  # type: ignore
        W: T.Tensor((N, K), dtype),  # type: ignore
        C: T.Tensor((M, N), out_dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)
            for tile_idx in T.Persistent([out_tiles], num_blocks, bx):
                m_tile, n_tile = tile_idx // n_tiles, tile_idx % n_tiles
                for kt in T.Pipelined(k_tiles, num_stages=STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=kt == 0)
                T.dual_copy(res, temp)
                T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mnk", type=int, default=4096, help="方阵边长 M=N=K")
    parser.add_argument("--blocks", default=None, help="逗号分隔的 num_blocks 列表，默认按 AIC 核数派生")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    perf_util.require_npu()
    aic, aiv = perf_util.core_counts()
    M = N = K = args.mnk
    out_tiles = (M // TILE_M) * (N // TILE_N)
    print(perf_util.device_line())
    print(f"GEMM {M}x{N}x{K} bf16->fp32，tile {TILE_M}x{TILE_N}x{TILE_K}，输出 tile 数 = {out_tiles}")
    print(f"含 T.gemm -> __mix__(1, 2)，波宽 = AIC = {aic}（AIV = {aiv} 不是这里的波宽）")
    print()

    if args.blocks:
        blocks = [int(x) for x in args.blocks.split(",")]
    else:
        blocks = sorted({max(1, aic // 4), max(1, aic // 2), aic, aic + 4, 2 * aic, 4 * aic})
    blocks = [nb for nb in blocks if nb <= out_tiles]

    torch.manual_seed(0)
    x = torch.randn((M, K), device="npu", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="npu", dtype=torch.bfloat16)
    ref = x.float().cpu() @ w.float().cpu().T

    measured = {}
    for nb in blocks:
        kernel = tilelang.compile(gemm(M, N, K, nb), out_idx=-1)
        c = kernel(x, w).float().cpu()
        err = (c - ref).abs().max().item()
        if not (err < 2.0 and torch.isfinite(c).all()):
            raise SystemExit(f"num_blocks={nb} 结果异常，max_abs_err={err}")
        del c
        median, lo, hi = perf_util.bench(lambda: kernel(x, w), reps=args.reps, rounds=args.rounds)
        measured[nb] = (median, lo, hi, err)
        print(f"  已测 num_blocks={nb}: {2 * M * N * K / (median * 1e12):.2f} TFLOPS")

    best = min(blocks, key=lambda nb: measured[nb][0])
    ideal_rounds = out_tiles / aic  # 全部 AIC 满载时每个核要跑的 tile 轮数
    rows = []
    for nb in blocks:
        median, lo, hi, err = measured[nb]
        waves = math.ceil(nb / aic)
        last_wave = nb - (waves - 1) * aic
        tiles_per_block = math.ceil(out_tiles / nb)
        rounds = waves * tiles_per_block
        tflops = 2 * M * N * K / (median * 1e12)
        rows.append(
            [
                nb,
                tiles_per_block,
                waves,
                f"{last_wave}/{aic}",
                f"{median * 1e3:.3f}",
                f"{tflops:.1f}",
                f"{measured[best][0] / median:.3f}",
                f"{ideal_rounds / rounds:.3f}",
                f"{err:.2f}",
            ]
        )

    print()
    perf_util.print_table(
        ["num_blocks", "每核tile", "波数", "末波核数", "延迟(ms)", "TFLOPS", "/最优", "模型估算", "max_err"],
        rows,
        aligns=[">", ">", ">", ">", ">", ">", ">", ">", ">"],
    )
    print()
    print(f"最优 num_blocks = {best}（AIC = {aic}）。")
    print("模型估算 = (out_tiles / AIC) / (波数 × 每核 tile 数)，即只按波次和 tile 轮数估算的相对吞吐；")
    print("它忽略 L2 复用、流水充排和末波带宽差异，所以是趋势参考，不是精确预测。")
    if aic + 4 in measured:
        print()
        print(f"注意 num_blocks={aic + 4}：只比 AIC 核数多 {4} 个，却要跑 2 波，")
        print(f"实测只有最优的 {measured[best][0] / measured[aic + 4][0]:.2f} 倍 —— 零头波的代价和向量 kernel 一样。")


if __name__ == "__main__":
    main()
