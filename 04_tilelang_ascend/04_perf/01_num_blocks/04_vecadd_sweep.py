"""实验三：真实算子上的 num_blocks 扫描 —— 流水化 vector add 的有效带宽。

kernel 结构与 TileLang 官方示例 ``examples/ascend/example_vecadd.py`` 一致
（TILE=8192、num_stages=2、SimtVF threads=2048），只把 ``num_blocks`` 参数化。

数据划分（block-cyclic）：

    iterations = N // (TILE * num_blocks)
    begin      = (step * num_blocks + bx) * TILE

每个 num_blocks 取整数倍的总工作量（N ≈ --target），因此比较的是同样字节数下的带宽。
纯向量 kernel 的波宽是 AIV 核数，表里同时给出波数和末波占用的核数：
末波只用到少量核时，整波的时间几乎不变，吞吐就会掉。

从仓库根目录运行：

    python 04_tilelang_ascend/04_perf/01_num_blocks/04_vecadd_sweep.py
    python 04_tilelang_ascend/04_perf/01_num_blocks/04_vecadd_sweep.py --mode simd
    python 04_tilelang_ascend/04_perf/01_num_blocks/04_vecadd_sweep.py --blocks 28,56,64,112
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

TILE = 8192
STAGES = 2
THREADS = 2048


def vector_add(N, num_blocks, mode="simt"):
    tile, num_stages, num_threads = TILE, STAGES, THREADS
    if N <= 0 or N % (tile * num_blocks):
        raise ValueError(f"N 必须是 {tile * num_blocks} 的正整数倍，当前 N={N}")
    iterations = N // (tile * num_blocks)

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),  # type: ignore
        B: T.Tensor((N,), "float32"),  # type: ignore
        C: T.Tensor((N,), "float32"),  # type: ignore
    ):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                if mode == "simt":
                    with T.SimtVF(threads=num_threads):
                        for i in T.Parallel(tile):
                            c[i] = a[i] + b[i]
                else:
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for i in range(tile // 64):
                            a_reg = T.simd.vld(a[i * 64])
                            b_reg = T.simd.vld(b[i * 64])
                            result = T.simd.vadd(a_reg, b_reg, mask)
                            T.simd.vsts(c[i * 64], result, mask)
                T.copy(c, C[begin : begin + tile])

    return main


def sweep(mode, blocks, target, rounds, reps):
    aiv = perf_util.core_counts()[1]
    print(f"=== mode={mode}（纯向量 kernel，波宽 = AIV = {aiv}）===")

    sizes = {}
    for nb in blocks:
        iters = max(2, round(target / (TILE * nb)))
        sizes[nb] = (TILE * nb * iters, iters)
    max_n = max(n for n, _ in sizes.values())

    # 共用一对大缓冲区，按需切片，避免每个配置各占一份显存
    torch.manual_seed(0)
    big_a = torch.randn(max_n, device="npu")
    big_b = torch.randn(max_n, device="npu")

    kernels = {}
    for nb in blocks:
        N, _ = sizes[nb]
        kernels[nb] = tilelang.compile(vector_add(N, nb, mode), out_idx=-1)
        print(f"  已编译 num_blocks={nb} (N={N})")

    measured = {}
    for nb in blocks:
        N, _ = sizes[nb]
        a, b = big_a[:N], big_b[:N]
        c = kernels[nb](a, b)
        if not torch.allclose(c.cpu(), (a + b).cpu(), rtol=1e-5, atol=1e-5):
            raise SystemExit(f"num_blocks={nb} 结果错误，先排查正确性再谈性能。")
        del c
        median, lo, hi = perf_util.bench(lambda: kernels[nb](a, b), reps=reps, rounds=rounds)
        measured[nb] = (3 * N * 4 / (median * 1e9), 3 * N * 4 / (hi * 1e9), 3 * N * 4 / (lo * 1e9))
        print(f"  已测 num_blocks={nb}: {measured[nb][0]:.2f} GB/s")

    best = max(blocks, key=lambda nb: measured[nb][0])
    rows = []
    for nb in blocks:
        N, iters = sizes[nb]
        gbps, gbps_lo, gbps_hi = measured[nb]
        waves = math.ceil(nb / aiv)
        last_wave = nb - (waves - 1) * aiv
        rows.append(
            [
                nb,
                iters,
                f"{N * 4 / 2**20:.0f}",
                waves,
                f"{last_wave}/{aiv}",
                f"{gbps:.2f}",
                f"{gbps_lo:.0f}~{gbps_hi:.0f}",
                f"{gbps / measured[best][0]:.3f}",
            ]
        )

    print()
    perf_util.print_table(
        ["num_blocks", "每核迭代", "单张量(MB)", "波数", "末波核数", "带宽中位数(GB/s)", "min~max", "/最优"],
        rows,
        aligns=[">", ">", ">", ">", ">", ">", ">", ">"],
    )
    print()
    print(f"最优 num_blocks = {best}（AIV = {aiv}）。")
    ragged = [nb for nb in blocks if nb % aiv and math.ceil(nb / aiv) > 1]
    if ragged:
        worst = min(ragged, key=lambda nb: measured[nb][0])
        print(f"零头波最吃亏的配置：num_blocks={worst} = {worst // aiv} 整波 + {worst % aiv} 个核，")
        print(f"带宽只有最优的 {measured[worst][0] / measured[best][0]:.2f} 倍 —— 多出来的那一波只用少数核，却花了整波的时间。")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["simt", "simd", "both"], default="simt")
    parser.add_argument("--target", type=int, default=2**28, help="单个张量的目标元素数（默认 2^28 = 1 GiB float32）")
    parser.add_argument("--blocks", default=None, help="逗号分隔的 num_blocks 列表，默认按 AIV 核数派生")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    perf_util.require_npu()
    aiv = perf_util.core_counts()[1]
    print(perf_util.device_line())
    print(f"TILE={TILE} num_stages={STAGES} threads={THREADS} target={args.target} 元素/张量")
    print()

    modes = ["simt", "simd"] if args.mode == "both" else [args.mode]
    for mode in modes:
        if args.blocks:
            blocks = [int(x) for x in args.blocks.split(",")]
        else:
            blocks = sorted({max(1, aiv // 4), max(1, aiv // 2), aiv, aiv + 8, 2 * aiv, 4 * aiv, 8 * aiv})
        sweep(mode, blocks, args.target, args.rounds, args.reps)


if __name__ == "__main__":
    main()
