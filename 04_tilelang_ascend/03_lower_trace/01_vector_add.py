"""Ascend A5 vector add：追踪编译 pass 并验证 NPU 结果。"""

import os

# TileLang 在导入时读取 tracing 配置；命令行环境变量可以覆盖默认值。
os.environ.setdefault("TL_LOWER_TRACE", "html")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")


import tilelang
import tilelang.ascend.language as T
from tilelang.env import env
import torch


N = 2**20
DTYPE = "float32"


@tilelang.jit(out_idx=[-1])
def vector_add(n):
    num_blocks, tile, num_stages = 64, 8192, 2
    if n <= 0 or n % (tile * num_blocks):
        raise ValueError(f"n must be a positive multiple of {tile * num_blocks}")
    iterations = n // (tile * num_blocks)

    @T.prim_func
    def kernel(
        A: T.Tensor((n,), DTYPE), # type: ignore
        B: T.Tensor((n,), DTYPE), # type: ignore
        C: T.Tensor((n,), DTYPE), # type: ignore
    ):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), DTYPE)
            b = T.alloc_shared((tile,), DTYPE)
            c = T.alloc_shared((tile,), DTYPE)
            T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                with T.SimdVF():
                    for i in T.Parallel(tile):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + tile])
                T.print(c[0], msg="c[0] = ")

    return kernel


def main():
    # 调用 JIT 工厂触发编译，此时产生 lower-trace 报告。
    compiled = vector_add(N)

    torch.manual_seed(0)
    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()
    actual = compiled(a, b)
    torch.testing.assert_close(actual, a + b, rtol=1e-5, atol=1e-5)
    print(f"Test passed: C = A + B (N={N}, dtype={DTYPE}).")


if __name__ == "__main__":
    main()
