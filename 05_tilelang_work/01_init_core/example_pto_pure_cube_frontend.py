"""Lower a pure Cube TileLang kernel to PTO Python source."""

import tilelang
import tilelang.ascend.language as T


def pure_cube_kernel():
    tile = 16

    @T.prim_func
    def main(
        A: T.Buffer((tile, tile), "bfloat16"), # type: ignore
        B: T.Buffer((tile, tile), "bfloat16"), # type: ignore
        C: T.Buffer((tile, tile), "float32"), # type: ignore
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            accum = T.alloc_l0c((tile, tile), "float32")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(
                a_l1,
                b_l1,
                accum,
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(accum, C)

    return main


if __name__ == "__main__":
    artifact = tilelang.lower(pure_cube_kernel())
    print(artifact.kernel_source)
