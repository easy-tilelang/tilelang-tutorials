"""Lower an explicit Cube/Vector TileLang kernel to PTO Python source."""

import tilelang
import tilelang.ascend.language as T


def mixed_kernel():
    tile = 16
    vector_size = 128

    @T.prim_func
    def main(
        cube_a: T.Buffer((tile, tile), "bfloat16"), # type: ignore
        cube_b: T.Buffer((tile, tile), "bfloat16"), # type: ignore
        cube_out: T.Buffer((tile, tile), "float32"), # type: ignore
        vector_in: T.Buffer((vector_size,), "float32"), # type: ignore
        vector_out: T.Buffer((vector_size,), "float32"), # type: ignore
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            accum = T.alloc_l0c((tile, tile), "float32")
            vector_ub = T.alloc_shared((vector_size,), "float32")

            with T.Cube():
                T.copy(cube_a, a_l1)
                T.copy(cube_b, b_l1)
                T.gemm(
                    a_l1,
                    b_l1,
                    accum,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.copy(accum, cube_out)

            with T.Vector():
                T.copy(vector_in, vector_ub)
                T.copy(vector_ub, vector_out)

    return main


if __name__ == "__main__":
    with tilelang.transform.PassContext(
        config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}
    ):
        artifact = tilelang.lower(mixed_kernel())
    print(artifact.kernel_source)
