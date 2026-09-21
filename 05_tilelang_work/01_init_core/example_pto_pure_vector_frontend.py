"""Lower a pure Vector TileLang kernel to PTO Python source."""

import tilelang
import tilelang.ascend.language as T


def pure_vector_kernel():
    size = 128

    @T.prim_func
    def main(
        A: T.Buffer((size,), "float32"), # type: ignore
        B: T.Buffer((size,), "float32"), # type: ignore
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((size,), "float32")

            T.copy(A, a_ub)
            T.copy(a_ub, B)

    return main


if __name__ == "__main__":
    artifact = tilelang.lower(pure_vector_kernel())
    print(artifact.kernel_source)
