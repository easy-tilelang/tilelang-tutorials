"""查询本机 Ascend 的 AIC/AIV 核数，并用三条独立路径交叉验证。

后续实验（波次模型、num_blocks 扫描）都以这里的核数为基准，所以先确认读数。
三条路径最终都落到驱动的同一个查询接口，读数应当一致：

1. ``torch.npu.get_device_properties`` —— torch_npu 封装
2. ``tilelang.carver...ascend_driver`` —— TileLang 封装
3. ``aclGetDeviceCapability`` (ctypes) —— 直接调 ACL，不经过 torch

从仓库根目录运行：

    python 04_tilelang_ascend/04_perf/01_num_blocks/01_device_cores.py
"""

import ctypes
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

import tilelang
import tilelang.carver.arch.driver.ascend_driver as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 共用工具在上一级 04_perf/
import perf_util  # noqa: E402

# include/acl/acl_base_rt.h 中的 aclDeviceInfo 枚举
ACL_DEVICE_INFO_AI_CORE_NUM = 0
ACL_DEVICE_INFO_VECTOR_CORE_NUM = 1
ACL_DEVICE_INFO_L2_SIZE = 2


def acl_device_capability(device_id=0):
    """直接通过 ACL C 接口查询核数，作为不依赖 torch_npu 的交叉验证。"""
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclInit(None)  # 进程内已初始化时返回非 0，可忽略
    lib.aclGetDeviceCapability.restype = ctypes.c_int
    lib.aclGetDeviceCapability.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]

    def query(item):
        value = ctypes.c_int64(0)
        ret = lib.aclGetDeviceCapability(device_id, item, ctypes.byref(value))
        if ret != 0:
            raise RuntimeError(f"aclGetDeviceCapability(item={item}) failed, ret={ret}")
        return value.value

    return query(ACL_DEVICE_INFO_AI_CORE_NUM), query(ACL_DEVICE_INFO_VECTOR_CORE_NUM), query(ACL_DEVICE_INFO_L2_SIZE)


def main():
    perf_util.require_npu()
    props = torch.npu.get_device_properties(0)
    cube_a, vector_a = props.cube_core_num, props.vector_core_num
    cube_b, vector_b = ad.get_num_cube_cores(), ad.get_num_vector_cores()
    cube_c, vector_c, l2_c = acl_device_capability()

    print(f"device            : {props.name}")
    print(f"tilelang          : {tilelang.__version__}")
    print(f"torch / torch_npu : {torch.__version__} / {torch_npu.__version__}")
    print()
    perf_util.print_table(
        ["查询方式", "AIC(cube)", "AIV(vector)"],
        [
            ["torch.npu.get_device_properties", cube_a, vector_a],
            ["tilelang carver ascend_driver", cube_b, vector_b],
            ["ctypes aclGetDeviceCapability", cube_c, vector_c],
        ],
        aligns=["<", ">", ">"],
    )
    print()
    print(f"L2 cache          : {props.L2_cache_size / 2**20:.0f} MB (ACL 读数 {l2_c / 2**20:.0f} MB)")
    print(f"HBM               : {props.total_memory / 2**30:.0f} GB")
    print()
    if not (cube_a == cube_b == cube_c and vector_a == vector_b == vector_c):
        raise SystemExit("三条路径读数不一致，请先排查驱动 / CANN / torch_npu 版本匹配问题。")

    print("读数一致。后续实验的波宽（wave size）取值：")
    print(f"  纯向量 kernel（无 T.gemm，生成 __vector__）       -> 一波最多 {vector_a} 个 block")
    print(f"  含 Cube 的 kernel（有 T.gemm，生成 __mix__(1,2)） -> 一波最多 {cube_a} 个 block")


if __name__ == "__main__":
    main()
