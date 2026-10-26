import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

arch_flags = ["-gencode=arch=compute_121,code=sm_121"]

ext = CUDAExtension(
    name="w8a16_cuda_qmm",
    sources=["qmm_kernel.cu", "qmm_w8a16.cpp"],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            *arch_flags,
        ],
    },
)

from setuptools import setup
setup(
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
