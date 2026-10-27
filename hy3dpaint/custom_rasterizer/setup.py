# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import sys
import subprocess
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

is_windows = sys.platform == "win32"
is_rocm = getattr(torch.version, "hip", None) is not None

base_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = os.path.join(base_dir, "lib", "custom_rasterizer_kernel")
cu_source = os.path.join(kernel_dir, "rasterizer_gpu.cu")
obj_name = "rasterizer_gpu.obj" if is_windows else "rasterizer_gpu.o"
obj_path = os.path.join(kernel_dir, obj_name)

cxx_sources = [
    os.path.join("lib", "custom_rasterizer_kernel", "rasterizer.cpp"),
    os.path.join("lib", "custom_rasterizer_kernel", "grid_neighbor.cpp"),
]

cxx_args = ["/O2", "/std:c++20"] if is_windows else ["-O3", "-std=c++20"]

if is_rocm:
    hipcc_bin = "hipcc"
    rocm_home = getattr(torch.utils.cpp_extension, "ROCM_HOME", None) or os.environ.get("ROCM_PATH")
    if rocm_home:
        for ext in ([".bat", ".exe", ""] if is_windows else [""]):
            cand = os.path.join(rocm_home, "bin", f"hipcc{ext}")
            if os.path.exists(cand):
                hipcc_bin = cand
                break

    archs = ["gfx1100", "gfx1201"]
    try:
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            if hasattr(props, "gcnArchName") and props.gcnArchName:
                curr_arch = props.gcnArchName.split(":")[0]
                if curr_arch not in archs:
                    archs.append(curr_arch)
    except Exception:
        pass

    arch_flags = [f"--offload-arch={a}" for a in archs]
    runtime_flags = ["-fms-runtime-lib=dll"] if is_windows else ["-fPIC"]

    hipcc_cmd = [
        hipcc_bin,
        *arch_flags,
        "-O3",
        *runtime_flags,
        "-c",
        cu_source,
        "-o",
        obj_path,
    ]
    print(f"Compiling HIP rasterizer kernel: {' '.join(hipcc_cmd)}")
    subprocess.check_call(hipcc_cmd)

    libraries = ["amdhip64"] if is_windows else ["amdhip64"]
    library_dirs = []
    if rocm_home:
        lib_dir = os.path.join(rocm_home, "lib")
        if os.path.exists(lib_dir):
            library_dirs.append(lib_dir)

    custom_rasterizer_module = CppExtension(
        "custom_rasterizer_kernel",
        sources=cxx_sources,
        extra_objects=[obj_path],
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=cxx_args,
    )
elif torch.cuda.is_available():
    custom_rasterizer_module = CUDAExtension(
        "custom_rasterizer_kernel",
        cxx_sources + [cu_source],
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": ["-O3", "-std=c++20"],
        },
    )
else:
    # CPU fallback
    custom_rasterizer_module = CppExtension(
        "custom_rasterizer_kernel",
        sources=cxx_sources,
        extra_compile_args=cxx_args,
    )

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)
