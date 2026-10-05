from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 运行python setup.py install之后，只需要在要用自定义算子的文件顶部加from custom_attention import flash_attn_varlen_func即可

setup(
    name='custom_attention_pkg',
    version='0.2.0',
    description='BF16 paged FlashAttention kernels for nano-vLLM',
    ext_modules=[
        CUDAExtension(
            name='custom_attention_ext',
            sources=[
                'custom_attn_wrapper.cu'
            ],
            depends=['attention_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math', '-lineinfo']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
