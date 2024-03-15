from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mask_mode',
      ext_modules=[
            cpp_extension.CppExtension('mask_mode', ['mask_mode.cu']),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})