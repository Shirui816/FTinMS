from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

ext_modules = [
    Extension("func",
              ["func.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']
              )
]

setup(
    name="func",
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
)
