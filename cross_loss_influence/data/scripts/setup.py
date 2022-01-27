# Created by Andrew Silva
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("py_data_speeder.pyx",
                          annotate=True, language="c++")
)