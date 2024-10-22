# from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "delight.photoz_kernels_cy",
        ["src/delight/photoz_kernels_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("CYTHON_LIMITED_API", "1")],
        py_limited_api=True,
    ),
    Extension(
        "delight.utils_cy",
        ["src/delight/utils_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("CYTHON_LIMITED_API", "1")],
        py_limited_api=True,
    ),
]

setup(ext_modules=cythonize(ext_modules), include_dirs=[numpy.get_include()])
