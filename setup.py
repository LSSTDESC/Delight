# from distutils.core import setup

from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "delight.photoz_kernels_cy",
        ["delight/photoz_kernels_cy.pyx"],
        define_macros=[("CYTHON_LIMITED_API", "1")],
        py_limited_api=True,
    ),
    Extension(
        "delight.utils_cy",
        ["delight/utils_cy.pyx"],
        define_macros=[("CYTHON_LIMITED_API", "1")],
        py_limited_api=True,
    ),
]

setup(
    ext_modules=cythonize(ext_modules),
)
