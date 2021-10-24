from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from sphinx.setup_command import BuildDoc

import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('./data')



version = '1.0.1'

args = {
    "libraries": ["m"],
    "include_dirs": [numpy.get_include()],
    "extra_link_args": ['-fopenmp'],
    "extra_compile_args": ["-ffast-math", "-fopenmp",
                           "-Wno-uninitialized",
                           "-Wno-maybe-uninitialized",
                           "-Wno-unused-function"]  # -march=native
    }

ext_modules = [
    Extension("delight.photoz_kernels_cy",
              ["delight/photoz_kernels_cy.pyx"], **args),
    Extension("delight.utils_cy",
              ["delight/utils_cy.pyx"], **args)
    ]

setup(
  name="delight",
  version=version,
  cmdclass={"build_ext": build_ext,
            'build_sphinx': BuildDoc},
  packages=['delight','interfaces.rail'],
  package_dir={'delight': './delight','interfaces.rail':'./interfaces/rail'},
  #package_data={'delightdata': ['data/BROWN_SEDs/*.dat', 'data/CWW_SEDs/*.dat','data/FILTERS/*.res']},
  package_data={'': extra_files},
  #include_package_data = True,

  command_options={
        'build_sphinx': {
            'project': (None, "delight"),
            'version': ('setup.py', version),
            'build_dir': (None, 'docs/_build'),
            'config_dir': (None, 'docs'),
            }},
  ext_modules=ext_modules)
