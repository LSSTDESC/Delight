#from distutils.core import setup 

from setuptools import setup, find_packages, find_namespace_packages


from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from sphinx.setup_command import BuildDoc



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
 
  #packages=find_packages(exclude=['tests','scripts','data']),  
  #packages=['delight'],
  #packages=['delight','delight.interfaces','delight.interfaces.rail'],
  packages = find_namespace_packages(),
  package_dir={'delight': './delight','delight.interfaces':'./delight/interfaces','delight.interfaces.rail':'./delight/interfaces/rail'},
  #package_data={'delightdata': ['data/BROWN_SEDs/*.dat', 'data/CWW_SEDs/*.dat','data/FILTERS/*.res']},
  #package_data={'': extra_files},
  command_options={
        'build_sphinx': {
            'project': (None, "delight"),
            'version': ('setup.py', version),
            'build_dir': (None, 'docs/_build'),
            'config_dir': (None, 'docs'),
            }},
  ext_modules=ext_modules)
