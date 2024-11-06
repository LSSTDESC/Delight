# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from distutils import dir_util
import pytest
import os

from delight.io import *
#import delight
#import os

#PATH = delight.__path__[0]

paramFile = "tests/parametersTest.cfg"
#paramFile = os.path.join(PATH, "tests/parametersTest.cfg")

#https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data



#@pytest.fixture(scope="module")
#def datadir(tmpdir, request):
#    '''
#    Fixture responsible for searching a folder with the same name of test
#    module and, if available, moving all contents to a temporary directory so
#    tests can use them freely.
#    '''
#    filename = request.module.__file__
#    test_dir, _ = os.path.splitext(filename)
#
#    if os.path.isdir(test_dir):
#        dir_util.copy_tree(test_dir, bytes(tmpdir))
#
#    return tmpdir
#  >> got an error ScopeMismatch: You tried to access the function scoped fixture tmpdir with a module scoped request object. Requesting fixture stack:
#  >> tests/test_io.py:21:  def datadir(tmpdir, request)



@pytest.mark.skip(reason="Unable to read an external file in pytest (at NERSC)")
def test_Parser(datadir):
    params = parseParamFile(datadir.join(paramFile), verbose=False)


@pytest.mark.skip(reason="Unable to read an external file in pytest (at NERSC)")
def test_createGrids(datadir):
    params = parseParamFile(datadir.join(paramFile), verbose=False)
    out = createGrids(params)

@pytest.mark.skip(reason="Unable to read an external file in pytest (at NERSC)")
def test_readBandCoefficients(datadir):
    params = parseParamFile(datadir.join(paramFile), verbose=False)
    out = readBandCoefficients(params)

@pytest.mark.skip(reason="Unable to read an external file in pytest (at NERSC)")
def test_readColumnPositions(datadir):
    params = parseParamFile(datadir.join(paramFile), verbose=False)
    out = readColumnPositions(params)
