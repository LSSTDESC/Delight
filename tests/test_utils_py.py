# -*- coding: utf-8 -*-
# to run the test do
# pytest -vv tests/test_utils_py.py

from delight.utils_py import find_positions
from time import time
import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose

def test_utils_py_findpositions():
    """
    
    """

    fz1 = np.array([0.5,1,6.,3.],dtype = np.float64)+1.
    fzGrid = np.linspace(0.,10.,20)+1.
    N01 = len(fz1)
    nz = len(fzGrid)
    ps1 = np.zeros(N01,dtype = np.uint64)
    ps1_out = find_positions(N01,nz,fz1,ps1,fzGrid)
    assert_allclose(ps1_out,np.array([ 0,  1, 11,  5]),verbose=True)
    