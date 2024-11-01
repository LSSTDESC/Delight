import sys
import os
import numpy as np
from functools import reduce

import pprint

from delight.io import *
from delight.utils import *
import h5py

import logging


logger = logging.getLogger(__name__)



def getDelightRedshiftEstimation(configfilename,chunknum,nsize,index_sel):
    """
    zmode, PDFs = getDelightRedshiftEstimation(delightparamfilechunk,self.chunknum,nsize,indexes_sel)

    input args:
      - nsize : size of arrays to return
      - index_sel : indexes in final arays of processed redshits by delight

    :return:
    """

    msg = "--- getDelightRedshiftEstimation({}) for chunk {}---".format(nsize,chunknum)
    logger.info(msg)

    # initialize arrays to be returned
    zmode  = np.full(nsize, fill_value=-1,dtype=np.float64)

    params = parseParamFile(configfilename, verbose=False)

    # redshiftGrid has nz size
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

    # the pdfs have (m x nz) size
    # where m is the number of redshifts calculated by delight
    # nz is the number of redshifts
    pdfs = np.loadtxt(params['redshiftpdfFile'])
    pdfs /= np.trapz(pdfs, x=redshiftGrid, axis=1)[:, None]
    nzbins = len(redshiftGrid)
    full_pdfs = np.zeros([nsize, nzbins])
    full_pdfs[index_sel] = pdfs

    # find the index of the redshift where there is the mode
    # the following arrays have size m
    indexes_of_zmode = np.argmax(pdfs,axis=1)

    redshifts_of_zmode = redshiftGrid[indexes_of_zmode]


    # array of zshift (z-zmode) :  of size (m x nz)
    zshifts_of_mode = redshiftGrid[np.newaxis,:]-redshifts_of_zmode[:,np.newaxis]

    # copy only the processed redshifts and widths into the final arrays of size nsize
    # for RAIL
    zmode[index_sel] = redshifts_of_zmode


    return zmode, full_pdfs

def getdatah5(filename,prefix):
    """
    read hdf5 data
    """
    hdf5file_fn =  os.path.basename(filename).split(".")[0]+".h5"
    input_path = os.path.dirname(filename)
    hdf5file_fullfn = os.path.join(input_path , hdf5file_fn)
    with h5py.File(hdf5file_fullfn, 'r') as hdf5_file:
        f_array = hdf5_file[prefix][:]
    return f_array


def getDelightRedshiftEstimationh5(configfilename,chunknum,nsize,index_sel):
    """
    zmode, PDFs = getDelightRedshiftEstimation(delightparamfilechunk,self.chunknum,nsize,indexes_sel)

    input args:
      - nsize : size of arrays to return
      - index_sel : indexes in final arays of processed redshits by delight

    :return:
    """

    msg = "--- getDelightRedshiftEstimation({}) for chunk {}---".format(nsize,chunknum)
    logger.info(msg)

    # initialize arrays to be returned
    zmode  = np.full(nsize, fill_value=-1,dtype=np.float64)

    params = parseParamFile(configfilename, verbose=False)

    # redshiftGrid has nz size
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

    # the pdfs have (m x nz) size
    # where m is the number of redshifts calculated by delight
    # nz is the number of redshifts
    #pdfs = np.loadtxt(params['redshiftpdfFile'])
    pdfs = getdatah5(params['redshiftpdfFile'],prefix="gp_pdfs_")
    pdfs /= np.trapz(pdfs, x=redshiftGrid, axis=1)[:, None]
    nzbins = len(redshiftGrid)
    full_pdfs = np.zeros([nsize, nzbins])
    full_pdfs[index_sel] = pdfs

    # find the index of the redshift where there is the mode
    # the following arrays have size m
    indexes_of_zmode = np.argmax(pdfs,axis=1)

    redshifts_of_zmode = redshiftGrid[indexes_of_zmode]


    # array of zshift (z-zmode) :  of size (m x nz)
    zshifts_of_mode = redshiftGrid[np.newaxis,:]-redshifts_of_zmode[:,np.newaxis]

    # copy only the processed redshifts and widths into the final arrays of size nsize
    # for RAIL
    zmode[index_sel] = redshifts_of_zmode


    return zmode, full_pdfs

