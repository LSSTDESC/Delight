#######################################################################################################
#
# script : convertDESCcat.py
#
# convert DESC catalog to be injected in Delight
# produce files `galaxies-redshiftpdfs.txt` and `galaxies-redshiftpdfs2.txt` for training and target
#
#########################################################################################################


import sys
import os
import numpy as np
import pandas as pd
import h5py
from functools import reduce

from delight.io import *
from delight.utils import *
from tables_io import io
import logging

logger = logging.getLogger(__name__)

# option to convert DC2 flux level (in AB units) into internal Delight units
# this option will be removed when optimisation of parameters will be implemented
FLAG_CONVERTFLUX_TODELIGHTUNIT = True

if FLAG_CONVERTFLUX_TODELIGHTUNIT:
    flux_multiplicative_factor = 2.22e10
else:
    flux_multiplicative_factor = 1



# the order by which one order the dataframe
list_of_cols = [
    "id",
    "redshift",
    "mag_u_lsst",
    "mag_g_lsst",
    "mag_r_lsst",
    "mag_i_lsst",
    "mag_z_lsst",
    "mag_y_lsst",
    "mag_err_u_lsst",
    "mag_err_g_lsst",
    "mag_err_r_lsst",
    "mag_err_i_lsst",
    "mag_err_z_lsst",
    "mag_err_y_lsst",
]
list_of_filters = ["u", "g", "r", "i", "z", "y"]
list_of_Filters = [ b.upper() for b in  list_of_filters ]
dict_of_filters = {0:"u", 1:"g", 2:"r", 3:"i", 4:"z", 5:"y"}
dict_of_Filters = {0:"U", 1:"G", 2:"R", 3:"I", 4:"Z", 5:"Y"}

Nf = len(list_of_filters)


def h5filetodataframe(filename, group="photometry"):
    """
    Function to convert the LSST magnitudes hdf5 file into a pandas dataFrame
    """
    data = h5py.File(filename, "r")
    list_of_keys = list(data[group].keys())
    all_data = np.array([data[group][key][:] for key in list_of_keys])
    df = pd.DataFrame(all_data.T, columns=list_of_keys)
    if "id" in list_of_keys:
        df = df.astype({"id": int})
    return df

def dicttodataframe(dict_data):
    """
    Function to convert the LSST magnitudes in rail dict into a pandas dataFrame
    """
   
    list_of_keys = list(dict_data.keys())
    
    df = pd.DataFrame()
    for key in list_of_keys:
        df[key] = dict_data[key]

    if "id" in list_of_keys:
        df = df.astype({"id": int})
    return df

def CheckBadFluxes(fl, dfl, mag, dmag, maxmag=30.0):
    """
    Interpolate fluxes as ther are missing
    Parameters:
       fl : array of fluxes
       dfl : array of flux error
       mag : array of magnitudes
       dmag: array on magnitude errors
       maxmag: max magnitude 
    Return : 
       the arrays of fluxes and flux error corrected for mission values
    """
    indexes_bad = np.where(mag > maxmag)[0]
    indexes_good = np.where(mag < maxmag)[0]

    if len(indexes_bad) > 0:
        for idx in indexes_bad:
            # in band g,r,i,z
            if idx > 0 and idx < Nf - 1:
                # have two good neighbourgs
                if idx - 1 in indexes_good and idx + 1 in indexes_good:
                    fl[idx] = np.mean([fl[idx - 1], fl[idx + 1]])
                    dfl[idx] = np.max([dfl[idx - 1], dfl[idx + 1]]) * 5.0
                elif idx - 1 in indexes_good:
                    fl[idx] = fl[idx - 1]
                    dfl[idx] = dfl[idx - 1] * 10.0
                elif idx + 1 in indexes_good:
                    fl[idx] = fl[idx + 1]
                    dfl[idx] = dfl[idx + 1] * 10.0
                else:
                    fl[idx] = np.mean(fl[indexes_good])
                    dfl[idx] = np.max(fl[indexes_good]) * 100.0
            elif idx == 0:
                if idx + 1 in indexes_good:
                    fl[idx] = fl[idx + 1]
                    dfl[idx] = dfl[idx + 1] * 10.0
                else:
                    fl[idx] = np.mean(fl[indexes_good])
                    dfl[idx] = np.max(fl[indexes_good]) * 100.0
            elif idx == Nf - 1:
                if idx - 1 in indexes_good:
                    fl[idx] = fl[idx - 1]
                    dfl[idx] = dfl[idx - 1] * 10.0
                else:
                    fl[idx] = np.mean(fl[indexes_good])
                    dfl[idx] = np.max(fl[indexes_good]) * 100.0

    return fl, dfl


def convert_to_ABflux(row):
    """
    Convert AB magnitudes into FAB flux (units AB, that is per 3631 Jy
    This function is dedicated to be applied to a pandas dataframe containing magnitudes and magnitudes error
    Parameters:
      row : one row of the pandas dataframe
    Returns:
      the pandas series of flux and flux error corrected for missing values by usingthe CheckBadFluxes 
    """

    fl = np.zeros(Nf)
    dfl = np.zeros(Nf)
    mag = np.zeros(Nf)
    dmag = np.zeros(Nf)
    all_fname = []
    all_ferrname = []

    for idx, band in enumerate(list_of_filters):
        mag_label = f"mag_{band}_lsst"
        magerr_label = f"mag_err_{band}_lsst"
        flux_label = f"fab_{band}_lsst"
        fluxerr_label = f"fab_err_{band}_lsst"
        m = row[mag_label]
        dm = row[magerr_label]
        f = np.power(10.0, -0.4 * m)
        df = np.log(10.0) / 2.5 * f * dm
        fl[idx] = f
        mag[idx] = m
        dfl[idx] = df
        dmag[idx] = dm
        all_fname.append(flux_label)
        all_ferrname.append(fluxerr_label)

    # decide what to do if one magnitude is too high
    fl, dfl = CheckBadFluxes(fl, dfl, mag, dmag)
    column_names = all_fname + all_ferrname
    data = np.concatenate((fl, dfl))
    return pd.Series(data, index=column_names)


def generatedelightdataarray(df,params,prefix):
    """Generate the 2D array for sving it in file

    Args:
        df (pd.dataFrame): dataframe
        params (dict): dictionnary of params
        prefix : prefix to select training or target
    """

    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix=prefix)

    data = np.array([])
    return data




def convertDESCcatChunk(configfilename,data,chunknum):

        """
        convertDESCcatChunk(configfilename,data,chunknum)

        Convert files in ascii format to be used by Delight
        Input data can be filtered by series of filters. But it is necessary to remember which entries are kept,
        which are eliminated

        input args:
        - configfilename : Delight configuration file containing path for output files (flux variances and redshifts)
        - data : the DC2 data
        - chunknum : number of the chunk
       
        output :
        - the target file of the chunk which path is in configuration file
        :return:
        - the list of selected (unfiltered DC2 data)
        """



        msg="--- Convert DESC catalogs chunk {}---".format(chunknum)
        logger.info(msg)

        
        print("====> ********************************************************************************\n")
        print("====>  ** convertDESCcatChunk data from rail = ",data)
        print("====> ********************************************************************************\n")

       
        # new implementation 
        df_test = dicttodataframe(data)
        df_test = df_test[list_of_cols]

        # assert 1
        assert df_test.isnull().values.any() == False

        # convert AB magnitude in AB flux (in units of 3630.78 Jy) into a new pandas dataframe
        # Note missing magnitudes (high magnitudes values) are replaced by interpolated flux
        # No row is removed

        df_test_fl = df_test.apply(convert_to_ABflux, axis=1)

        # assert 2
        df_test_fl.isnull().values.any()

        # merge the id and redshift with the fluxes
        df_test = df_test[["id","redshift"]].join(df_test_fl)


        print(">>>>> TEST-FLUX pandas dataframe = ",df_test)

        # produce a numpy array
        #magdata = group_entries(data)


        # remember the number of entries
        Nin = len(df_test)
        idxFinal = Nin
        msg = "Number of objects = {} , in chunk : {}".format(Nin,chunknum)
        logger.debug(msg)


        gid =  df_test["id"].values
        rs = df_test["redshift"].values

        # 2) parameter file

        params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

        numB = len(params['bandNames'])
        numObjects = len(gid)

        msg = "get {} objects ".format(numObjects)
        logger.debug(msg)

        logger.debug(params['bandNames'])

        # Generate target data (keep the original implementation)
        # --------------------------------------------------------

        # what is fluxes and fluxes variance
        fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

        # container for target galaxies output
        # at some redshift, provides the flux and its variance inside each band
        # assume bands are in order u,g,r,i,z,y
        for i in range(numB):
            # the band order is defined in the parameter file
            band_name = dict_of_filters[i]
            flux_label = f"fab_{band_name}_lsst"
            fluxerr_label = f"fab_err_{band_name}_lsst"
            fluxes[:,i] = df_test[flux_label].values*flux_multiplicative_factor
            fluxesVar[:,i] = (df_test[fluxerr_label].values*flux_multiplicative_factor)**2


        data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
        bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn, refBandColumn = readColumnPositions(params,
                                                                                                                 prefix="target_")

        for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
            data[:, pf] = fluxes[:, ib]
            data[:, pfv] = fluxesVar[:, ib]
        data[:, redshiftColumn] = rs
        data[:, -1] = 0  # NO TYPE

        msg = "write file {}".format(os.path.basename(params['targetFile']))
        logger.debug(msg)

        msg = "write target file {}".format(params['targetFile'])
        logger.debug(msg)

        outputdir = os.path.dirname(params['targetFile'])
        if not os.path.exists(outputdir):  # pragma: no cover
            msg = " outputdir not existing {} then create it ".format(outputdir)
            logger.info(msg)
            os.makedirs(outputdir)

        # save txt file and hdf5 files
        np.savetxt(params['targetFile'], data)
        hdf5file_fn =  os.path.basename(params['targetFile']).split(".")[0]+".h5"
        output_path = os.path.dirname(params['targetFile'])
        hdf5file_fullfn = os.path.join(output_path,hdf5file_fn)
        writedataarrayh5(hdf5file_fullfn,'target_',data)

        



################################################################################
# New version of RAIL with data structure directly provided: (SDC 2024/11/04)  #
################################################################################

def convertDESCcatTrainData(configfilename,descatalogdata):

    """
    convertDESCcatData(configfilename,desccatalogdata):


    Convert files in ascii format to be used by Delight

    input args:
    - configfilename : Delight configuration file containingg path for output files (flux variances and redshifts)
    - desccatalogdata : data provided by RAIL (dictionary format)
 


    output :
    - the Delight training  which path is in configuration file

    :return: nothing

    """


    logger.info("--- Convert DESC training  catalogs data ---")

    if FLAG_CONVERTFLUX_TODELIGHTUNIT:
        flux_multiplicative_factor = 2.22e10
    else:
        flux_multiplicative_factor = 1


    print("====> ********************************************************************************\n")
    print("====> ** convertDESCcatTrainData data from rail = ",descatalogdata)
    print("====> ********************************************************************************\n")    


    # get dict data into a pandas dataframe
    df_train = dicttodataframe(descatalogdata)

    # Check there are not nan
    assert df_train.isnull().values.any() == False

    # Compute the AB fluxes from the AB magnitudes
    # Note if data are missing (high magnitudes), the flux are interpolated with higher errors
    # No row is removed
    df_train_fl = df_train.apply(convert_to_ABflux, axis=1)

    # Check flux are not nan
    df_train_fl.isnull().values.any() == False

    # merge the id and redshift with the fluxes
    df_train = df_train[["id","redshift"]].join(df_train_fl)


    print(">>>>> TRAIN-FLUX pandas dataframe = ",df_train)


    
    # remember the number of entries
    Nin = len(df_train)
    msg = "Number of objects = {} , in  training dataset".format(Nin)
    logger.debug(msg)

    gid =  df_train["id"].values
    rs = df_train["redshift"].values
    

    # 2) parameter file
    #-------------------

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

    numB = len(params['bandNames'])
    numObjects = len(gid)

    msg = "ConvertDESCcatTrainData  :: get {} objects ".format(numObjects)
    logger.debug(msg)
    logger.debug(params['bandNames'])


    # Generate training data
    #-------------------------


    # what is fluxes and fluxes variance
    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # container for training galaxies output
    # at some redshift, provides the flux and its variance inside each band
    # assume bands are in order u,g,r,i,z,y
    for i in range(numB):
        # the band order is defined in the parameter file
        band_name = dict_of_filters[i]
        flux_label = f"fab_{band_name}_lsst"
        fluxerr_label = f"fab_err_{band_name}_lsst"
        fluxes[:,i] = df_train[flux_label].values*flux_multiplicative_factor
        fluxesVar[:,i] = (df_train[fluxerr_label].values*flux_multiplicative_factor)**2


    # container for training galaxies output
    # at some redshift, provides the flux and its variance inside each band
    data = np.zeros((numObjects, 1 + len(params['training_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="training_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]

    data[:, redshiftColumn] = rs
    data[:, -1] = 0  # NO type


    msg="write training file {}".format(params['trainingFile'])
    logger.debug(msg)

    outputdir=os.path.dirname(params['trainingFile'])
    if not os.path.exists(outputdir):
        msg = " outputdir not existing {} then create it ".format(outputdir)
        logger.info(msg)
        os.makedirs(outputdir)
    np.savetxt(params['trainingFile'], data)
    hdf5file_fn =  os.path.basename(params['trainingFile']).split(".")[0]+".h5"
    output_path = os.path.dirname(params['trainingFile'])
    hdf5file_fullfn = os.path.join(output_path,hdf5file_fn)
    writedataarrayh5(hdf5file_fullfn,'training_',data)





#---

def convertDESCcatTargetFile(configfilename,desctargetcatalogfile):

    """
    convertDESCcatTargetFile(configfilename,desctargetcatalogfile)
    

    Convert files in ascii format to be used by Delight

    input args:
    - configfilename : Delight configuration file containingg path for output files (flux variances and redshifts)
    - desctargetcatalogfile : target file provided by RAIL (hdf5 format)
    - flag_filter_ : Activate filtering on validation data
    - snr_cut: Cut on flux SNR in validation data

    output :
    - the Delight target file which path is in configuration file

    :return: nothing

    """


    logger.info("--- Convert DESC target catalogs ---")

    if FLAG_CONVERTFLUX_TODELIGHTUNIT:
        flux_multiplicative_factor = 2.22e10
    else:
        flux_multiplicative_factor = 1


    print("====> ********************************************************************************")
    print("====>  ** convertDESCcatTargetFile : desctargetcatalogfile = ", desctargetcatalogfile)
    print("====> ********************************************************************************")    



    df_target = h5filetodataframe(desctargetcatalogfile, group="photometry")
    print(">>>>> TARGET-MAG pandas dataframe = ",df_target)

    

    # Check there are not nan
    assert df_target.isnull().values.any() == False

    # Compute the AB fluxes from the AB magnitudes
    # Note if data are missing (high magnitudes), the flux are interpolated with higher errors
    # No row is removed
    df_target_fl = df_target.apply(convert_to_ABflux, axis=1)

    # Check flux are not nan
    df_target_fl.isnull().values.any() == False

    # merge the id and redshift with the fluxes
    df_target = df_target[["id","redshift"]].join(df_target_fl)


    print(">>>>> TARGET-FLUX pandas dataframe = ",df_target)


    gid =  df_target["id"].values
    rs = df_target["redshift"].values

    
    
    # 2) decode parameter file
    #------------------- 

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

    numB = len(params['bandNames'])
    numObjects = len(gid)

    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    logger.debug(params['bandNames'])
    
    
    # 3) Generate target data
    #------------------------

    numObjects = len(gid)
    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))


    # container for target galaxies output
    # at some redshift, provides the flux and its variance inside each band
    # assume bands are in order u,g,r,i,z,y
    for i in range(numB):
        # the band order is defined in the parameter file
        band_name = dict_of_filters[i]
        flux_label = f"fab_{band_name}_lsst"
        fluxerr_label = f"fab_err_{band_name}_lsst"
        fluxes[:,i] = df_target[flux_label].values*flux_multiplicative_factor
        fluxesVar[:,i] = (df_target[fluxerr_label].values*flux_multiplicative_factor)**2
            

    data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="target_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = rs
    data[:, -1] = 0  # NO TYPE

    msg = "write file {}".format(os.path.basename(params['targetFile']))
    logger.debug(msg)

    msg = "write target file {}".format(params['targetFile'])
    logger.debug(msg)

    outputdir = os.path.dirname(params['targetFile'])
    if not os.path.exists(outputdir):
        msg = " outputdir not existing {} then create it ".format(outputdir)
        logger.info(msg)
        os.makedirs(outputdir)

    np.savetxt(params['targetFile'], data)
    hdf5file_fn =  os.path.basename(params['targetFile']).split(".")[0]+".h5"
    output_path = os.path.dirname(params['targetFile'])
    hdf5file_fullfn = os.path.join(output_path,hdf5file_fn)
    writedataarrayh5(hdf5file_fullfn,'target_',data)


    

if __name__ == "__main__":  # pragma: no cover
    # execute only if run as a script


    msg="Start convertDESCcat.py"
    logger.info(msg)
    logger.info("--- convert DESC catalogs ---")



    if len(sys.argv) < 4:
        raise Exception('Please provide a parameter file and the training and validation and catalog files')

    convertDESCcatTargetFile(sys.argv[1],sys.argv[2])
