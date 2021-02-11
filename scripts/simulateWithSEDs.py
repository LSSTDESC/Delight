#######################################################################################################
#
# script : simulateWithSED.py
#
# simulate mock data with those filters and SEDs
# produce files `galaxies-redshiftpdfs.txt` and `galaxies-redshiftpdfs2.txt` for training and target
#
#########################################################################################################


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from delight.io import *
from delight.utils import *

if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')

params = parseParamFile(sys.argv[1], verbose=False, catFilesNeeded=False)
dir_seds = params['templates_directory']
sed_names = params['templates_names']

# redshift grid
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

numZ = redshiftGrid.size
numT = len(sed_names)
numB = len(params['bandNames'])
numObjects = params['numObjects']
noiseLevel = params['noiseLevel']

# container of interpolation functions: row sed, column bands
# one row per sed, one column per band
f_mod = np.zeros((numT, numB), dtype=object)

# loop on SED
# read the fluxes file at different redshift : file sed_name + '_fluxredshiftmod.txt'
for it, sed_name in enumerate(sed_names):
    # data : redshifted fluxes (row vary with z, columns: filters)
    data = np.loadtxt(dir_seds + '/' + sed_name + '_fluxredshiftmod.txt')
    # build the interpolation of flux wrt redshift for each band
    for jf in range(numB):
        f_mod[it, jf] = interp1d(redshiftGrid, data[:, jf], kind='linear')

# Generate training data
#-------------------------
# pick set of redshift at random
redshifts = np.random.uniform(low=redshiftGrid[0],
                              high=redshiftGrid[-1],
                              size=numObjects)
#pick some SED type at random
types = np.random.randint(0, high=numT, size=numObjects)

ell = 1e6
# what is fluxes and fluxes variance
fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))
# loop on objects to simulate for the training
for k in range(numObjects):
    #loop on number of bands
    for i in range(numB):
        trueFlux = ell * f_mod[types[k], i](redshifts[k]) # noiseless flux at the random redshift
        noise = trueFlux * noiseLevel
        fluxes[k, i] = trueFlux + noise * np.random.randn() # noisy flux
        fluxesVar[k, i] = noise**2.

# container for training galaxies output
data = np.zeros((numObjects, 1 + len(params['training_bandOrder'])))
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, prefix="training_")

for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
    data[:, pf] = fluxes[:, ib]
    data[:, pfv] = fluxesVar[:, ib]
data[:, redshiftColumn] = redshifts
data[:, -1] = types
np.savetxt(params['trainingFile'], data)

# Generate Target data
#----------------------
# pick set of redshift at random
redshifts = np.random.uniform(low=redshiftGrid[0],
                              high=redshiftGrid[-1],
                              size=numObjects)
types = np.random.randint(0, high=numT, size=numObjects)

fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

# loop on objects in target files
for k in range(numObjects):
    # loop on bands
    for i in range(numB):
        # compute the flux in that band at the redshift
        trueFlux = f_mod[types[k], i](redshifts[k])
        noise = trueFlux * noiseLevel
        fluxes[k, i] = trueFlux + noise * np.random.randn()
        fluxesVar[k, i] = noise**2.

data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, prefix="target_")

for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
    data[:, pf] = fluxes[:, ib]
    data[:, pfv] = fluxesVar[:, ib]
data[:, redshiftColumn] = redshifts
data[:, -1] = types
np.savetxt(params['targetFile'], data)
