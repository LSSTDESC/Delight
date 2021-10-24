# README.md : Interface of LSST-DESC RAIL to Delight package 

---------------------------------------------------------------------------------------

## RAIL interface class to Delight

Steering Delight from RAIL


Used on Vera C. Runbin LSST only estimation

- Author        : Sylvie Dagoret-Campagne

- Affiliation   : IJCLab/IN2P3/CNRS/France

- Creation date : March 2021

- Last update   : March 29th 2021

----------------------------------------------------------------------------------------



These set of python modules may be called from LSST-DESC framework called RAIL.
In the context of RAIL an algorithm called
 **DelightPZ** for performing for PhotoZ estimation with Delight.
When Delight is used in the context of DESC-LSST, a training dataset with a validation dataset of photometrics redshift data are provided in RAIL.
The running mode of Delight in RAIL framework in the context of LSST-DESC is refered as the **STANDARD MODE**.

Optionaly it is possible to run Delight in **TUTORIAL MODE**. In the later case Delight generate internally training and test datasets. However, there is no connexion data between RAIL and Delight.
One has to consider the **TUTORIAL MODE** should be used by experts only for debugging purposes.

Delight is controled by its configuration parameter file **parametersTest.cfg**. This means that this configuration parameter file is generated from RAIL according to its configuration parameter **delightPZ.yaml**.

Then the RAIL traning and validation dataset are converted into an ascii format files readable by Delight. Conversely results of Delight PhotoZ estimation are written in an ascii file, then must be given to RAIL. 

## Main Delight scripts transposed to be called by rail

### Initialisation

- **processFilters.py**: Decompose the Filter transmission into a gaussian mixture sum. The filters to be used are selected in the Delight configuration file. In the context of DESC, only LSST filters are selected (u,g,r,i,z,y). The parameters of this fit are written in the file **'_gaussian_coefficients.txt'** in the Filter directory.

- **processSEDs.py**: Build a model of fluxes-redshifts from a set of SED templates in the chosen filters. It is possible to use either the CWW model or the Brown model. By now, CWW SED are chosen in the delight configuration file. The fluxes-redshifts models, one per SED template are written in the file '_fluxredshiftmod.txt'.

- It is important to notice that Filters and SED data are shipped in the installation of Delight. This is probably not the best choice because during its processing, Delight writes the resuls of **processFilters.py** and **processSEDs.py** in those directory, unless Delight is installed for each user (python setup.py install --user).

### creation

- **simulateWithSEDs.py**: In **TUTORIAL MODE** it generates mock dataset inside Delight. These data are the traning and validation samples of fluxes-fluxeserrors-redshift drawing ramdomly the type of the underlying SED template Fluxes-Redshifts model build in **processSEDs.py**. These data are written in the ascii files **'trainingFile'** and **'targetFile'** (which true name is given in the configuration parameter file).

### Learn

- **delightLearn.py**:	Build the Gaussian Process from the training data. The Gaussian process is written in the file **'training_paramFile'**.
- whether one is running in **TUTORIAL MODE** and **STANDARD MODE**, all the training data are processed in one step.

- Note for template estimation of redshift, there is no learning phase. 

### Estimation	


							
- **templateFitting.py**: The Gaussian Process estimation is also based on latent SED templates. These lated SED template are only usefull for validation dataset to help the GP to estimate the redshift. That is why this module is called from RAIL estimation function (note Learn). Namely it writes the file **'redshiftpdfFileTemp'**.


- **delightApply.py**:	Estimate the redshift posterior distribution using Gaussian process and writes its results in file  **'globalPDFs'**. 						

## Other Delight modules

- **calibrateTemplateMixturePriors.py**: This module must be used outside RAIL. It is only tool to check. 


## Python modules for interface purposes


- **makeConfigParam.py**: it generates Delight configuration file **parametersTest.cfg** from Delight configuration file **delightPZ.yaml**. In the case of RAIL estimation, the validation dataset is split into different chunks. In the later case, there must be is one Delight configuration file per chunk. A Delight configuration file must contain all datafile paths used by Delight.

- **convertDESCcat.py**: Convert DESC PhotoZ dataset into ascii files readable by Delight. For now, only files with hdf5 format are used.

- **getDelightRedshiftEstimation.py**: Retrieve results on PhotoZ estimation of Delight for RAIL. 


## Metrics
- Delight builds its own metrics in **templateFitting.py** and **delightApply.py**. Those modules write their metric files results in **'metricsFileTemp'** and **'globalMetrics'** respectively.
- Because for DC2, the validation datasetp is split in chunks, it is better to use upward RAIL metrics.

## Utilities

- **utils.py** : Library of IO inside RAIL used in **convertDESCcat.py** for reading DESC dataset. (It means it has been copied from RAIL to Delight bacause it is not possible to make cross dependence between these packages (Delight depending on RAIL and RAIL depending on Delight).


## Optimisation

Like any machine learning program, Delight has hyper parameters that must be tuned trained on and optimized according a metric.
Thes hyperparameter are provided in the **delightPZ.yaml** file: 


- Delight hyper-parameters that must be optimized

   zPriorSigma: 0.2
   
   ellPriorSigma: 0.5
   
   fluxLuminosityNorm: 1.0
  
   alpha_C: 1.0e3
 
   V_C: 0.1

   alpha_L: 1.0e2

   V_L: 0.1
   
   lineWidthSigma: 20


This work has not been done on test-DC2 data up to now. It will be the next step. Perhaps RAIL is the good framework to do this task.
 

