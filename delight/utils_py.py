# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# from cython.parallel import prange
# from cpython cimport bool
# cimport cython
# from libc.math cimport sqrt, M_PI, exp, pow, log
# from libc.stdlib cimport malloc, free
from math import M_PI, exp, log, pow

import numpy as np

# M_PI = np.pi


def find_positions(
    NO1: int,
    nz: int,
    fz1: np.ndarray,  # array of long
    p1s: np.ndarray,  # array of long
    fzGrid: np.ndarray,  # array of double
) -> np.ndarray:
    """Find the position of the redshift in a grid
       Note the size is defined outside this function
    Args:
        NO1 (int): Size of the input array
        nz (int): Number of bins in redshift
        fz1 (np.ndarray): 1D-array of redshift
        ps1 (np.ndarray): 1D-array of positions (indexed by long integer)
        fzGrid (np.ndarray) : 1D array of 1+z

    Returns:
        np.ndarray: output ps1 filled with the position of the redshift in the redshift grid
    """

    # cdef long p1, o1
    for o1 in range(NO1):
        for p1 in range(nz - 1):
            if fz1[o1] >= fzGrid[p1] and fz1[o1] <= fzGrid[p1 + 1]:
                p1s[o1] = p1
                break
    return p1s


# Not existing function
# def bilininterp_precomputedbins(
#            int numBands, int nobj,
#            double[:, :] Kinterp, # nbands x nobj
#            double[:] v1s, # nobj (val in grid1)
#            double[:] v2s, # nobj (val in grid1)
#            long[:] p1s, # nobj (pos in grid1)
#            long[:] p2s, # nobj (pos in grid2)
#            double[:] grid1,
#            double[:] grid2,
#            double[:, :, :] Kgrid): # nbands x ngrid1 x ngrid2

#    cdef int p1, p2, o, b
#    cdef double dzm2, v1, v2
#    for o in prange(nobj, nogil=True):
#        p1 = p1s[o]
#        p2 = p2s[o]
#        v1 = v1s[o]
#        v2 = v2s[o]
#        dzm2 = 1. / (grid1[p1+1] - grid1[p1]) / (grid2[p2+1] - grid2[p2])
#        for b in range(numBands):
#            Kinterp[b, o] = dzm2 * (
#                (grid1[p1+1] - v1) * (grid2[p2+1] - v2) * Kgrid[b, p1, p2]
#                + (v1 - grid1[p1]) * (grid2[p2+1] - v2) * Kgrid[b, p1+1, p2]
#                + (grid1[p1+1] - v1) * (v2 - grid2[p2]) * Kgrid[b, p1, p2+1]
#                + (v1 - grid1[p1]) * (v2 - grid2[p2]) * Kgrid[b, p1+1, p2+1]
#                )


def kernel_parts_interp(
    NO1: int,
    NO2: int,
    Kinterp: np.ndarray,  # 2D- array long
    b1: np.ndarray,  # array of long
    fz1: np.ndarray,  # array of double
    p1s: np.ndarray,  # array of long
    b2: np.ndarray,  # array of long
    fz2: np.ndarray,  # array of double
    p2s: np.ndarray,  # array of long
    fzGrid: np.ndarray,  # 1D array of double
    Kgrid: np.ndarray,
) -> np.ndarray:
    """_summary_

    Args:
        NO1 (int): _description_
        NO2 (int): _description_
        Kinterp (np.ndarray): _description_

    Returns:
        np.ndarray:  Kgrid):  double[:,:,:,:] Kgrid 4D dimenssional array of double
    """

    # cdef int p1, p2, o1, o2
    # cdef double dzm2, opz1, opz2
    for o1 in range(NO1):
        opz1 = fz1[o1]
        p1 = p1s[o1]
        for o2 in range(NO2):
            opz2 = fz2[o2]
            p2 = p2s[o2]
            dzm2 = 1.0 / (fzGrid[p1 + 1] - fzGrid[p1]) / (fzGrid[p2 + 1] - fzGrid[p2])
            Kinterp[o1, o2] = dzm2 * (
                (fzGrid[p1 + 1] - opz1) * (fzGrid[p2 + 1] - opz2) * Kgrid[b1[o1], b2[o2], p1, p2]
                + (opz1 - fzGrid[p1]) * (fzGrid[p2 + 1] - opz2) * Kgrid[b1[o1], b2[o2], p1 + 1, p2]
                + (fzGrid[p1 + 1] - opz1) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1, p2 + 1]
                + (opz1 - fzGrid[p1]) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1 + 1, p2 + 1]
            )
    return Kinterp


def approx_flux_likelihood_cy(
    like: np.ndarray,
    nz: long,
    nt: long,
    nf: long,
    f_obs: np.ndarray,
    f_obs_var: np.ndarray,
    f_mod: np.ndarray,
    f_mod_covar: np.ndarrray,
    ell_hat: np.ndarray,
    ell_var: np.ndarray,
) -> np.ndarray:
    """_summary_

    Args:
        like (np.ndarray): _description_
        nz (long): _description_
        nt (long): _description_
        nf (long): _description_
        f_obs (np.ndarray): _description_
        f_obs_var (np.ndarray): _description_
        f_mod (np.ndarray): _description_
        f_mod_covar (np.ndarrray): _description_
        ell_hat (np.ndarray): _description_
        ell_var (np.ndarray): _description_
    """

    #   double [:, :] like,  # nz, nt
    #   long nz,
    #    long nt,
    #    long nf,
    #    double[:] f_obs,  # nf
    #    double[:] f_obs_var,  # nf
    #    double[:,:,:] f_mod,  # nz, nt, nf
    #    double[:,:,:] f_mod_covar, # nz, nt, nf
    #    double[:] ell_hat, # 1
    #    double[:] ell_var # 1

    # cdef long i, i_t, i_z, i_f,
    niter = 2
    # cdef double var, FOT, FTT, FOO, chi2, ellML, logDenom, loglikemax
    for i_z in range(nz):
        for i_t in range(nt):
            ellML = 0
            for i in range(niter):
                FOT = ell_hat[i_z] / ell_var[i_z]
                FTT = 1.0 / ell_var[i_z]
                FOO = ell_hat[i_z] ** 2 / ell_var[i_z]
                logDenom = 0
                for i_f in range(nf):
                    var = f_obs_var[i_f] + ellML**2 * f_mod_covar[i_z, i_t, i_f]
                    FOT = FOT + f_mod[i_z, i_t, i_f] * f_obs[i_f] / var
                    FTT = FTT + pow(f_mod[i_z, i_t, i_f], 2) / var
                    FOO = FOO + pow(f_obs[i_f], 2) / var
                    if i == niter - 1:
                        logDenom = logDenom + log(var * 2 * M_PI)
                ellML = FOT / FTT
                if i == niter - 1:
                    chi2 = FOO - pow(FOT, 2) / FTT
                    logDenom = logDenom + log(2 * M_PI * ell_var[i_z])
                    logDenom = logDenom + log(FTT / (2 * M_PI))
                    like[i_z, i_t] = -0.5 * chi2 - 0.5 * logDenom  # nz * nt

    if True:
        loglikemax = like[0, 0]
        for i_z in range(nz):
            for i_t in range(nt):
                if like[i_z, i_t] > loglikemax:
                    loglikemax = like[i_z, i_t]
        for i_z in range(nz):
            for i_t in range(nt):
                like[i_z, i_t] = exp(like[i_z, i_t] - loglikemax)
    return like


# cdef double gauss_prob(double x, double mu, double var) nogil:
def gauss_prob(x: double, mu: double, var: double) -> double:
    """_summary_

    Args:
        doublex (double): _description_
        mu (double): _description_
        var (double): _description_

    Returns:
        double: _description_
    """
    return exp(-0.5 * pow(x - mu, 2.0) / var) / sqrt(2.0 * M_PI * var)


# cdef double gauss_lnprob(double x, double mu, double var) nogil:
def gauss_lnprob(x: double, mu: double, var: double) -> double:
    """_summary_

    Args:
        x (double): _description_
        mu (double): _description_
        var (double): _description_

    Returns:
        double: _description_
    """
    return -0.5 * pow(x - mu, 2) / var - 0.5 * log(2 * M_PI * var)


# cdef double logsumexp(double* arr, long dim) nogil:
def logsumexp(arr: np.ndarray, dim: long) -> float:
    """_summary_

    Args:
        arr (np.ndarray): _description_
        dim (long): _description_

    Returns:
        double: _description_
    """
    # cdef int i
    result = 0.0
    largest_in_a = arr[0]
    for i in range(1, dim):
        if arr[i] > largest_in_a:
            largest_in_a = arr[i]
    for i in range(dim):
        result += exp(arr[i] - largest_in_a)
    return largest_in_a + log(result)


def photoobj_evidences_marglnzell(
    logevidences: np.ndarray,
    alphas: np.ndarray,
    nobj: long,
    numTypes: long,
    nz: long,
    nf: long,
    f_obs: np.ndarray,
    f_obs_var: np.ndarray,
    f_mod: np.ndarray,
    z_grid_centers: np.ndarray,
    z_grid_sizes: np.ndarray,
    mu_ell: np.ndarray,
    mu_lnz: np.ndarray,
    var_ell: np.ndarray,
    var_lnz: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    #
    # double [:] logevidences, # nobj
    # double [:] alphas, # nt
    # long nobj, long numTypes, long nz, long nf,
    # double [:, :] f_obs,  # nobj * nf
    # double [:, :] f_obs_var,  # nobj * nf
    # double [:, :, :] f_mod,  # nt * nz * nf
    # double [:] z_grid_centers, # nz
    # double [:] z_grid_sizes, # nz
    # double [:] mu_ell, # nt
    # double [:] mu_lnz, double [:] var_ell,  # nt
    # double [:] var_lnz, double [:] rho  # nt
    # cdef long o, i_t, i_z, i_f
    # cdef double FOT, FTT, FOO, chi2, ellML, logDenom
    # cdef double mu_ell_prime, var_ell_prime, lnprior_lnz
    # cdef double *logpost

    for o in range(nobj):  # prange(nobj, nogil=True):
        # logpost = <double *> malloc(sizeof(double) * (nz*numTypes))
        logpost = np.zeros(nz * numTypes,dtype = np.float64)
        for i_z in range(nz):
            for i_t in range(numTypes):
                mu_ell_prime = (
                    mu_ell[i_t] + rho[i_t] * (log(z_grid_centers[i_z]) - mu_lnz[i_t]) / var_lnz[i_t]
                )
                var_ell_prime = var_ell[i_t] - pow(rho[i_t], 2) / var_lnz[i_t]
                FOT = mu_ell_prime / var_ell_prime
                FTT = 1.0 / var_ell_prime
                FOO = pow(mu_ell_prime, 2) / var_ell_prime
                logDenom = 0
                for i_f in range(nf):
                    FOT = FOT + f_mod[i_t, i_z, i_f] * f_obs[o, i_f] / f_obs_var[o, i_f]
                    FTT = FTT + pow(f_mod[i_t, i_z, i_f], 2) / f_obs_var[o, i_f]
                    FOO = FOO + pow(f_obs[o, i_f], 2) / f_obs_var[o, i_f]
                    logDenom = logDenom + log(f_obs_var[o, i_f] * 2 * M_PI)
                # ellML = FOT / FTT
                chi2 = FOO - pow(FOT, 2) / FTT
                logDenom = logDenom + log(var_ell_prime) + log(FTT)
                lnprior_lnz = gauss_lnprob(log(z_grid_centers[i_z]), mu_lnz[i_t], var_lnz[i_t])
                logpost[i_t * nz + i_z] = (
                    log(alphas[i_t]) - 0.5 * chi2 - 0.5 * logDenom + lnprior_lnz + log(z_grid_sizes[i_z])
                )

        for i_t in range(numTypes):
            logevidences[o] = logsumexp(logpost, nz * numTypes)

        # free(logpost)


def specobj_evidences_margell(
    logevidences: np.ndarray,
    alphas: np.ndarray,
    nobj: np.ndarray,
    numTypes: np.ndarray,
    nf: np.ndarray,
    f_obs: np.ndarray,
    f_obs_var: np.ndarray,
    f_mod: np.ndarray,
    redshifts: np.ndarray,
    mu_ell: np.ndarray,
    mu_lnz: np.ndarray,
    var_ell: np.ndarray,
    var_lnz: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    # double [:] logevidences, # nobj
    # double [:] alphas, # nt
    # long nobj, long numTypes, long nf,
    # double [:, :] f_obs,  # nobj * nf
    # double [:, :] f_obs_var,  # nobj * nf
    # double [:, :, :] f_mod,  # nt * nobj * nf
    # double [:] redshifts, # nobj
    # double [:] mu_ell, # nt
    # double [:] mu_lnz, double [:] var_ell,  # nt
    # double [:] var_lnz, double [:] rho  # nt
    #    ):

    # cdef long o, i_t, i_f
    # cdef double FOT, FTT, FOO, chi2, ellML, logDenom
    # cdef double mu_ell_prime, var_ell_prime, lnprior_lnz
    # cdef double *logpost

    for o in range(nobj):  # prange(nobj, nogil=True):
        # logpost = <double *> malloc(sizeof(double) * (numTypes))
        logpost = np.zeros(numTypes ,dtype=np.float64)
        for i_t in range(numTypes):
            mu_ell_prime = mu_ell[i_t] + rho[i_t] * (log(redshifts[o]) - mu_lnz[i_t]) / var_lnz[i_t]
            var_ell_prime = var_ell[i_t] - pow(rho[i_t], 2) / var_lnz[i_t]
            FOT = mu_ell_prime / var_ell_prime
            FTT = 1.0 / var_ell_prime
            FOO = pow(mu_ell_prime, 2) / var_ell_prime
            logDenom = 0
            for i_f in range(nf):
                FOT = FOT + f_mod[i_t, o, i_f] * f_obs[o, i_f] / f_obs_var[o, i_f]
                FTT = FTT + pow(f_mod[i_t, o, i_f], 2) / f_obs_var[o, i_f]
                FOO = FOO + pow(f_obs[o, i_f], 2) / f_obs_var[o, i_f]
                logDenom = logDenom + log(f_obs_var[o, i_f] * 2 * M_PI)
            # ellML = FOT / FTT
            chi2 = FOO - pow(FOT, 2) / FTT
            logDenom = logDenom + log(var_ell_prime) + log(FTT)
            lnprior_lnz = gauss_lnprob(log(redshifts[o]), mu_lnz[i_t], var_lnz[i_t])
            logpost[i_t] = log(alphas[i_t]) - 0.5 * chi2 - 0.5 * logDenom + lnprior_lnz

        for i_t in range(numTypes):
            logevidences[o] = logsumexp(logpost, numTypes)

        # free(logpost)
    return logevidences


def photoobj_lnpost_zgrid_margell(
    lnpost: np.ndarray,
    alphas: np.ndarray,
    nobj: long,
    numTypes: long,
    nz: long,
    nf: long,
    f_obs: np.ndarray,
    f_obs_var: np.ndarray,
    f_mod: np.ndarray,
    z_grid_centers: np.ndarray,
    z_grid_sizes: np.ndarray,
    mu_ell: np.ndarray,
    mu_lnz: np.ndarray,
    var_ell: np.ndarray,
    var_lnz: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """_summary_

    Args:
        lnpost (np.ndarray): _description_
        alphas (np.ndarray): _description_
        nobj (long): _description_
        numTypes (long): _description_
        nz (long): _description_
        nf (long): _description_
        f_obs (np.ndarray): _description_
        f_obs_var (np.ndarray): _description_
        f_mod (np.ndarray): _description_
        z_grid_centers (np.ndarray): _description_
        z_grid_sizes (np.ndarray): _description_
        mu_ell (np.ndarray): _description_
        mu_lnz (np.ndarray): _description_
        var_ell (np.ndarray): _description_
        var_lnz (np.ndarray): _description_
        rho (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    # double [:, :, :] lnpost, # nobj * nt * nz
    # double [:] alphas, # nt
    # long nobj, long numTypes, long nz, long nf,
    # double [:, :] f_obs,  # nobj * nf
    # double [:, :] f_obs_var,  # nobj * nf
    # double [:, :, :] f_mod,  # nt * nz * nf
    # double [:] z_grid_centers, # nz
    # double [:] z_grid_sizes, # nz
    # double [:] mu_ell, # nt
    # double [:] mu_lnz, double [:] var_ell,  # nt
    # double [:] var_lnz, double [:] rho  # nt

    # cdef long o, i_t, i_z, i_f
    # cdef double FOT, FTT, FOO, chi2, ellML, logDenom
    # cdef double mu_ell_prime, var_ell_prime, lnprior_lnz

    # for o in prange(nobj, nogil=True):
    for o in range(nobj):
        for i_z in range(nz):
            for i_t in range(numTypes):
                mu_ell_prime = (
                    mu_ell[i_t] + rho[i_t] * (log(z_grid_centers[i_z]) - mu_lnz[i_t]) / var_lnz[i_t]
                )
                var_ell_prime = var_ell[i_t] - pow(rho[i_t], 2) / var_lnz[i_t]
                FOT = mu_ell_prime / var_ell_prime
                FTT = 1.0 / var_ell_prime
                FOO = pow(mu_ell_prime, 2) / var_ell_prime
                logDenom = 0
                for i_f in range(nf):
                    FOT = FOT + f_mod[i_t, i_z, i_f] * f_obs[o, i_f] / f_obs_var[o, i_f]
                    FTT = FTT + pow(f_mod[i_t, i_z, i_f], 2) / f_obs_var[o, i_f]
                    FOO = FOO + pow(f_obs[o, i_f], 2) / f_obs_var[o, i_f]
                    logDenom = logDenom + log(f_obs_var[o, i_f] * 2 * M_PI)
                # ellML = FOT / FTT
                chi2 = FOO - pow(FOT, 2) / FTT
                logDenom = logDenom + log(var_ell_prime) + log(FTT)
                lnprior_lnz = gauss_lnprob(log(z_grid_centers[i_z]), mu_lnz[i_t], var_lnz[i_t])
                lnpost[o, i_t, i_z] = log(alphas[i_t]) - 0.5 * chi2 - 0.5 * logDenom + lnprior_lnz

    return lnpost