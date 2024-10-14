
import jax.numpy as jnp
from jax import jit,lax
from jax import vmap
from functools import partial
from jax.scipy.special import logsumexp


# find_positions:
# - lax.fori_loop: Replaces the for loop. It is used to scroll through the grid indices, and returns the correct index when a condition is met.
# - cond_fun and body_fun: Are defined to manage conditions in the loop and index incrementation.
# - vmap: Is always used to vectorize the find_position_single function in order to apply the position search to each element of fz1.
# - In this way, the code remains differentiable and compatible with JAX transformations, while avoiding classic Python loops.

def find_position_single(fz_val, fzGrid):
    nz = fzGrid.shape[0]

    # Fonction which check if fz_val is between fzGrid[i] and fzGrid[i+1]
    def cond_fun(i):
        #return jnp.logical_or(fz_val >= fzGrid[i], fz_val <= fzGrid[i+1])
        #return jnp.logical_and(fz_val >= fzGrid[i], fz_val <= fzGrid[i+1])
        return jnp.logical_not(jnp.logical_and(fz_val >= fzGrid[i], fz_val <= fzGrid[i+1]))

    # op body function: simply update index i
    def body_fun(i):
        return i + 1

    # lax.while_loop with out condition
    i = lax.while_loop(lambda i: cond_fun(i) & (i < nz - 1), body_fun, 0)
    return i

@jit
def find_positions(fz1, fzGrid):
    """_find the position indexes of the fz1 values inside the grid fZGrid_

    :param fz1: Array of 1+z positions (of a series of objects for exampke)
    :type fz1: jnp.array
    :param fzGrid: Grid for 1+z values (in increasing values)
    :type fzGrid: jnp.array
    :return: The array of indexes of the fz1 values in the grid fzGrid
    :rtype: jnp.array
    """
    # Applique la recherche sur chaque élément de fz1
    positions = vmap(lambda fz_val: find_position_single(fz_val, fzGrid))(fz1)
    return positions



@jit
def bilininterp_precomputedbins(numBands, nobj, Kinterp, v1s, v2s, p1s, p2s, grid1, grid2, Kgrid):
    """
    Performs bilinear interpolation using precomputed bins in JAX.
    See the formula for bilinear interpilation at https://en.wikipedia.org/wiki/Bilinear_interpolation
    Notice : I am not sure that Kgrid is a 3D matrix (numband, pos-z1, pos z2)
    probably this function in deprecated
    
    Args:
    - numBands (int): Number of bands.
    - nobj (int): Number of objects to interpolate.
    - Kinterp (array): Output array (numBands x nobj) for interpolated values.
    - v1s (array): Array (nobj) of values in grid1.
    - v2s (array): Array (nobj) of values in grid2.
    - p1s (array): Array (nobj) of positions in grid1.
    - p2s (array): Array (nobj) of positions in grid2.
    - grid1 (array): The 1D grid1 array.
    - grid2 (array): The 1D grid2 array.
    - Kgrid (array): 3D array (numBands x grid1_size x grid2_size) of grid values.
    
    Returns:
    - Kinterp (array): Updated with interpolated values.
    """
    
    # This function will calculate for a single object
    def interp_single(o, Kinterp):
        p1 = p1s[o]
        p2 = p2s[o]
        v1 = v1s[o]
        v2 = v2s[o]
        
        # Compute dzm2 (normalization factor for bilinear interpolation)
        dzm2 = 1. / ((grid1[p1 + 1] - grid1[p1]) * (grid2[p2 + 1] - grid2[p2]))

        # Loop over the bands (use lax.fori_loop for efficient loop over numBands)
        def inner_loop(b, Kinterp):
            Kinterp = Kinterp.at[b, o].set(dzm2 * (
                (grid1[p1 + 1] - v1) * (grid2[p2 + 1] - v2) * Kgrid[b, p1, p2]
                + (v1 - grid1[p1]) * (grid2[p2 + 1] - v2) * Kgrid[b, p1 + 1, p2]
                + (grid1[p1 + 1] - v1) * (v2 - grid2[p2]) * Kgrid[b, p1, p2 + 1]
                + (v1 - grid1[p1]) * (v2 - grid2[p2]) * Kgrid[b, p1 + 1, p2 + 1]
            ))
            return Kinterp
        
        # Use lax.fori_loop for numBands
        Kinterp = lax.fori_loop(0, numBands, inner_loop, Kinterp)
        return Kinterp

    # Use lax.fori_loop for nobj (objects)
    Kinterp = lax.fori_loop(0, nobj, interp_single, Kinterp)
    
    return Kinterp



def kernel_parts_interp(
    NO1, NO2, 
    Kinterp, 
    b1, b2, 
    fz1, fz2, 
    p1s, p2s, 
    fzGrid, 
    Kgrid
):
    """Function to interpolate the Kernel between left-type objects (to test or evaluate) and right-type objects for the training

    :param NO1: Number of objects of type 1 (left type : test)
    :type NO1: int
    :param NO2: Number of objects of type 2 (left type: training)
    :type NO2: int
    :param Kinterp: _description_
    :type Kinterp: _type_
    :param b1: _description_
    :type b1: _type_
    :param b2: _description_
    :type b2: _type_
    :param fz1: _description_
    :type fz1: _type_
    :param fz2: _description_
    :type fz2: _type_
    :param p1s: _description_
    :type p1s: _type_
    :param p2s: _description_
    :type p2s: _type_
    :param fzGrid: _description_
    :type fzGrid: _type_
    :param Kgrid: _description_
    :type Kgrid: _type_
    :return: _description_
    :rtype: _type_

    f(x,y) =
      1/(x1-x2)/(y1-y2)*
      ((x2-x)(y2-y)*f(Q11)+(x-x1)(y2-y)f(Q21)+f(Q12)(x2-x)(y-y1)+f(Q22)(x-x1)(y-y1))
    
    )
    """
    # Fonction qui calcule une interpolation pour un couple d'objets
    def interp_single(o1, o2):
        # Extraction des valeurs nécessaires
        opz1 = fz1[o1] # (1+z) value for object 1 (left type)
        p1 = p1s[o1]   # position of fz1 in the fzGrid grid of (1+z)
        opz2 = fz2[o2] # (1+z) value for object 2 (right type)
        p2 = p2s[o2]   # position of fz2 in the fzGrid grid of (1+z)
        
        # Vérification des indices pour éviter les erreurs d'indexation
        #p1 = int(p1)  # Assure-toi que p1 et p2 sont des entiers
        #p2 = int(p2)
        
        dzm2 = 1. / (fzGrid[p1 + 1] - fzGrid[p1]) / (fzGrid[p2 + 1] - fzGrid[p2])
        
        # Calcul du résultat de l'interpolation
        result = dzm2 * (
            (fzGrid[p1 + 1] - opz1) * (fzGrid[p2 + 1] - opz2) * Kgrid[b1[o1], b2[o2], p1, p2]
            + (opz1 - fzGrid[p1]) * (fzGrid[p2 + 1] - opz2) * Kgrid[b1[o1], b2[o2], p1 + 1, p2]
            + (fzGrid[p1 + 1] - opz1) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1, p2 + 1]
            + (opz1 - fzGrid[p1]) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1 + 1, p2 + 1]
        )
        
        return result

    # Vectorisation avec vmap pour appliquer l'interpolation sur toutes les paires (o1, o2)
    Kinterp = vmap(lambda o1: vmap(lambda o2: interp_single(o1, o2))(jnp.arange(NO2)))(jnp.arange(NO1))

    return Kinterp



@partial(jit, static_argnums=(0,1))
def kernel_parts_interp_jax(
    NO1, NO2, 
    Kinterp, 
    b1, b2, 
    fz1, fz2, 
    p1s, p2s, 
    fzGrid, 
    Kgrid
):
    """Interpolates the kernel between objects of type 1 (test) and objects of type 2 (training).

    Args:
        NO1 (int): Number of objects of type 1 (test objects).
        NO2 (int): Number of objects of type 2 (training objects).
        b1 (array): Band indices for objects of type 1.
        b2 (array): Band indices for objects of type 2.
        fz1 (array): (1 + z) values for objects of type 1.
        fz2 (array): (1 + z) values for objects of type 2.
        p1s (array): Indices in fzGrid for fz1 values.
        p2s (array): Indices in fzGrid for fz2 values.
        fzGrid (array): Grid of (1 + z) values.
        Kgrid (array): Kernel grid of shape (numBands1, numBands2, nz1, nz2).

    Returns:
        array: Interpolated kernel values of shape (NO1, NO2).
    """

    # Function to compute interpolation for a single pair of objects (o1, o2)
    def interp_single(o1, o2):
        opz1 = fz1[o1]
        p1 = p1s[o1]
        opz2 = fz2[o2]
        p2 = p2s[o2]
        
        dzm2 = 1. / (fzGrid[p1 + 1] - fzGrid[p1]) / (fzGrid[p2 + 1] - fzGrid[p2])
        
        result = dzm2 * (
            (fzGrid[p1 + 1] - opz1) * (fzGrid[p2 + 1] - opz2) * Kgrid[b1[o1], b2[o2], p1, p2]
            + (opz1 - fzGrid[p1]) * (fzGrid[p2 + 1] - opz2) * Kgrid[b1[o1], b2[o2], p1 + 1, p2]
            + (fzGrid[p1 + 1] - opz1) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1, p2 + 1]
            + (opz1 - fzGrid[p1]) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1 + 1, p2 + 1]
        )
        
        return result

    # Apply vectorization with vmap over NO1 and NO2 objects
    Kinterp = vmap(lambda o1: vmap(lambda o2: interp_single(o1, o2))(jnp.arange(NO2)))(jnp.arange(NO1))

    return Kinterp





@jit
def approx_flux_likelihood_jax(f_obs, f_obs_var, f_mod, f_mod_covar, ell_hat, ell_var, niter=2):
    """Compute likelihood function for fluxes

    TBC : To Be checked

    :param f_obs: _description_
    :type f_obs: _type_
    :param f_obs_var: _description_
    :type f_obs_var: _type_
    :param f_mod: _description_
    :type f_mod: _type_
    :param f_mod_covar: _description_
    :type f_mod_covar: _type_
    :param ell_hat: _description_
    :type ell_hat: _type_
    :param ell_var: _description_
    :type ell_var: _type_
    :param niter: _description_, defaults to 2
    :type niter: int, optional
    :return: _description_
    :rtype: _type_
    """
    nz, nt, nf = f_mod.shape[:3]  # Dimensions

    # Fonction qui réalise une itération de calcul pour une combinaison (i_z, i_t)
    def compute_likelihood(i_z, i_t):

        # Initialisation de ellML à 0.0
        def body_fun(i, ellML_FOT_FTT_FOO):
            ellML, FOT, FTT, FOO, logDenom = ellML_FOT_FTT_FOO

            # Mise à jour des valeurs FOT, FTT, FOO pour une itération
            def update_fot_ftt_foo(i_f, vals):
                ellML, FOT, FTT, FOO, logDenom = vals
                var = f_obs_var[i_f] + ellML ** 2 * f_mod_covar[i_z, i_t, i_f]
                FOT += f_mod[i_z, i_t, i_f] * f_obs[i_f] / var
                FTT += (f_mod[i_z, i_t, i_f] ** 2) / var
                FOO += (f_obs[i_f] ** 2) / var
                logDenom += jnp.where(i == niter - 1, jnp.log(var * 2 * jnp.pi), 0.0)
                return ellML, FOT, FTT, FOO, logDenom

            # Application de la mise à jour pour chaque i_f
            ellML, FOT, FTT, FOO, logDenom = lax.fori_loop(0, nf, update_fot_ftt_foo, (ellML, FOT, FTT, FOO, logDenom))

            # Calcul de la nouvelle valeur de ellML
            ellML = FOT / FTT
            return ellML, FOT, FTT, FOO, logDenom

        # Boucle externe sur le nombre d'itérations niter
        ellML_FOT_FTT_FOO = (0.0, ell_hat[i_z] / ell_var[i_z], 1. / ell_var[i_z], ell_hat[i_z] ** 2 / ell_var[i_z], 0.0)
        ellML, FOT, FTT, FOO, logDenom = lax.fori_loop(0, niter, body_fun, ellML_FOT_FTT_FOO)

        # Calcul final du chi2 et du likelihood
        chi2 = FOO - (FOT ** 2) / FTT
        logDenom += jnp.log(2 * jnp.pi * ell_var[i_z])
        logDenom += jnp.log(FTT / (2 * jnp.pi))

        return -0.5 * chi2 - 0.5 * logDenom

    # Vectorisation avec vmap pour parcourir les dimensions nz et nt
    likelihoods = vmap(lambda i_z: vmap(lambda i_t: compute_likelihood(i_z, i_t))(jnp.arange(nt)))(jnp.arange(nz))

    # Normalisation pour éviter les overflow
    loglikemax = jnp.max(likelihoods)
    normalized_likelihoods = jnp.exp(likelihoods - loglikemax)

    return normalized_likelihoods



# Gaussian probability function
@jit
def gauss_prob(x, mu, var):
    """Compute the Gaussian probability."""
    return jnp.exp(-0.5 * jnp.power(x - mu, 2.) / var) / jnp.sqrt(2. * jnp.pi * var)

# Log of Gaussian probability function
@jit
def gauss_lnprob(x, mu, var):
    """Compute the log of the Gaussian probability."""
    return -0.5 * jnp.power(x - mu, 2) / var - 0.5 * jnp.log(2 * jnp.pi * var)

# LogSumExp function
@jit
def logsumexp(arr):
    """Compute the log-sum-exp of the input array."""
    largest_in_a = jnp.max(arr)
    return largest_in_a + jnp.log(jnp.sum(jnp.exp(arr - largest_in_a)))

