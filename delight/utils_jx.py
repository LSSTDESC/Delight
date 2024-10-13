
import jax.numpy as jnp
from jax import jit,lax
from jax import vmap


# find_positions:
#    lax.fori_loop : Remplace la boucle for. Elle est utilisée pour parcourir les indices de la grille, et retourne l'indice correct lorsqu'une condition est remplie.
#    cond_fun et body_fun : Sont définies pour la gestion des conditions dans la boucle et l'incrémentation de l'indice.
#    vmap : Est toujours utilisé pour vectoriser la fonction find_position_single afin d'appliquer la recherche de position à chaque élément de fz1.
#    Ainsi, le code reste différentiable et compatible avec les transformations de JAX tout en évitant les boucles classiques de Python.


def find_position_single(fz_val, fzGrid):
    nz = fzGrid.shape[0]

    # Fonction qui vérifie si fz_val se situe entre fzGrid[i] et fzGrid[i+1]
    def cond_fun(i):
        #return jnp.logical_or(fz_val >= fzGrid[i], fz_val <= fzGrid[i+1])
        #return jnp.logical_and(fz_val >= fzGrid[i], fz_val <= fzGrid[i+1])
        return jnp.logical_not(jnp.logical_and(fz_val >= fzGrid[i], fz_val <= fzGrid[i+1]))

    # Fonction de corps de boucle : on met simplement à jour l'indice i
    def body_fun(i):
        return i + 1

    # lax.while_loop avec condition de sortie
    i = lax.while_loop(lambda i: cond_fun(i) & (i < nz - 1), body_fun, 0)
    return i

@jit
def find_positions(fz1, fzGrid):
    """_summary_

    :param fz1: _description_
    :type fz1: _type_
    :param fzGrid: _description_
    :type fzGrid: _type_
    :return: _description_
    :rtype: _type_
    """
    # Applique la recherche sur chaque élément de fz1
    positions = vmap(lambda fz_val: find_position_single(fz_val, fzGrid))(fz1)
    return positions



@jit
def bilininterp_precomputedbins(numBands, nobj, Kinterp, v1s, v2s, p1s, p2s, grid1, grid2, Kgrid):
    """
    Performs bilinear interpolation using precomputed bins in JAX.
    
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
    # Fonction qui calcule une interpolation pour un couple d'objets
    def interp_single(o1, o2):
        # Extraction des valeurs nécessaires
        opz1 = fz1[o1]
        p1 = p1s[o1]
        opz2 = fz2[o2]
        p2 = p2s[o2]
        
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


