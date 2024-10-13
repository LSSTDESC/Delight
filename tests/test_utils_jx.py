## pytest -vv tests/test_utils_jx.py 
import pytest
import jax.numpy as jnp
from jax import jit
from delight.utils_jx import find_positions,bilininterp_precomputedbins



def test_find_positions():
    # Exemple de valeurs de fz1 et fzGrid
    fz1 = jnp.array([0.5, 1.5, 2.5])
    fzGrid = jnp.array([0.0, 1.0, 2.0, 3.0])

    # Positions attendues (indices dans fzGrid)
    expected_positions = jnp.array([0, 1, 2])

    # Calcul des positions avec la fonction
    positions = find_positions(fz1, fzGrid)

    # Vérification que les positions calculées correspondent aux positions attendues
    assert jnp.array_equal(positions, expected_positions), f"Expected {expected_positions}, but got {positions}"

def test_find_positions_out_of_bounds():
    # Cas où les valeurs de fz1 sont en dehors de fzGrid
    fz1 = jnp.array([-0.5, 3.5])
    fzGrid = jnp.array([0.0, 1.0, 2.0, 3.0])

    # Indices attendus (en fonction de ce que votre fonction retourne dans ces cas)
    # Ici, on suppose que find_positions retournera l'indice du dernier intervalle dans ces cas limites
    expected_positions = jnp.array([0, 2])

    # Calcul des positions avec la fonction
    positions = find_positions(fz1, fzGrid)

    # Vérification que les positions calculées correspondent aux positions attendues
    assert jnp.array_equal(positions, expected_positions), f"Expected {expected_positions}, but got {positions}"

def test_find_positions_exact_bounds():
    # Cas où fz1 contient des valeurs exactement aux bornes des intervalles
    fz1 = jnp.array([0.0, 1.0, 2.0])
    fzGrid = jnp.array([0.0, 1.0, 2.0, 3.0])

    # Indices attendus
    expected_positions = jnp.array([0, 1, 2])

    # Calcul des positions
    positions = find_positions(fz1, fzGrid)

    # Vérification
    assert jnp.array_equal(positions, expected_positions), f"Expected {expected_positions}, but got {positions}"


# Test 1 : Test avec des valeurs simples
def test_bilininterp_simple():
    numBands = 3
    nobj = 2
    Kinterp = jnp.zeros((numBands, nobj))
    v1s = jnp.array([0.5, 1.5])
    v2s = jnp.array([0.5, 1.5])
    p1s = jnp.array([0, 1])
    p2s = jnp.array([0, 1])
    grid1 = jnp.array([0.0, 1.0, 2.0])
    grid2 = jnp.array([0.0, 1.0, 2.0])
    Kgrid = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.5, 2.5], [3.5, 4.5]],
        [[2.0, 3.0], [4.0, 5.0]]
    ])

    Kinterp_result = bilininterp_precomputedbins(numBands, nobj, Kinterp, v1s, v2s, p1s, p2s, grid1, grid2, Kgrid)

    # Teste si Kinterp a la forme attendue
    assert Kinterp_result.shape == (numBands, nobj), f"Expected shape {(numBands, nobj)} but got {Kinterp_result.shape}"
    
    # Vérifie les valeurs interpolées attendues
    expected_result = jnp.array([
        [2.0, 2.5],  # Exemples de valeurs attendues pour les interpolations de Kinterp
        [2.5, 3.0],
        [3.0, 3.5]
    ])
    assert jnp.allclose(Kinterp_result, expected_result), f"Expected {expected_result} but got {Kinterp_result}"

# Test 2 : Test avec des valeurs plus grandes
def test_bilininterp_large():
    numBands = 5
    nobj = 3
    Kinterp = jnp.zeros((numBands, nobj))
    v1s = jnp.array([0.5, 1.5, 1.0])
    v2s = jnp.array([0.5, 1.5, 0.5])
    p1s = jnp.array([0, 1, 1])
    p2s = jnp.array([0, 1, 0])
    grid1 = jnp.array([0.0, 1.0, 2.0])
    grid2 = jnp.array([0.0, 1.0, 2.0])
    Kgrid = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.5, 2.5], [3.5, 4.5]],
        [[2.0, 3.0], [4.0, 5.0]],
        [[2.5, 3.5], [4.5, 5.5]],
        [[3.0, 4.0], [5.0, 6.0]]
    ])

    Kinterp_result = bilininterp_precomputedbins(numBands, nobj, Kinterp, v1s, v2s, p1s, p2s, grid1, grid2, Kgrid)

    # Teste si Kinterp a la forme attendue
    assert Kinterp_result.shape == (numBands, nobj), f"Expected shape {(numBands, nobj)} but got {Kinterp_result.shape}"

    # Vous pouvez ajuster ces valeurs en fonction du calcul exact d'interpolation attendu
    expected_result = jnp.array([
        [3.5, 4.5, 4.0],  # Valeurs approximatives d'interpolation
        [4.0, 5.0, 4.5],
        [4.5, 5.5, 5.0],
        [5.0, 6.0, 5.5],
        [5.5, 6.5, 6.0]
    ])
    
    assert jnp.allclose(Kinterp_result, expected_result), f"Expected {expected_result} but got {Kinterp_result}"

# Test 3 : Test avec des grilles de tailles différentes
def test_bilininterp_different_grid_sizes():
    numBands = 2
    nobj = 2
    Kinterp = jnp.zeros((numBands, nobj))
    v1s = jnp.array([0.5, 1.5])
    v2s = jnp.ar

