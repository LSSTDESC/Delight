## pytest -vv tests/test_utils_jx.py 
## Check how to imeplent tests : https://gouarin.github.io/python-packaging-2023/pytest
import pytest
import jax.numpy as jnp
from jax import jit
from delight.utils_jx import find_positions,bilininterp_precomputedbins,kernel_parts_interp
from delight.utils_jx import approx_flux_likelihood_jax, gauss_prob,gauss_lnprob,logsumexp



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


@pytest.mark.skip(reason="TBD test :: test_find_positions_out_of_bounds !!")
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

@pytest.mark.skip(reason="TBD test :: test_find_positions_exact_bounds !!")
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
    v2s = jnp.array([0.5, 1.5])
    p1s = jnp.array([0, 1])
    p2s = jnp.array([0, 1])
    grid1 = jnp.array([0.0, 1.0, 2.0])
    grid2 = jnp.array([0.0, 1.0])
    Kgrid = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.5, 2.5], [3.5, 4.5]]
    ])

    Kinterp_result = bilininterp_precomputedbins(numBands, nobj, Kinterp, v1s, v2s, p1s, p2s, grid1, grid2, Kgrid)

    # Teste si Kinterp a la forme attendue
    assert Kinterp_result.shape == (numBands, nobj), f"Expected shape {(numBands, nobj)} but got {Kinterp_result.shape}"
    
    # Vérifie les valeurs interpolées attendues
    expected_result = jnp.array([
        [2.0, 2.5],  # Exemples de valeurs attendues pour les interpolations de Kinterp
        [2.5, 3.0]
    ])
    assert jnp.allclose(Kinterp_result, expected_result), f"Expected {expected_result} but got {Kinterp_result}"




def test_kernel_parts_interp_jax():
    # Définir les dimensions
    NO1 = 2
    NO2 = 2
    numBands1 = 3
    numBands2 = 3
    nz1 = 4
    nz2 = 4
    
    # Initialisation de Kgrid (avec 4 dimensions)
    Kgrid = jnp.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                       [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
                       [[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]]])

    # Autres entrées
    b1 = jnp.array([0, 1])
    b2 = jnp.array([0, 1])
    fz1 = jnp.array([0.5, 1.5])
    fz2 = jnp.array([1.0, 2.0])
    p1s = jnp.array([0, 1])
    p2s = jnp.array([0, 1])
    fzGrid = jnp.array([0.0, 1.0, 2.0, 3.0])

    # Tableau Kinterp à remplir
    Kinterp = jnp.zeros((NO1, NO2))

    # Appel de la fonction
    result = kernel_parts_interp(NO1, NO2, Kinterp, b1, b2, fz1, fz2, p1s, p2s, fzGrid, Kgrid)

    # Vérification que le résultat n'est pas vide
    assert result.shape == (NO1, NO2), "The shape of the result is incorrect."

    # Test des valeurs interpolées pour certains indices spécifiques
    expected_result = 0.0  # Valeur d'attente, ajuster en fonction du calcul exact.
    assert jnp.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"



def test_kernel_parts_interp_jax_tbt():
    NO1 = 2
    NO2 = 2
    b1 = jnp.array([0, 1])
    b2 = jnp.array([0, 1])
    fz1 = jnp.array([1.1, 2.2])
    fz2 = jnp.array([1.1, 2.2])
    p1s = jnp.array([0, 1])
    p2s = jnp.array([0, 1])
    fzGrid = jnp.array([1.0, 2.0, 3.0])
    Kgrid = jnp.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                       [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]])

    Kinterp = jnp.zeros((NO1, NO2))

    # Appel de la fonction avec des valeurs fictives
    result = kernel_parts_interp_jax(NO1, NO2, Kinterp, b1, b2, fz1, fz2, p1s, p2s, fzGrid, Kgrid)

    # Vérifier que le résultat est du bon type et de la bonne forme
    assert result.shape == (NO1, NO2), "La forme du résultat est incorrecte"
    
    # Vérifier les valeurs retournées (ceci dépendra de votre logique d'implémentation)
    # On pourrait ici comparer à des valeurs attendues calculées à la main
    assert jnp.all(result >= 0), "Certaines valeurs du résultat sont incorrectes"
    
    # Test d'une valeur spécifique si vous avez des attentes précises
    assert jnp.isclose(result[0, 0], 3.0), "La valeur interpolée est incorrecte"
