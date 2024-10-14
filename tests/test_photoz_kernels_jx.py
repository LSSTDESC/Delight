import pytest
import jax.numpy as jnp
from delight.photoz_kernels_jx import kernel_parts_interp_jx  # Remplacez par le bon chemin d'importation

@pytest.fixture
def setup_data():
    # Initialisation des données d'exemple pour le test
    NO1, NO2 = 3, 4
    Kgrid = jnp.ones((3, 4, 5, 5))  # Exemple de tableau Kgrid 4D
    b1 = jnp.array([0, 1, 2])
    fz1 = jnp.array([0.1, 0.2, 0.3])
    p1s = jnp.array([0, 1, 2])
    b2 = jnp.array([0, 1, 2, 3])
    fz2 = jnp.array([0.1, 0.2, 0.3, 0.4])
    p2s = jnp.array([0, 1, 2, 3])
    fzGrid = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    return NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid

def test_kernel_parts_interp_jx_shape(setup_data):
    NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid = setup_data
    
    # Appel de la fonction
    result = kernel_parts_interp_jx(NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid)
    
    # Vérification de la forme de la sortie
    assert result.shape == (NO1, NO2), f"Shape incorrecte, attendu {(NO1, NO2)}, mais obtenu {result.shape}"

def test_kernel_parts_interp_jx_values(setup_data):
    NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid = setup_data
    
    # Appel de la fonction
    result = kernel_parts_interp_jx(NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid)
    
    # Vérification des valeurs, ici nous vérifions que le résultat est supérieur à zéro (car Kgrid contient des 1)
    assert jnp.all(result >= 0), f"Certaines valeurs sont négatives dans le résultat: {result}"

def test_kernel_parts_interp_jx_output_type(setup_data):
    NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid = setup_data
    
    # Appel de la fonction
    result = kernel_parts_interp_jx(NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid)
    
    # Vérification du type du résultat
    assert isinstance(result, jnp.ndarray), f"Le résultat devrait être de type jnp.ndarray, mais c'est {type(result)}"

def test_kernel_parts_interp_jx_large_input():
    # Test avec un grand jeu de données
    NO1, NO2 = 100, 200
    Kgrid = jnp.ones((NO1, NO2, 5, 5))  # Exemple de tableau Kgrid 4D plus grand
    b1 = jnp.arange(NO1)
    fz1 = jnp.linspace(0.0, 0.5, NO1)
    p1s = jnp.arange(NO1) % 5
    b2 = jnp.arange(NO2)
    fz2 = jnp.linspace(0.0, 0.5, NO2)
    p2s = jnp.arange(NO2) % 5
    fzGrid = jnp.linspace(0.0, 0.5, 6)

    # Appel de la fonction
    result = kernel_parts_interp_jx(NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid)

    # Vérification de la forme pour un grand jeu de données
    assert result.shape == (NO1, NO2), f"Shape incorrecte pour grand jeu de données, attendu {(NO1, NO2)}, mais obtenu {result.shape}"


