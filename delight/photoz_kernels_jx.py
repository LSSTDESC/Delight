import jax.numpy as jnp
from jax import jit,lax
from jax import vmap
from functools import partial
from jax.scipy.special import logsumexp



# Kernel parts interpolation en JAX
@partial(jit, static_argnums=(0,1))
def kernel_parts_interp_jx(
    NO1, NO2, Kgrid, b1, fz1, p1s, b2, fz2, p2s, fzGrid
):
    # Fonction de calcul d'un élément Kinterp
    def kernel_part(o1, o2):
        # Extraire les valeurs nécessaires
        opz1 = fz1[o1]
        p1 = p1s[o1]
        opz2 = fz2[o2]
        p2 = p2s[o2]
        
        # Calcul de dzm2
        dzm2 = 1. / (fzGrid[p1+1] - fzGrid[p1]) / (fzGrid[p2+1] - fzGrid[p2])

        # Calcul de Kinterp
        part1 = (fzGrid[p1+1] - opz1) * (fzGrid[p2+1] - opz2) * Kgrid[b1[o1], b2[o2], p1, p2]
        part2 = (opz1 - fzGrid[p1]) * (fzGrid[p2+1] - opz2) * Kgrid[b1[o1], b2[o2], p1+1, p2]
        part3 = (fzGrid[p1+1] - opz1) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1, p2+1]
        part4 = (opz1 - fzGrid[p1]) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1+1, p2+1]
        
        return dzm2 * (part1 + part2 + part3 + part4)

    # Créer une grille de tous les indices (o1, o2)
    o1_grid, o2_grid = jnp.meshgrid(jnp.arange(NO1), jnp.arange(NO2), indexing='ij')

    # Vectoriser la fonction sur o1 et o2
    kernel_part_vec = vmap(kernel_part, in_axes=(0, 0))

    # Appliquer la fonction vectorisée sur tous les indices o1 et o2
    Kinterp_result = kernel_part_vec(o1_grid, o2_grid)

    return Kinterp_result



# Fonction principale, avec NO1 en argument statique
@partial(jit, static_argnames=["NO1", "NC", "NL", "grad_needed"])
def kernelparts_diag_jx(NO1, NC, NL, alpha_C, alpha_L, fcoefs_amp, fcoefs_mu, fcoefs_sig, 
                        lines_mu, lines_sig, norms, b1, fz1, grad_needed):
    
    sqrt2pi = jnp.sqrt(2 * jnp.pi)
    
    # Fonction calculant KC et KL pour un o1
    def kernel_for_o1(o1):
        opz1 = fz1[o1]
        opz2 = opz1  # D'après la logique, opz2 = opz1

        norm_factor = norms[b1[o1]] ** 2

        def kc_inner(i, j):
            mu1 = fcoefs_mu[b1[o1], i]
            amp1 = fcoefs_amp[b1[o1], i]
            sig1 = fcoefs_sig[b1[o1], i]
            mu2 = fcoefs_mu[b1[o1], j]
            amp2 = fcoefs_amp[b1[o1], j]
            sig2 = fcoefs_sig[b1[o1], j]

            sigma = jnp.sqrt(opz1**2 * sig2**2 + opz2**2 * sig1**2 + (opz1 * opz2 * alpha_C)**2)

            theexp = amp1 * amp2 * 2 * jnp.pi * sig1 * sig2 * jnp.exp(-0.5 * ((opz1 * mu2 - opz2 * mu1) / sigma)**2) / sigma
            kc_val = alpha_C * theexp
            
            d_alpha_c_val = 0.0
            if grad_needed:
                d_alpha_c_val = theexp * (1 - (alpha_C * opz1 * opz2 / sigma)**2 + 
                                          ((alpha_C * (opz1 * mu2 - opz2 * mu1) * opz1 * opz2) / sigma**4)**2)
            
            return kc_val, d_alpha_c_val

        # Appliquer vmap deux fois sur i et j
        KC_vals, D_alpha_C_vals = vmap(lambda i: vmap(lambda j: kc_inner(i, j))(jnp.arange(NC)))(jnp.arange(NC))

        # Somme des contributions pour KC et D_alpha_C
        KC = KC_vals.sum() / norm_factor
        D_alpha_C = D_alpha_C_vals.sum() / norm_factor
        
        return KC, D_alpha_C

    # Calcul pour KL (vectorisé avec vmap)
    def calc_kl(o1):
        opz1 = fz1[o1]
        opz2 = opz1  # opz2 = opz1 selon la logique du code initial

        def kl_inner(l1, l2):
            mul1 = lines_mu[l1]
            mul2 = lines_mu[l2]
            mu1 = fcoefs_mu[b1[o1], 0]
            amp1 = fcoefs_amp[b1[o1], 0]
            sig1 = fcoefs_sig[b1[o1], 0]
            mu2 = fcoefs_mu[b1[o1], 0]
            amp2 = fcoefs_amp[b1[o1], 0]
            sig2 = fcoefs_sig[b1[o1], 0]
            
            exp_part = jnp.exp(-0.5 * ((mu1 - opz1 * mul1) / sig1)**2 + 
                               ((mu2 - opz2 * mul2) / sig2)**2 + 
                               ((mul1 - mul2) / alpha_L)**2)

            kl_val = 2 * amp1 * amp2 * exp_part
            
            d_alpha_l_val = 0.0
            if grad_needed:
                d_alpha_l_val = 2 * amp1 * amp2 * exp_part * ((mul1 - mul2)**2) / alpha_L**3
            
            return kl_val, d_alpha_l_val

        # Appliquer vmap sur l1 et l2
        KL_vals, D_alpha_L_vals = vmap(lambda l1: vmap(lambda l2: kl_inner(l1, l2))(jnp.arange(l1)))(jnp.arange(NL))

        # Somme des contributions pour KL et D_alpha_L
        KL = KL_vals.sum() / norm_factor
        D_alpha_L = D_alpha_L_vals.sum() / norm_factor

        return KL, D_alpha_L

    # Utilisation de vmap pour vectoriser sur NO1
    KC_out, D_alpha_C_out = vmap(kernel_for_o1)(jnp.arange(NO1))
    KL_out, D_alpha_L_out = vmap(calc_kl)(jnp.arange(NO1))

    return KC_out, KL_out, D_alpha_C_out, D_alpha_L_out


# NO1, NO2, NC, NL sont statiques
@partial(jit, static_argnums=(0,1,2,3))
def kernelparts_jax(NO1, NO2, NC, NL, alpha_C, alpha_L, fcoefs_amp, fcoefs_mu, fcoefs_sig, 
                    lines_mu, lines_sig, norms, b1, fz1, b2, fz2, grad_needed):

    def compute_sigma(opz1, opz2, sig1, sig2):
        return jnp.sqrt(opz1**2 * sig2**2 + opz2**2 * sig1**2 + (opz1 * opz2 * alpha_C)**2)

    def compute_exp(mu1, mu2, amp1, amp2, sig1, sig2, opz1, opz2, sigma):
        return amp1 * amp2 * 2 * jnp.pi * sig1 * sig2 * jnp.exp(-0.5 * ((opz1 * mu2 - opz2 * mu1) / sigma)**2) / sigma

    def compute_kc_inner(i, j, o1, o2):
        mu1 = fcoefs_mu[b1[o1], i]
        amp1 = fcoefs_amp[b1[o1], i]
        sig1 = fcoefs_sig[b1[o1], i]
        mu2 = fcoefs_mu[b2[o2], j]
        amp2 = fcoefs_amp[b2[o2], j]
        sig2 = fcoefs_sig[b2[o2], j]
        opz1 = fz1[o1]
        opz2 = fz2[o2]
        sigma = compute_sigma(opz1, opz2, sig1, sig2)
        exp_term = compute_exp(mu1, mu2, amp1, amp2, sig1, sig2, opz1, opz2, sigma)
        kc_val = alpha_C * exp_term

        d_alpha_c_val, d_alpha_z_val = 0.0, 0.0
        if grad_needed:
            d_alpha_c_val = exp_term * (1 - (alpha_C * opz1 * opz2 / sigma)**2 + 
                                        ((alpha_C * (opz1 * mu2 - opz2 * mu1) * opz1 * opz2) / sigma**4)**2)
            
            d_alpha_z_val = alpha_C * exp_term * ((sig2**2 * opz1 + opz1 * opz2**2 * alpha_C**2) * 
                             ((mu2 * opz1 - mu1 * opz2)**2 / sigma**4 - 1 / sigma**2) - 
                             mu2 * (mu2 * opz1 - mu1 * opz2) / sigma**2)

        return kc_val, d_alpha_c_val, d_alpha_z_val

    def compute_kl_inner(l1, l2, o1, o2):
        opz1 = fz1[o1]
        opz2 = fz2[o2]
        mul1 = lines_mu[l1]
        mul2 = lines_mu[l2]
        mu1 = fcoefs_mu[b1[o1], 0]  # assuming mu1, mu2 only depend on index 0 for simplicity
        amp1 = fcoefs_amp[b1[o1], 0]
        sig1 = fcoefs_sig[b1[o1], 0]
        mu2 = fcoefs_mu[b2[o2], 0]
        amp2 = fcoefs_amp[b2[o2], 0]
        sig2 = fcoefs_sig[b2[o2], 0]

        exp_part = jnp.exp(-0.5 * ((mu1 - opz1 * mul1) / sig1)**2 + 
                            ((mu2 - opz2 * mul2) / sig2)**2 + 
                            ((mul1 - mul2) / alpha_L)**2)

        kl_val = 2 * amp1 * amp2 * exp_part

        d_alpha_l_val = 0.0
        if grad_needed:
            d_alpha_l_val = 2 * amp1 * amp2 * exp_part * ((mul1 - mul2)**2) / alpha_L**3

        return kl_val, d_alpha_l_val

    def compute_kc(o1, o2):
        kc_vals, d_alpha_c_vals, d_alpha_z_vals = vmap(
            lambda i: vmap(
                lambda j: compute_kc_inner(i, j, o1, o2)
            )(jnp.arange(NC))
        )(jnp.arange(NC))

        kc = kc_vals.sum() / (norms[b1[o1]] * norms[b2[o2]])
        d_alpha_c = d_alpha_c_vals.sum() / (norms[b1[o1]] * norms[b2[o2]])
        d_alpha_z = d_alpha_z_vals.sum() / (norms[b1[o1]] * norms[b2[o2]])
        return kc, d_alpha_c, d_alpha_z

    def compute_kl(o1, o2):
        kl_vals, d_alpha_l_vals = vmap(
            lambda l1: vmap(
                lambda l2: compute_kl_inner(l1, l2, o1, o2)
            )(jnp.arange(l1))
        )(jnp.arange(NL))

        kl = kl_vals.sum() / (norms[b1[o1]] * norms[b2[o2]])
        d_alpha_l = d_alpha_l_vals.sum() / (norms[b1[o1]] * norms[b2[o2]])
        return kl, d_alpha_l

    KC_out, D_alpha_C_out, D_alpha_z_out = vmap(lambda o1: vmap(lambda o2: compute_kc(o1, o2))(jnp.arange(NO2)))(jnp.arange(NO1))
    KL_out, D_alpha_L_out = vmap(lambda o1: vmap(lambda o2: compute_kl(o1, o2))(jnp.arange(NO2)))(jnp.arange(NO1))

    return KC_out, KL_out, D_alpha_C_out, D_alpha_L_out, D_alpha_z_out
