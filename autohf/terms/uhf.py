"""
Library of wick contractions for UHF wave functions
This implies the RDMs (one body Green's function)
are in the form [G_up, G_dn]
"""

import numpy as np
import jax.numpy as jnp
import jax

known_terms = ["T", "U", "U1", "U2", "J", "Jheis"]


# one body hop - c^\dagger_i c_j
# slightly special, assumes only one spin species
def rhf_hop(rdm, hops):
  return jnp.sum(rdm[hops[0], hops[1]])


# Hubbard U n_{i\uparrow} n_{j\downarrow}
def uhf_U(rdms, sites):
  dup, ddn = rdms[0].diagonal(), rdms[1].diagonal()
  return dup[sites] @ ddn[sites]


# U1 mixed spin or \sum_\sigma n_{i \sigma}n_{j \sigma'}
def uhf_U1(rdms, ii, jj, ij, ji):
  rho_up, rho_dn = rdms[0], rdms[1]
  return rho_up[ii[0], ii[1]] @ rho_dn[jj[0], jj[1]] + rho_up[jj[0], jj[1]] @ rho_dn[ii[0], ii[1]]


# U2 same spin or \sum_\sigma n{i \sigma} n_{j \sigma}
# split into ii,jj for (i,i),(j,j) and ij,ji for (i,j),(j,i)
def uhf_U2(rdms, ii, jj, ij, ji):
  rho_up, rho_dn = rdms[0], rdms[1]
  E = rho_up[ii[0], ii[1]] @ rho_up[jj[0], jj[1]] + rho_dn[ii[0], ii[1]] @ rho_dn[jj[0], jj[1]]
  # off-diag in spatial
  E -= rho_up[ij[0], ij[1]] @ rho_up[ji[0], ji[1]] + rho_dn[ij[0], ij[1]] @ rho_dn[ji[0], ji[1]]
  return E


# J term (see HK model)
# split into ij,ji
def uhf_J(rdms, ii, jj, ij, ji):
  rho_up, rho_down = rdms[0], rdms[1]
  return (
    rho_up[ij[0], ij[1]] @ rho_down[ji[0], ji[1]]
    + rho_up[ij[0], ij[1]] @ rho_down[ij[0], ij[1]]
    + rho_up[ji[0], ji[1]] @ rho_down[ij[0], ij[1]]
    + rho_up[ji[0], ji[1]] @ rho_down[ji[0], ji[1]]
  )


def uhf_Jheis(rdms, ii, jj, ij, ji):
  rho_up, rho_down = rdms[0], rdms[1]
  return (
    rho_up[ii[0], ii[1]] @ rho_down[jj[0], jj[1]] + rho_up[ji[0], ji[1]] @ rho_down[ij[0], ij[1]]
  )


def uhf_onebody(rdms, onebody_operator):
  """Calculate the UHF expectation value of a onebody operator. The operator can be either spin × basis × basis or basis × basis shaped."""

  if onebody_operator.ndim == 2:
    return jnp.einsum("σij,ij->", jnp.stack(rdms), onebody_operator)

  return jnp.sum(jnp.stack(rdms) * onebody_operator)


def uhf_hartree(rdms, cholesky_operators):
  v = jnp.einsum("σij,nσij->n", jnp.stack(rdms), cholesky_operators)

  return -0.5 * jnp.sum(v**2)


def uhf_fock(rdms, cholesky_operators):
  G = np.stack(rdms)
  vG = jnp.einsum("nσik,σjk->nσij", cholesky_operators, G)
  GvG = jnp.einsum("σji,nσjk->nσik", G, vG)
  trGvGv = jnp.einsum("nσij,nσji->", GvG, cholesky_operators)

  return 0.5 * trGvGv


def uhf_cholesky(rdms, cholesky_operators):
  r"""
  Calculate the UHF energy of a two-body term expressed by a sum of squares

  .. math::

    H = - \frac 12 \sum_{n=1}^M v_n^2

  of `cholesky_operators`, which are specified as a M × spin × basis × basis array.
  """

  product_term = -0.5 * uhf_onebody(
    rdms, jnp.einsum("nσij,nσjk->σik", cholesky_operators, cholesky_operators)
  )

  return uhf_hartree(rdms, cholesky_operators) + uhf_fock(rdms, cholesky_operators) + product_term
