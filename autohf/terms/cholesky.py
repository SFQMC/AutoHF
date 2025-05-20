"""
Library of Wick contractions for UHF and GHF wave functions for general second quantized
Hamiltonians with interactions expressed by a sum of cholesky operators.

Per convention, if rdms have only a single spin component, they are considered GHF.
"""

import jax.numpy as jnp


def hf_onebody(rdms, onebody_operator):
  """Calculate the UHF expectation value of a onebody operator. The operator can be either spin × basis × basis or basis × basis shaped."""

  return jnp.sum(jnp.stack(rdms) * onebody_operator)


def hf_hartree(rdms, cholesky_operators):
  v = jnp.einsum("σij,nσij->n", jnp.stack(rdms), cholesky_operators)

  return -0.5 * jnp.sum(v**2)


def hf_fock(rdms, cholesky_operators):
  G = jnp.stack(rdms)
  vG = jnp.einsum("nσik,σjk->nσij", cholesky_operators, G)
  GvG = jnp.einsum("σji,nσjk->nσik", G, vG)
  trGvGv = jnp.einsum("nσij,nσji->", GvG, cholesky_operators)

  return 0.5 * trGvGv


def hf_cholesky(rdms, cholesky_operators):
  r"""
  Calculate the Hartree-Fock energy of a two-body term expressed by a sum of squares

  .. math::

    H = - \frac 12 \sum_{n=1}^M v_n^2

  of `cholesky_operators`, which are specified as a M × spin × basis × basis array.
  """

  product_term = -0.5 * hf_onebody(
    rdms, jnp.einsum("nσij,nσjk->σik", cholesky_operators, cholesky_operators)
  )

  return hf_hartree(rdms, cholesky_operators) + hf_fock(rdms, cholesky_operators) + product_term
