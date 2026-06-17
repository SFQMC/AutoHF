import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def sd_to_rdm(sds):
  r"""Build the one-body reduced density matrix from collinear orbitals.

  Computes the spin-resolved one-body reduced density matrix
  :math:`\langle\psi| c_i^\dagger c_j |\psi\rangle / \langle\psi|\psi\rangle`
  for a collinear (UHF) Slater determinant.

  Parameters
  ----------
  sds : array-like, shape ``(2, N, Ne)``
      The occupied orbitals of the up and down spin sectors. ``N`` is the
      number of orbitals and ``Ne`` the number of electrons in each sector
      (the two sectors may have different occupations).

  Returns
  -------
  rdm_up : jax.numpy.ndarray, shape ``(N, N)``
      One-body reduced density matrix of the up-spin sector.
  rdm_dn : jax.numpy.ndarray, shape ``(N, N)``
      One-body reduced density matrix of the down-spin sector.
  """
  # (2,N,Ne)
  # because we could have different number of particles, let's not batch
  phi_up, phi_dn = sds
  rdm_up = phi_up.conj() @ jnp.linalg.solve(phi_up.T @ phi_up.conj(), phi_up.T)
  rdm_dn = phi_dn.conj() @ jnp.linalg.solve(phi_dn.T @ phi_dn.conj(), phi_dn.T)
  return rdm_up, rdm_dn


def create_polarized_func(phi0, f):
  r"""Wrap a function so it receives spin-polarized, occupation-sliced orbitals.

  Builds a closure that slices an input state down to the occupied orbitals
  implied by the collinear reference ``phi0`` before passing it to ``f``. This
  is used to apply a function defined on the occupied subspace to a larger
  state array.

  Parameters
  ----------
  phi0 : sequence of array-like
      Collinear reference orbitals ``(phi_up, phi_dn)``. The shapes
      ``(N, Nup)`` and ``(_, Ndn)`` determine the slice applied to each input
      state.
  f : callable
      Function to call with the sliced ``(up, dn)`` orbital tuple.

  Returns
  -------
  rdm_func : callable
      Function that takes a state ``(state_up, state_dn)``, slices it to the
      occupied blocks ``state[0][:N, :Nup]`` and ``state[1][:N, :Ndn]``, and
      returns ``f`` evaluated on the result.
  """
  N, Nup = phi0[0].shape
  _, Ndn = phi0[1].shape

  def rdm_func(state):
    return f((state[0][0:N, 0:Nup], state[1][0:N, 0:Ndn]))

  return rdm_func


@jax.jit
def sd_to_rdm_noncollinear(sds):
  r"""Build the one-body reduced density matrix from noncollinear orbitals.

  Computes the one-body reduced density matrix
  :math:`\langle\psi| c_i^\dagger c_j |\psi\rangle / \langle\psi|\psi\rangle`
  for a noncollinear (GHF) Slater determinant, where the two spin sectors are
  combined into a single spinor block.

  Parameters
  ----------
  sds : array-like, shape ``(1, N, Ne_total)``
      The occupied spinor orbitals. ``N`` is the number of orbitals and
      ``Ne_total`` the total number of electrons.

  Returns
  -------
  rdm : jax.numpy.ndarray, shape ``(N, N)``
      The one-body reduced density matrix.

  Notes
  -----
  Unlike :func:`sd_to_rdm`, this returns a single array rather than a tuple,
  since the collinear spin decomposition does not apply.
  """
  # (1,N,Ne)
  phi = sds[0]
  rdm = phi.conj() @ jnp.linalg.solve(phi.T @ phi.conj(), phi.T)
  return rdm  # tuple for consistency with the collinear case!


def uhf_to_ghf(x):
  r"""Convert collinear (UHF) orbitals to block-diagonal noncollinear (GHF) form.

  Embeds the up- and down-spin blocks of a UHF input
  :math:`[o_\uparrow, o_\downarrow]` into a single block-diagonal GHF matrix:

  .. math::

      \begin{bmatrix} o_\uparrow & 0 \\ 0 & o_\downarrow \end{bmatrix}

  Parameters
  ----------
  x : numpy.ndarray or jax.numpy.ndarray
      A length-2 sequence ``[o_up, o_dn]`` of the up- and down-spin blocks.
      A NumPy input produces a NumPy result; otherwise a JAX array is returned.

  Returns
  -------
  numpy.ndarray or jax.numpy.ndarray
      The block-diagonal GHF matrix, of the same array type as the input.
  """
  if isinstance(x, np.ndarray):
    return np.block([[x[0], np.zeros_like(x[1])], [np.zeros_like(x[0]), x[1]]])
  return jnp.block([[x[0], jnp.zeros_like(x[1])], [jnp.zeros_like(x[0]), x[1]]])
