"""
Spin expectation values of a Slater determinant from its one-body reduced density matrix.

``rdm`` is always the ``(2M, 2M)`` spin-orbital density matrix, up-spin block first. A collinear
``(rdm_up, rdm_dn)`` pair enters as ``scipy.linalg.block_diag(rdm_up, rdm_dn)``.
"""

import numpy as np

# spin-1/2 operators S_x, S_y, S_z
_SPIN = 0.5 * np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def spin_z(rdm):
  r"""Expectation value :math:`\langle S_z \rangle`."""
  rdm = np.asarray(rdm)
  M = rdm.shape[0] // 2
  return 0.5 * (np.trace(rdm[:M, :M]) - np.trace(rdm[M:, M:])).real


def spin_squared(rdm):
  r"""
  Expectation value :math:`\langle S^2 \rangle`. By Wick's theorem,

  .. math:: \langle S^2 \rangle = \frac{3}{4} N + \sum_a \left( \mathrm{tr}(S_a \rho)^2
            - \mathrm{tr}(S_a \rho S_a \rho) \right)
  """
  rdm = np.asarray(rdm)
  M = rdm.shape[0] // 2
  S2 = 0.75 * np.trace(rdm)
  for Sa in _SPIN:
    # Sa @ rdm with Sa only acting on the spin block rows
    X = np.einsum("ab,bin->ain", Sa, rdm.reshape(2, M, 2 * M)).reshape(2 * M, 2 * M)
    S2 += np.trace(X) ** 2 - np.einsum("ij,ji->", X, X)
  return S2.real
