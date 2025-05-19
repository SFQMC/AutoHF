import pytest
import numpy as np
import autohf.terms.uhf as uhf
import autohf.tools


def rng_slater_det(dim, num_elec):
  return [
    np.linalg.qr(np.random.randn(dim, n) + 1.0j * np.random.randn(dim, n))[0] for n in num_elec
  ]


class TestTermsUHF:
  def test_onebody(self):
    dim = 10

    sd = rng_slater_det(dim, (2, 3))

    A = np.random.randn(dim, dim)

    expected = sum(sum(s[:, j].T.conj() @ A @ s[:, j] for j in range(s.shape[1])) for s in sd)

    assert np.allclose(expected, uhf.uhf_onebody(autohf.tools.sdToRDM(sd), A))

    A = np.random.randn(2, dim, dim)

    expected = sum(
      sum(s[:, j].T.conj() @ A[σ, :, :] @ s[:, j] for j in range(s.shape[1]))
      for σ, s in enumerate(sd)
    )

    assert np.allclose(expected, uhf.uhf_onebody(autohf.tools.sdToRDM(sd), A))

  def test_cholesky(self):
    dim = 10

    # single particle
    A = np.random.randn(10, 2, dim, dim)
    sd = rng_slater_det(dim, (1, 0))
    rdm = autohf.tools.sdToRDM(sd)

    expected = -0.5 * uhf.uhf_onebody(rdm, np.einsum("nsij,nsjk->sik", A, A))

    assert np.allclose(expected, uhf.uhf_cholesky(rdm, A))

    # hubbard spin-channel

    U = np.zeros([dim, 2, dim, dim])
    for σ in range(2):
      for i in range(dim):
        U[i, σ, i, i] = 1 - 2 * σ

    sd = rng_slater_det(dim, (3, 2))
    rdm = autohf.tools.sdToRDM(sd)
    shift = -5 / 2

    assert np.allclose(uhf.uhf_U(rdm, np.arange(dim)), uhf.uhf_cholesky(rdm, U) - shift)
    # hubbard charge-channel

    U = np.zeros([dim, 2, dim, dim], dtype=np.complex128)
    for σ in range(2):
      for i in range(dim):
        U[i, σ, i, i] = 1.0j

    assert np.allclose(uhf.uhf_U(rdm, np.arange(dim)), uhf.uhf_cholesky(rdm, U) + shift)
