import pytest
import numpy as np
import jax.numpy as jnp
import autohf.terms.uhf as uhf
import autohf.terms.ghf as ghf
import autohf.terms.cholesky as cholesky
import autohf.tools


def rng_slater_det(rng, dim, num_elec):
  return [
    np.linalg.qr(rng.standard_normal((dim, n)) + 1.0j * rng.standard_normal((dim, n)))[0]
    for n in num_elec
  ]


class TestCholeskyTerms:
  rng = np.random.default_rng(42)

  def test_onebody(self):
    dim = 10
    rng = self.rng

    sd = rng_slater_det(rng, dim, (2, 3))

    A = rng.standard_normal((dim, dim))

    expected = sum(sum(s[:, j].T.conj() @ A @ s[:, j] for j in range(s.shape[1])) for s in sd)

    assert np.allclose(expected, cholesky.hf_onebody(autohf.tools.sd_to_rdm(sd), A))

    A = rng.standard_normal((2, dim, dim))

    expected = sum(
      sum(s[:, j].T.conj() @ A[σ, :, :] @ s[:, j] for j in range(s.shape[1]))
      for σ, s in enumerate(sd)
    )

    assert np.allclose(expected, cholesky.hf_onebody(autohf.tools.sd_to_rdm(sd), A))

  def test_cholesky(self):
    dim = 10
    rng = self.rng

    # single particle
    A = rng.standard_normal((10, 2, dim, dim))
    sd = rng_slater_det(rng, dim, (1, 0))
    rdm = autohf.tools.sd_to_rdm(sd)

    expected = -0.5 * cholesky.hf_onebody(rdm, np.einsum("nsij,nsjk->sik", A, A))

    assert np.allclose(expected, cholesky.hf_cholesky(rdm, A))

    # hubbard spin-channel

    U = np.zeros([dim, 2, dim, dim])
    for σ in range(2):
      for i in range(dim):
        U[i, σ, i, i] = 1 - 2 * σ

    nelec = (3, 2)
    sd = rng_slater_det(rng, dim, nelec)
    rdm = autohf.tools.sd_to_rdm(sd)
    shift = -sum(nelec) / 2

    assert np.allclose(uhf.uhf_U(rdm, np.arange(dim)), cholesky.hf_cholesky(rdm, U) - shift)
    # hubbard charge-channel

    U = np.zeros([dim, 2, dim, dim], dtype=np.complex128)
    for σ in range(2):
      for i in range(dim):
        U[i, σ, i, i] = 1.0j

    assert np.allclose(uhf.uhf_U(rdm, np.arange(dim)), cholesky.hf_cholesky(rdm, U) + shift)


class TestUHFTerms:
  """Test UHF interaction term functions."""
  
  rng = np.random.default_rng(43)

  def test_rhf_hop(self):
    """Test RHF hopping term."""
    dim = 8
    nelec = (3, 3)
    
    # Create random Slater determinant and RDM
    sd = rng_slater_det(self.rng, dim, nelec)
    rdm = autohf.tools.sd_to_rdm(sd)[0]  # Just use spin-up RDM
    
    # Create random hops (i, j pairs)
    num_hops = 5
    hops = (self.rng.integers(0, dim, num_hops), self.rng.integers(0, dim, num_hops))
    
    # Calculate hopping energy
    result = uhf.rhf_hop(rdm, hops)
    
    # Verify it's a scalar
    assert isinstance(result, (jnp.ndarray, np.ndarray, float, complex))
    
    # Manually calculate expected result
    expected = jnp.sum(rdm[hops[0], hops[1]])
    assert np.allclose(result, expected)

  def test_uhf_U1(self):
    """Test UHF U1 (mixed spin) term."""
    dim = 8
    nelec = (3, 2)
    
    sd = rng_slater_det(self.rng, dim, nelec)
    rdm = autohf.tools.sd_to_rdm(sd)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = uhf.uhf_U1(rdm, ii, jj, ij, ji)
    
    # Verify it's a scalar
    assert isinstance(result, (jnp.ndarray, np.ndarray, float, complex))
    
    # Manually calculate
    rho_up, rho_dn = rdm[0], rdm[1]
    expected = (rho_up[ii[0], ii[1]] @ rho_dn[jj[0], jj[1]] + 
                rho_up[jj[0], jj[1]] @ rho_dn[ii[0], ii[1]])
    assert np.allclose(result, expected)

  def test_uhf_U2(self):
    """Test UHF U2 (same spin) term."""
    dim = 8
    nelec = (3, 2)
    
    sd = rng_slater_det(self.rng, dim, nelec)
    rdm = autohf.tools.sd_to_rdm(sd)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = uhf.uhf_U2(rdm, ii, jj, ij, ji)
    
    # Manually calculate
    rho_up, rho_dn = rdm[0], rdm[1]
    expected = (rho_up[ii[0], ii[1]] @ rho_up[jj[0], jj[1]] + 
                rho_dn[ii[0], ii[1]] @ rho_dn[jj[0], jj[1]])
    expected -= (rho_up[ij[0], ij[1]] @ rho_up[ji[0], ji[1]] + 
                 rho_dn[ij[0], ij[1]] @ rho_dn[ji[0], ji[1]])
    assert np.allclose(result, expected)

  def test_uhf_J(self):
    """Test UHF J term."""
    dim = 8
    nelec = (3, 2)
    
    sd = rng_slater_det(self.rng, dim, nelec)
    rdm = autohf.tools.sd_to_rdm(sd)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = uhf.uhf_J(rdm, ii, jj, ij, ji)
    
    # Manually calculate
    rho_up, rho_down = rdm[0], rdm[1]
    expected = (rho_up[ij[0], ij[1]] @ rho_down[ji[0], ji[1]] +
                rho_up[ij[0], ij[1]] @ rho_down[ij[0], ij[1]] +
                rho_up[ji[0], ji[1]] @ rho_down[ij[0], ij[1]] +
                rho_up[ji[0], ji[1]] @ rho_down[ji[0], ji[1]])
    assert np.allclose(result, expected)

  def test_uhf_Jheis(self):
    """Test UHF Heisenberg J term."""
    dim = 8
    nelec = (3, 2)
    
    sd = rng_slater_det(self.rng, dim, nelec)
    rdm = autohf.tools.sd_to_rdm(sd)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = uhf.uhf_Jheis(rdm, ii, jj, ij, ji)
    
    # Manually calculate
    rho_up, rho_down = rdm[0], rdm[1]
    expected = (rho_up[ii[0], ii[1]] @ rho_down[jj[0], jj[1]] +
                rho_up[ji[0], ji[1]] @ rho_down[ij[0], ij[1]])
    assert np.allclose(result, expected)


class TestGHFTerms:
  """Test GHF interaction term functions."""
  
  rng = np.random.default_rng(44)

  def _create_ghf_rdms(self, dim, nelec):
    """Create mock GHF RDMs (rdm_uu, rdm_dd, rdm_ud, rdm_du)."""
    # For GHF, we have all four spin components
    rdm_uu = self.rng.standard_normal((dim, dim)) + 1j * self.rng.standard_normal((dim, dim))
    rdm_dd = self.rng.standard_normal((dim, dim)) + 1j * self.rng.standard_normal((dim, dim))
    rdm_ud = self.rng.standard_normal((dim, dim)) + 1j * self.rng.standard_normal((dim, dim))
    rdm_du = self.rng.standard_normal((dim, dim)) + 1j * self.rng.standard_normal((dim, dim))
    
    # Make them hermitian-like (density matrices should be)
    rdm_uu = (rdm_uu + rdm_uu.conj().T) / 2
    rdm_dd = (rdm_dd + rdm_dd.conj().T) / 2
    
    return (rdm_uu, rdm_dd, rdm_ud, rdm_du)

  def test_ghf_U(self):
    """Test GHF Hubbard U term."""
    dim = 8
    nelec = (3, 2)
    
    rdms = self._create_ghf_rdms(dim, nelec)
    sites = self.rng.integers(0, dim, 5)
    
    result = ghf.ghf_U(rdms, sites)
    
    # Manually calculate
    rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
    dup, ddn = rdm_uu.diagonal(), rdm_dd.diagonal()
    dud, ddu = rdm_ud.diagonal(), rdm_du.diagonal()
    expected = dup[sites] @ ddn[sites] - dud[sites] @ ddu[sites]
    
    assert np.allclose(result, expected)

  def test_ghf_U1(self):
    """Test GHF U1 (mixed spin) term."""
    dim = 8
    nelec = (3, 2)
    
    rdms = self._create_ghf_rdms(dim, nelec)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = ghf.ghf_U1(rdms, ii, jj, ij, ji)
    
    # Manually calculate
    rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
    expected = (rdm_uu[ii[0], ii[1]] @ rdm_dd[jj[0], jj[1]] + 
                rdm_uu[jj[0], jj[1]] @ rdm_dd[ii[0], ii[1]])
    expected -= (rdm_ud[ij[0], ij[1]] @ rdm_du[ji[0], ji[1]] + 
                 rdm_du[ij[0], ij[1]] @ rdm_ud[ji[0], ji[1]])
    
    assert np.allclose(result, expected)

  def test_ghf_U2(self):
    """Test GHF U2 (same spin) term."""
    dim = 8
    nelec = (3, 2)
    
    rdms = self._create_ghf_rdms(dim, nelec)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = ghf.ghf_U2(rdms, ii, jj, ij, ji)
    
    # Manually calculate
    rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
    expected = (rdm_uu[ii[0], ii[1]] @ rdm_uu[jj[0], jj[1]] + 
                rdm_dd[ii[0], ii[1]] @ rdm_dd[jj[0], jj[1]])
    expected -= (rdm_uu[ij[0], ij[1]] @ rdm_uu[ji[0], ji[1]] + 
                 rdm_dd[ij[0], ij[1]] @ rdm_dd[ji[0], ji[1]])
    
    assert np.allclose(result, expected)

  def test_ghf_J(self):
    """Test GHF J term."""
    dim = 8
    nelec = (3, 2)
    
    rdms = self._create_ghf_rdms(dim, nelec)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = ghf.ghf_J(rdms, ii, jj, ij, ji)
    
    # Manually calculate
    rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
    expected = (rdm_uu[ij[0], ij[1]] @ rdm_dd[ji[0], ji[1]] +
                rdm_uu[ij[0], ij[1]] @ rdm_dd[ij[0], ij[1]] +
                rdm_uu[ji[0], ji[1]] @ rdm_dd[ij[0], ij[1]] +
                rdm_uu[ji[0], ji[1]] @ rdm_dd[ji[0], ji[1]])
    expected -= (rdm_ud[ii[0], ii[1]] @ rdm_du[jj[0], jj[1]] +
                 rdm_ud[ij[0], ij[1]] @ rdm_du[ij[0], ij[1]] +
                 rdm_ud[jj[0], jj[1]] @ rdm_du[ii[0], ii[1]] +
                 rdm_ud[ji[0], ji[1]] @ rdm_du[ji[0], ji[1]])
    
    assert np.allclose(result, expected)

  def test_ghf_Jheis(self):
    """Test GHF Heisenberg J term."""
    dim = 8
    nelec = (3, 2)
    
    rdms = self._create_ghf_rdms(dim, nelec)
    
    # Create site pairs
    num_pairs = 4
    i_sites = self.rng.integers(0, dim, num_pairs)
    j_sites = self.rng.integers(0, dim, num_pairs)
    
    ii = np.array([i_sites, i_sites])
    jj = np.array([j_sites, j_sites])
    ij = np.array([i_sites, j_sites])
    ji = np.array([j_sites, i_sites])
    
    result = ghf.ghf_Jheis(rdms, ii, jj, ij, ji)
    
    # Manually calculate
    rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
    expected = (rdm_uu[ii[0], ii[1]] @ rdm_dd[jj[0], jj[1]] +
                rdm_uu[ji[0], ji[1]] @ rdm_dd[ij[0], ij[1]])
    expected -= (rdm_ud[ij[0], ij[1]] @ rdm_du[ji[0], ji[1]] +
                 rdm_ud[jj[0], jj[1]] @ rdm_du[ii[0], ii[1]])
    
    assert np.allclose(result, expected)
