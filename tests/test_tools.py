"""
Tests for autohf.tools utility functions.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from autohf.tools import uhf_to_ghf, sd_to_rdm, sd_to_rdm_noncollinear, create_polarized_func


class TestUhfToGhf:
  """Test uhf_to_ghf function with different input types."""

  def test_uhf_to_ghf_with_numpy_arrays(self, rng):
    """Test uhf_to_ghf with numpy arrays."""
    N = 4
    x_up = rng.standard_normal(size=(N, N))
    x_dn = rng.standard_normal(size=(N, N))
    x = np.array([x_up, x_dn])

    result = uhf_to_ghf(x)

    # Should be (2N, 2N) with block diagonal structure
    assert result.shape == (2*N, 2*N)
    # Check upper-left block
    np.testing.assert_allclose(result[:N, :N], x_up)
    # Check lower-right block
    np.testing.assert_allclose(result[N:, N:], x_dn)
    # Check off-diagonal blocks are zero
    np.testing.assert_allclose(result[:N, N:], np.zeros((N, N)))
    np.testing.assert_allclose(result[N:, :N], np.zeros((N, N)))

  def test_uhf_to_ghf_with_jax_arrays(self, rng):
    """Test uhf_to_ghf with JAX arrays."""
    N = 4
    x_up = jnp.array(rng.standard_normal(size=(N, N)))
    x_dn = jnp.array(rng.standard_normal(size=(N, N)))
    x = jnp.array([x_up, x_dn])

    result = uhf_to_ghf(x)

    # Should be (2N, 2N) with block diagonal structure
    assert result.shape == (2*N, 2*N)
    # Check it's a JAX array
    assert isinstance(result, jnp.ndarray)

  def test_uhf_to_ghf_different_sizes(self, rng):
    """Test uhf_to_ghf with different sized blocks."""
    N = 3
    M = 5
    x_up = rng.standard_normal(size=(N, M))
    x_dn = rng.standard_normal(size=(N, M))
    x = np.array([x_up, x_dn])

    result = uhf_to_ghf(x)

    # Should handle different sized blocks
    assert result.shape == (2*N, 2*M)


class TestSdToRdm:
  """Test sd_to_rdm functions."""

  def test_sd_to_rdm_collinear(self, rng):
    """Test sd_to_rdm with collinear orbitals."""
    N = 8
    Nup = 3
    Ndn = 2

    # Create random orbitals
    phi_up = jnp.array(rng.standard_normal(size=(N, Nup)))
    phi_dn = jnp.array(rng.standard_normal(size=(N, Ndn)))
    sds = (phi_up, phi_dn)

    rdm_up, rdm_dn = sd_to_rdm(sds)

    # Check shapes
    assert rdm_up.shape == (N, N)
    assert rdm_dn.shape == (N, N)
    # RDMs should be hermitian
    np.testing.assert_allclose(rdm_up, rdm_up.conj().T, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rdm_dn, rdm_dn.conj().T, rtol=1e-10, atol=1e-12)

  def test_sd_to_rdm_noncollinear(self, rng):
    """Test sd_to_rdm_noncollinear with noncollinear orbitals."""
    N = 8
    Ne = 4

    # Create random orbitals
    phi = jnp.array(rng.standard_normal(size=(N, Ne)))
    sds = (phi,)

    rdm = sd_to_rdm_noncollinear(sds)

    # Check shape
    assert rdm.shape == (N, N)
    # RDM should be hermitian
    np.testing.assert_allclose(rdm, rdm.conj().T, rtol=1e-10, atol=1e-12)


class TestCreatePolarizedFunc:
  """Test create_polarized_func wrapper."""

  def test_create_polarized_func_with_sd_to_rdm(self, rng):
    """Test create_polarized_func wraps sd_to_rdm correctly."""
    N = 8
    Nup = 3
    Ndn = 2

    # Create initial phi0
    phi0_up = rng.standard_normal(size=(N, Nup))
    phi0_dn = rng.standard_normal(size=(N, Ndn))
    phi0 = (phi0_up, phi0_dn)

    # Create polarized function
    rdm_func = create_polarized_func(phi0, sd_to_rdm)

    # Create padded state (larger than phi0) - correct format is simple tuple
    N_padded = N
    Nup_padded = Nup + 2
    Ndn_padded = Ndn + 2
    state_up = jnp.array(rng.standard_normal(size=(N_padded, Nup_padded)))
    state_dn = jnp.array(rng.standard_normal(size=(N_padded, Ndn_padded)))
    state = (state_up, state_dn)

    # Should extract correct slices
    rdm_up, rdm_dn = rdm_func(state)

    assert rdm_up.shape == (N, N)
    assert rdm_dn.shape == (N, N)
