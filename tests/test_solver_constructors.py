import pytest
import numpy as np
import jax.numpy as jnp
import warnings

from autohf.solver import constructPhi0, construct_random_Phi
from autohf.options import SlaterType


class TestConstructPhi0:
  """Test constructPhi0 initial state generation."""

  def test_construct_phi0_none_input(self):
    """Test constructPhi0 returns None when Tlist is None."""
    result = constructPhi0(None, N=10, nelec=(3, 3))
    assert result is None

  def test_construct_phi0_collinear_single_T(self):
    """Test constructPhi0 with single T matrix (collinear)."""
    N = 10
    nelec = (3, 4)
    
    # Create simple hopping matrix
    T = np.random.randn(N, N)
    T = (T + T.T) / 2  # Make Hermitian
    
    result = constructPhi0([T], N, nelec, spin_symm=SlaterType.COLLINEAR)
    
    # Should return tuple of (orb_up, orb_dn)
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    orb_up, orb_dn = result
    assert orb_up.shape == (N, nelec[0])
    assert orb_dn.shape == (N, nelec[1])
    
    # Orbitals should be eigenvectors of T
    eigenvals, eigenvecs = np.linalg.eigh(T)
    assert np.allclose(orb_up, eigenvecs[:, :nelec[0]])
    assert np.allclose(orb_dn, eigenvecs[:, :nelec[1]])  # Same T used for both

  def test_construct_phi0_collinear_two_T(self):
    """Test constructPhi0 with separate T matrices for up/down."""
    N = 10
    nelec = (3, 4)
    
    # Create separate hopping matrices
    T_up = np.random.randn(N, N)
    T_up = (T_up + T_up.T) / 2
    
    T_dn = np.random.randn(N, N)
    T_dn = (T_dn + T_dn.T) / 2
    
    result = constructPhi0([T_up, T_dn], N, nelec, spin_symm=SlaterType.COLLINEAR)
    
    orb_up, orb_dn = result
    assert orb_up.shape == (N, nelec[0])
    assert orb_dn.shape == (N, nelec[1])
    
    # Should use different eigenvectors
    _, eigenvecs_up = np.linalg.eigh(T_up)
    _, eigenvecs_dn = np.linalg.eigh(T_dn)
    assert np.allclose(orb_up, eigenvecs_up[:, :nelec[0]])
    assert np.allclose(orb_dn, eigenvecs_dn[:, :nelec[1]])

  def test_construct_phi0_average_spins(self):
    """Test constructPhi0 with averageSpins=True."""
    N = 10
    nelec = (3, 4)
    
    T_up = np.random.randn(N, N)
    T_up = (T_up + T_up.T) / 2
    
    T_dn = np.random.randn(N, N)
    T_dn = (T_dn + T_dn.T) / 2
    
    result = constructPhi0(
      [T_up, T_dn], N, nelec, 
      spin_symm=SlaterType.COLLINEAR, 
      averageSpins=True
    )
    
    orb_up, orb_dn = result
    
    # Both should use averaged T
    T_avg = (T_up + T_dn) / 2
    _, eigenvecs_avg = np.linalg.eigh(T_avg)
    
    assert np.allclose(orb_up, eigenvecs_avg[:, :nelec[0]])
    assert np.allclose(orb_dn, eigenvecs_avg[:, :nelec[1]])

  def test_construct_phi0_noncollinear_from_collinear_T(self):
    """Test constructPhi0 with collinear T but noncollinear walkers."""
    N = 10
    nelec = (3, 4)
    
    T = np.random.randn(N, N)
    T = (T + T.T) / 2
    
    result = constructPhi0([T], N, nelec, spin_symm=SlaterType.NONCOLLINEAR)
    
    # Should return a 1-tuple containing an array of shape (2N, sum(nelec))
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (2*N, sum(nelec))
    
    # Should be block diagonal Hamiltonian
    T_block = np.block([[T, np.zeros((N, N))], [np.zeros((N, N)), T]])
    _, eigenvecs = np.linalg.eigh(T_block)
    
    assert np.allclose(result[0], eigenvecs[:, :sum(nelec)])

  def test_construct_phi0_noncollinear_from_noncollinear_T(self):
    """Test constructPhi0 with noncollinear T matrix."""
    N = 10
    nelec = (3, 4)
    
    # Create 2N x 2N noncollinear Hamiltonian
    T_nc = np.random.randn(2*N, 2*N)
    T_nc = (T_nc + T_nc.T) / 2
    
    result = constructPhi0([T_nc], N, nelec, spin_symm=SlaterType.NONCOLLINEAR)
    
    # Should return a 1-tuple containing an array of shape (2N, sum(nelec))
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (2*N, sum(nelec))
    
    _, eigenvecs = np.linalg.eigh(T_nc)
    assert np.allclose(result[0], eigenvecs[:, :sum(nelec)])

  def test_construct_phi0_noncollinear_T_collinear_walker_raises_error(self):
    """Test constructPhi0 raises error with NC Hamiltonian but collinear walkers."""
    N = 10
    nelec = (3, 4)
    
    # Create 2N x 2N noncollinear Hamiltonian
    T_nc = np.random.randn(2*N, 2*N)
    T_nc = (T_nc + T_nc.T) / 2
    
    with pytest.raises(RuntimeError, match="noncollinear"):
      constructPhi0([T_nc], N, nelec, spin_symm=SlaterType.COLLINEAR)

  def test_construct_phi0_noncollinear_multiple_T_warns(self):
    """Test constructPhi0 warns when multiple T provided for NC Hamiltonian."""
    N = 10
    nelec = (3, 4)
    
    T_nc1 = np.random.randn(2*N, 2*N)
    T_nc1 = (T_nc1 + T_nc1.T) / 2
    
    T_nc2 = np.random.randn(2*N, 2*N)
    T_nc2 = (T_nc2 + T_nc2.T) / 2
    
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      result = constructPhi0([T_nc1, T_nc2], N, nelec, spin_symm=SlaterType.NONCOLLINEAR)
      
      # Should warn about only using first T
      assert len(w) == 1
      assert "only taking the first" in str(w[0].message)
    
    # Should use only first T
    _, eigenvecs = np.linalg.eigh(T_nc1)
    assert np.allclose(result[0], eigenvecs[:, :sum(nelec)])

  def test_construct_phi0_multiple_calls_deterministic(self):
    """Test constructPhi0 produces same result with same input."""
    N = 10
    nelec = (3, 4)
    
    T = np.random.randn(N, N)
    T = (T + T.T) / 2
    
    result1 = constructPhi0([T], N, nelec, spin_symm=SlaterType.COLLINEAR)
    result2 = constructPhi0([T], N, nelec, spin_symm=SlaterType.COLLINEAR)
    
    # Should be deterministic
    orb_up1, orb_dn1 = result1
    orb_up2, orb_dn2 = result2
    
    assert np.allclose(orb_up1, orb_up2)
    assert np.allclose(orb_dn1, orb_dn2)


class TestConstructRandomPhi:
  """Test construct_random_Phi function."""

  def test_construct_random_phi_collinear(self):
    """Test construct_random_Phi creates randomized Slater determinant."""
    rng = np.random.default_rng(42)
    N = 10
    nelec = (3, 4)
    
    result = construct_random_Phi(
      rng, N, nelec, 
      spin_symm=SlaterType.COLLINEAR, 
      delta=0.01
    )
    
    # Should return tuple for collinear
    assert isinstance(result, tuple)
    assert len(result) == 2

  def test_construct_random_phi_noncollinear(self):
    """Test construct_random_Phi for noncollinear case."""
    rng = np.random.default_rng(42)
    N = 10
    nelec = (3, 4)
    
    result = construct_random_Phi(
      rng, N, nelec,
      spin_symm=SlaterType.NONCOLLINEAR,
      delta=0.01
    )
    
    # Should return a 1-tuple containing an array of shape (2*N, sum(nelec))
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (2*N, sum(nelec))

  def test_construct_random_phi_with_initial_phi(self):
    """Test construct_random_Phi with provided initial Phi."""
    rng = np.random.default_rng(42)
    N = 10
    nelec = (3, 4)
    
    # Provide custom initial Phi
    Phi_init = np.random.randn(N, sum(nelec))
    
    result = construct_random_Phi(
      rng, N, nelec,
      spin_symm=SlaterType.COLLINEAR,
      delta=0.01,
      Phi=Phi_init
    )
    
    # Should be different from input due to randomization
    orb_up, orb_dn = result
    # Can't easily check exact relationship, but should be valid
    assert orb_up.shape == (N, nelec[0])
    assert orb_dn.shape == (N, nelec[1])

  def test_construct_random_phi_small_delta(self):
    """Test construct_random_Phi with small delta stays close to identity."""
    rng = np.random.default_rng(42)
    N = 10
    nelec = (3, 4)
    
    # Very small delta should give result close to identity
    result = construct_random_Phi(
      rng, N, nelec,
      spin_symm=SlaterType.COLLINEAR,
      delta=0.001
    )
    
    orb_up, orb_dn = result
    
    # Should be close to identity matrix columns
    identity_cols_up = np.eye(N, sum(nelec))[:, :nelec[0]]
    identity_cols_dn = np.eye(N, sum(nelec))[:, nelec[0]:sum(nelec)]
    
    # With small delta, should be close (but not exactly equal)
    assert np.allclose(np.abs(orb_up), np.abs(identity_cols_up), atol=0.1)

  def test_construct_random_phi_different_seeds_different_results(self):
    """Test construct_random_Phi produces different results with different seeds."""
    N = 10
    nelec = (3, 4)
    
    rng1 = np.random.default_rng(42)
    result1 = construct_random_Phi(rng1, N, nelec, delta=0.1)
    
    rng2 = np.random.default_rng(123)
    result2 = construct_random_Phi(rng2, N, nelec, delta=0.1)
    
    # Results should be different
    orb1_up, orb1_dn = result1
    orb2_up, orb2_dn = result2
    
    assert not np.allclose(orb1_up, orb2_up)
    assert not np.allclose(orb1_dn, orb2_dn)
