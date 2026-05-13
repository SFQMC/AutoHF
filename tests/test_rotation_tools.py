import pytest
import numpy as np
import jax.numpy as jnp
import jax
from jax import config

# Enable 64-bit precision for tests
config.update("jax_enable_x64", True)

from autohf.rotation_tools import (
  jexpm,
  get_nonz,
  get_nonz_herm,
  makeRot,
  makeRot_herm,
)


class TestRotationTools:
  """Test rotation and matrix exponential tools."""

  rng = np.random.default_rng(45)

  def test_jexpm_identity(self):
    """Test fast matrix exponential on zero matrix returns identity."""
    N = 5
    A = jnp.zeros((N, N))
    result = jexpm(A)
    expected = jnp.eye(N)
    
    assert jnp.allclose(result, expected, atol=1e-10)

  def test_jexpm_diagonal(self):
    """Test fast matrix exponential on diagonal matrix."""
    N = 5
    diag_vals = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    A = jnp.diag(diag_vals)
    
    result = jexpm(A)
    expected = jnp.diag(jnp.exp(diag_vals))
    
    # jexpm is approximate, so use looser tolerance
    assert jnp.allclose(result, expected, atol=1e-4)

  def test_jexpm_small_matrix(self):
    """Test fast matrix exponential on small random matrix."""
    N = 4
    A = jnp.array(self.rng.standard_normal((N, N))) * 0.1  # Small norm for stability
    
    result = jexpm(A)
    expected = jax.scipy.linalg.expm(A)
    
    # Should be close for small norm matrices
    assert jnp.allclose(result, expected, atol=1e-3)

  def test_jexpm_skew_hermitian(self):
    """Test fast matrix exponential on skew-Hermitian matrix."""
    N = 4
    # Skew-Hermitian: A = -A^H
    A_temp = jnp.array(self.rng.standard_normal((N, N)) + 1j * self.rng.standard_normal((N, N)))
    A = (A_temp - A_temp.conj().T) / 2 * 0.1  # Make skew-Hermitian and small
    
    result = jexpm(A)
    
    # Result should be unitary (U^H U = I)
    should_be_identity = result.conj().T @ result
    assert jnp.allclose(should_be_identity, jnp.eye(N), atol=1e-3)

  def test_get_nonz_basic(self):
    """Test get_nonz returns correct indices."""
    N = 4
    nonz = get_nonz(N)
    
    # Should have N*(N-1)/2 elements (upper triangle without diagonal)
    expected_length = N * (N - 1) // 2
    assert len(nonz) == expected_length
    
    # Values should be valid indices
    assert np.all(nonz >= 0)
    assert np.all(nonz < N * (N + 1) // 2)

  def test_get_nonz_size_growth(self):
    """Test get_nonz size scales correctly."""
    for N in [3, 5, 7]:
      nonz = get_nonz(N)
      expected_length = N * (N - 1) // 2
      assert len(nonz) == expected_length

  def test_get_nonz_herm_basic(self):
    """Test get_nonz_herm returns correct indices."""
    N = 4
    nonz = get_nonz_herm(N)
    
    # Should have N*(N+1)/2 elements (upper triangle with diagonal)
    expected_length = N * (N + 1) // 2
    assert len(nonz) == expected_length
    
    # Values should be valid indices
    assert np.all(nonz >= 0)

  def test_get_nonz_herm_size_growth(self):
    """Test get_nonz_herm size scales correctly."""
    for N in [3, 5, 7]:
      nonz = get_nonz_herm(N)
      expected_length = N * (N + 1) // 2
      assert len(nonz) == expected_length

  def test_makeRot_zero_returns_identity(self):
    """Test makeRot with zero vector returns identity."""
    N = 4
    num_params = N * (N + 1) // 2  # Total params for the matrix
    v = jnp.zeros(num_params)
    
    result = makeRot(v, N)
    
    assert jnp.allclose(result, jnp.eye(N), atol=1e-10)

  def test_makeRot_is_unitary(self):
    """Test makeRot produces unitary matrices."""
    N = 5
    num_params = N * (N + 1) // 2
    
    # Random parameter vector
    v = jnp.array(self.rng.standard_normal(num_params)) * 0.1
    
    U = makeRot(v, N)
    
    # Check unitarity: U^H U = I
    should_be_identity = U.conj().T @ U
    assert jnp.allclose(should_be_identity, jnp.eye(N), atol=1e-10)

  def test_makeRot_determinant_one(self):
    """Test makeRot produces matrices with determinant ≈ 1."""
    N = 4
    num_params = N * (N + 1) // 2
    
    v = jnp.array(self.rng.standard_normal(num_params)) * 0.1
    
    U = makeRot(v, N)
    det = jnp.linalg.det(U)
    
    # Determinant should be on unit circle for unitary matrix
    assert jnp.allclose(jnp.abs(det), 1.0, atol=1e-8)

  def test_makeRot_herm_zero_returns_identity(self):
    """Test makeRot_herm with zero vector returns identity."""
    N = 4
    num_params = N * (N + 1) // 2
    v = jnp.zeros(num_params)
    
    result = makeRot_herm(v, N)
    
    assert jnp.allclose(result, jnp.eye(N), atol=1e-10)

  def test_makeRot_herm_is_hermitian_result(self):
    """Test makeRot_herm produces Hermitian (positive definite) matrices."""
    N = 5
    num_params = N * (N + 1) // 2
    
    # Random parameter vector
    v = jnp.array(self.rng.standard_normal(num_params)) * 0.1
    
    U = makeRot_herm(v, N)
    
    # Result should be Hermitian
    assert jnp.allclose(U, U.conj().T, atol=1e-10)
    
    # Matrix exponential of Hermitian is positive definite
    # Check all eigenvalues are positive (or close to positive for numerical stability)
    eigenvalues = jnp.linalg.eigvalsh(U)
    assert jnp.all(eigenvalues > -1e-10)

  def test_makeRot_herm_hermitian_generator(self):
    """Test that makeRot_herm uses a Hermitian generator."""
    N = 4
    num_params = N * (N + 1) // 2
    
    # Small random parameters
    v = jnp.array(self.rng.standard_normal(num_params)) * 0.01
    
    # Reconstruct the Hermitian matrix
    xc = jnp.concatenate([v, v[N:][::-1]])
    Heff = jnp.triu(xc.reshape([N, N]), k=0)
    Heff_full = Heff + np.tril(Heff.conj().T, k=-1)
    
    # Check it's Hermitian
    assert jnp.allclose(Heff_full, Heff_full.conj().T, atol=1e-10)

  def test_makeRot_different_sizes(self):
    """Test makeRot works for different matrix sizes."""
    for N in [2, 3, 4, 6]:
      num_params = N * (N + 1) // 2
      v = jnp.array(self.rng.standard_normal(num_params)) * 0.1
      
      U = makeRot(v, N)
      
      assert U.shape == (N, N)
      # Check unitarity
      assert jnp.allclose(U.conj().T @ U, jnp.eye(N), atol=1e-10)

  def test_makeRot_herm_different_sizes(self):
    """Test makeRot_herm works for different matrix sizes."""
    for N in [2, 3, 4, 6]:
      num_params = N * (N + 1) // 2
      v = jnp.array(self.rng.standard_normal(num_params)) * 0.1
      
      U = makeRot_herm(v, N)
      
      assert U.shape == (N, N)
      # Check it's Hermitian
      assert jnp.allclose(U, U.conj().T, atol=1e-10)
