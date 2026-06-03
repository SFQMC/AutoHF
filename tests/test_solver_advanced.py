"""
Tests for advanced solver functionality not covered in basic tests.

Covers:
- Riemannian optimization
- Alternative optimizers (adam, sgd, basin hopping)
- Complex dtype handling
- Spin measurement output
- Edge cases and error paths
"""

import os
import pytest
import warnings
import numpy as np
from autohf import solve_hf

# Ensure CPU-only execution for testing
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class TestRiemannianOptimization:
  """Test Riemannian optimization paths in solver."""

  def test_riemannian_sd(self, hubbard_4x4):
    """Test Riemannian optimization with SD ansatz."""
    settings = {
      "nelec": (7, 7),
      "steps": 20,
      "opt_method": "lbfgs",
      "ansatz": "SD",
      "batch_size": 2,
      "riemannian": True,
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    assert "state" in results
    assert "E" in results
    assert isinstance(results["E"], (float, np.floating))

  def test_riemannian_sdc(self, hubbard_4x4):
    """Test Riemannian optimization with SDC ansatz."""
    settings = {
      "nelec": (7, 7),
      "steps": 20,
      "opt_method": "lbfgs",
      "ansatz": "SDC",
      "batch_size": 2,
      "riemannian": True,
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    assert "state" in results
    assert "E" in results

  def test_riemannian_invalid_ansatz_raises(self, hubbard_4x4):
    """Test that Riemannian with DIAG ansatz raises error."""
    settings = {
      "nelec": (7, 7),
      "steps": 10,
      "opt_method": "lbfgs",
      "ansatz": "DIAG",  # Not compatible with Riemannian
      "batch_size": 1,
      "riemannian": True,
    }

    with pytest.raises(NotImplementedError):
      solve_hf(hamiltonian=hubbard_4x4, settings=settings)


class TestAlternativeOptimizers:
  """Test non-LBFGS optimization methods."""

  def test_grad_optimizer(self, hubbard_4x4):
    """Test gradient descent optimizer."""
    settings = {
      "nelec": (7, 7),
      "steps": 50,
      "opt_method": "grad",
      "ansatz": "SD_ROT",
      "batch_size": 2,
      "eta": 0.1,
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    assert "state" in results
    assert "E" in results
    assert "Es" in results
    assert len(results["Es"]) == 2  # batch_size

  # this one is likely slow; consider marking as slow test if needed
  def test_basin_hopping_optimizer(self, hubbard_4x4):
    """Test basin hopping optimizer (scipy)."""
    settings = {
      "nelec": (7, 7),
      "steps": 5,  # niter for basin hopping
      "opt_method": "basin",
      "ansatz": "SD_ROT",
      "batch_size": 1,  # Basin hopping requires batch_size=1
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    assert "state" in results
    assert "E" in results

  def test_basin_hopping_warns_on_batch_size(self, hubbard_4x4):
    """Test that basin hopping warns if batch_size > 1."""
    settings = {
      "nelec": (7, 7),
      "steps": 2,
      "opt_method": "basin",
      "ansatz": "SD_ROT",
      "batch_size": 3,  # Should trigger warning
    }

    # Basin hopping requires float64
    from jax import config

    config.update("jax_enable_x64", True)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

      # Check for warning about batch_size
      warning_messages = [str(warning.message) for warning in w]
      assert any("batch_size" in msg for msg in warning_messages)

  def test_invalid_optimizer_raises(self, hubbard_4x4):
    """Test that invalid optimizer raises NotImplementedError."""
    settings = {
      "nelec": (7, 7),
      "steps": 10,
      "opt_method": "fake_optimizer",
      "ansatz": "SD_ROT",
      "batch_size": 1,
    }

    with pytest.raises(NotImplementedError, match="not implemented"):
      solve_hf(hamiltonian=hubbard_4x4, settings=settings)


class TestComplexDtype:
  """Test complex dtype handling and approx_expm."""

  def test_complex_flag_accepted(self, hubbard_4x4):
    """Test that complex flag is accepted (dtype determined by Hamiltonian)."""
    settings = {
      "nelec": (7, 7),
      "steps": 20,
      "opt_method": "lbfgs",
      "ansatz": "SD_ROT",
      "batch_size": 2,
      "complex": True,
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    assert "state" in results
    assert "E" in results
    # Note: Real Hamiltonian produces real states regardless of complex flag

  def test_approx_expm_flag_accepted(self, hubbard_4x4):
    """Test that approx_expm flag is accepted."""
    settings = {
      "nelec": (7, 7),
      "steps": 10,
      "opt_method": "lbfgs",
      "ansatz": "SD_ROT",
      "batch_size": 1,
      "approx_expm": True,
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    assert "state" in results
    assert "E" in results


class TestSpinMeasurements:
  """Test spin measurement output functionality."""

  def test_spin_measurements_collinear(self, hubbard_4x4, capfd):
    """Test spin_z and spin_squared output for collinear."""
    settings = {
      "nelec": (8, 6),  # Polarized for non-zero Sz
      "steps": 20,
      "opt_method": "lbfgs",
      "ansatz": "SD_ROT",
      "batch_size": 1,
      "verbose": True,
      "measure_spin": True,  # Required to print spin measurements
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    # measure_spin=True prints spin measurements
    captured = capfd.readouterr()
    assert "<Sz>" in captured.out
    assert "<S^2>" in captured.out

  def test_spin_measurements_noncollinear(self, hubbard_4x4, capfd):
    """Test spin_z and spin_squared output for noncollinear."""
    settings = {
      "nelec": (7, 7),
      "steps": 20,
      "opt_method": "lbfgs",
      "ansatz": "SD_ROT",
      "batch_size": 1,
      "noncollinear": True,
      "verbose": True,
      "measure_spin": True,  # Required to print spin measurements
    }
    results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

    # measure_spin=True prints spin measurements
    captured = capfd.readouterr()
    assert "<Sz>" in captured.out
    assert "<S^2>" in captured.out


class TestEdgeCases:
  """Test edge cases and error paths in solver."""

  def test_energy_above_reference_warns(self, hubbard_4x4):
    """Test warning when final energy exceeds reference."""
    # Start with very bad initial guess to potentially exceed reference
    settings = {
      "nelec": (7, 7),
      "steps": 5,  # Few steps to potentially not converge well
      "opt_method": "lbfgs",
      "ansatz": "DIAG",
      "batch_size": 2,
      "state0_scale": 10.0,  # Large perturbation
    }

    # This might trigger the warning if optimization fails
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      results, funcs = solve_hf(hamiltonian=hubbard_4x4, settings=settings)

      # Check if warning about energy was raised (might not always trigger)
      warning_messages = [str(warning.message) for warning in w]
      # This is conditional - warning only if final E > reference E
      if any("final energy" in msg for msg in warning_messages):
        assert any("reference energy" in msg for msg in warning_messages)

  def test_multiple_hopping_matrices(self):
    """Test that Hamiltonian can handle multiple hopping matrices."""
    from autohf import AutoHFHamiltonian
    import numpy as np

    N = 16
    T_up = np.random.randn(N, N)
    T_up = 0.5 * (T_up + T_up.T)  # Make Hermitian
    T_dn = np.random.randn(N, N)
    T_dn = 0.5 * (T_dn + T_dn.T)  # Make Hermitian

    ham = AutoHFHamiltonian(
      T=[T_up, T_dn],  # Two hopping matrices
      U=2.5,
      N=N,
      Nx=4,
      Ny=4,
    )

    settings = {
      "nelec": (7, 7),
      "steps": 10,
      "opt_method": "lbfgs",
      "ansatz": "SD_ROT",
      "batch_size": 1,
      "noncollinear": True,
    }

    results, funcs = solve_hf(hamiltonian=ham, settings=settings)
    assert "state" in results
    assert "E" in results
