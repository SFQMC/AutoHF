"""
Extended tests for AutoHFHamiltonian covering edge cases.
"""

import pytest
import numpy as np
from autohf import AutoHFHamiltonian


class TestHamiltonianEdgeCases:
  """Test edge cases in Hamiltonian construction."""

  def test_hamiltonian_with_U1_only(self, rng):
    """Test Hamiltonian with only U1 interaction term."""
    N = 16
    U1 = rng.standard_normal(size=(N, N))
    U1 = 0.5 * (U1 + U1.T)  # Make symmetric

    ham = AutoHFHamiltonian(U1=U1, N=N, Nx=4, Ny=4)

    assert "U1" in ham.keys()
    assert "U" not in ham.keys()

  def test_hamiltonian_with_U2_only(self, rng):
    """Test Hamiltonian with only U2 interaction term."""
    N = 16
    U2 = rng.standard_normal(size=(N, N))
    U2 = 0.5 * (U2 + U2.T)  # Make symmetric

    ham = AutoHFHamiltonian(U2=U2, N=N, Nx=4, Ny=4)

    assert "U2" in ham.keys()
    assert "U" not in ham.keys()

  def test_hamiltonian_with_J_exchange(self, rng):
    """Test Hamiltonian with J exchange interaction."""
    N = 16
    J = rng.standard_normal(size=(N, N))
    J = 0.5 * (J + J.T)  # Make symmetric

    ham = AutoHFHamiltonian(J=J, N=N, Nx=4, Ny=4)

    assert "J" in ham.keys()

  def test_hamiltonian_with_heisenberg(self, rng):
    """Test Hamiltonian with Heisenberg exchange."""
    N = 16
    Jheis = rng.standard_normal(size=(N, N))
    Jheis = 0.5 * (Jheis + Jheis.T)  # Make symmetric

    ham = AutoHFHamiltonian(Jheisenberg=Jheis, N=N, Nx=4, Ny=4)

    assert "J_heisenberg" in ham.keys()

  def test_hamiltonian_T_wrong_shape_raises(self, rng):
    """Test that T matrix with wrong shape raises RuntimeError."""
    N = 16
    # Create T with wrong shape (should be (nspin, N, N))
    T_wrong = rng.standard_normal(size=(N, N))  # Wrong: 2D instead of 3D

    with pytest.raises(RuntimeError, match="needs to be"):
      AutoHFHamiltonian(T=T_wrong, N=N, Nx=4, Ny=4)

  def test_hamiltonian_U_scalar_without_N_raises(self):
    """Test that scalar U without N raises RuntimeError."""
    with pytest.raises(RuntimeError, match="N must be set"):
      AutoHFHamiltonian(U=2.5, Nx=4, Ny=4)

  def test_hamiltonian_invalid_source_raises(self):
    """Test that invalid source type raises ValueError."""
    class InvalidSource:
      pass

    with pytest.raises(ValueError, match="Can't initialize"):
      AutoHFHamiltonian(source=InvalidSource())

  def test_hamiltonian_unknown_term_raises(self, rng):
    """Test that accessing unknown term raises ValueError."""
    # Need to pass T to set N before scalar U
    T = rng.standard_normal(size=(2, 16, 16))
    ham = AutoHFHamiltonian(T=T, U=2.5, N=16, Nx=4, Ny=4)

    with pytest.raises(ValueError, match="Unknown term"):
      ham.set("INVALID_TERM", 1.0)

  def test_hamiltonian_from_dict(self, rng):
    """Test creating Hamiltonian from existing Hamiltonian via source parameter."""
    # Create initial Hamiltonian (need T to set N before scalar U)
    T = rng.standard_normal(size=(2, 16, 16))
    ham1 = AutoHFHamiltonian(T=T, U=2.5, N=16, Nx=4, Ny=4)

    # Create another from the first using source= parameter (tests copy constructor)
    ham2 = AutoHFHamiltonian(source=ham1)

    assert ham2.N == ham1.N
    assert ham2.Nx == ham1.Nx
    assert ham2.Ny == ham1.Ny
    assert "U" in ham2.keys()
    assert "T" in ham2.keys()

  def test_hamiltonian_T_and_U_and_J(self, rng):
    """Test Hamiltonian with hopping, Hubbard, and exchange."""
    N = 16
    T_2d = rng.standard_normal(size=(N, N))
    T_2d = 0.5 * (T_2d + T_2d.T)
    T = np.stack([T_2d, T_2d])  # Make 3D (nspin=2, N, N)
    J = rng.standard_normal(size=(N, N))
    J = 0.5 * (J + J.T)

    ham = AutoHFHamiltonian(T=T, U=2.5, J=J, N=N, Nx=4, Ny=4)

    assert "T" in ham.keys()
    assert "U" in ham.keys()
    assert "J" in ham.keys()

  def test_hamiltonian_all_interaction_types(self, rng):
    """Test Hamiltonian with all possible interaction terms."""
    N = 16
    T_2d = rng.standard_normal(size=(N, N))
    T_2d = 0.5 * (T_2d + T_2d.T)
    T = np.stack([T_2d, T_2d])  # Make 3D (nspin=2, N, N)
    U1 = rng.standard_normal(size=(N, N))
    U1 = 0.5 * (U1 + U1.T)
    U2 = rng.standard_normal(size=(N, N))
    U2 = 0.5 * (U2 + U2.T)
    J = rng.standard_normal(size=(N, N))
    J = 0.5 * (J + J.T)
    Jheis = rng.standard_normal(size=(N, N))
    Jheis = 0.5 * (Jheis + Jheis.T)

    ham = AutoHFHamiltonian(
      T=T, U=2.5, U1=U1, U2=U2, J=J, Jheisenberg=Jheis,
      N=N, Nx=4, Ny=4
    )

    assert "T" in ham.keys()
    assert "U" in ham.keys()
    assert "U1" in ham.keys()
    assert "U2" in ham.keys()
    assert "J" in ham.keys()
    assert "J_heisenberg" in ham.keys()
