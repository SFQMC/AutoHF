import pytest
import numpy as np
import jax.numpy as jnp

from autohf.hamiltonian import AutoHFHamiltonian
from autohf.converters import convertInitialSlaterDet
from autohf.options import SlaterType


def _build_lattice_hamiltonian(package, settings):
  pytest.importorskip(package)
  if package == "afqmctools":
    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder

    return HamiltonianBuilder.from_input(settings).hamiltonian
  from safiretools import LatticeHamiltonian

  return LatticeHamiltonian.from_dict(settings)


class TestSAFIREHamiltonians:
  @pytest.mark.dev
  @pytest.mark.parametrize("package", ["afqmctools", "safiretools"])
  @pytest.mark.parametrize(
    "t,U,U1,U2,J",
    [(None, 2, 1.5, 1.0, 0.5), (None, 2, -1.5, 1.0, 0.5), ([1, 0.5], 2, -1.5, 1.0, None)],
  )
  def test_correct_hamiltonian(self, package, t, U, U1, U2, J):
    """
    Tests that the Hamiltonian is properly converted to
    autoHF's format.

    This test is indirect. It only checks that each of the
    unique values in the input parameters t,U,U1,U2,J appear
    *somewhere* in the appropriate "term".
    """
    settings = {
      "hamiltonian": dict(nbands=2, t=t, U=U, U1=U1, U2=U2, J=J),
      "lattice": dict(L1=3, L2=3, boundary1="PBC", boundary2="PBC"),
    }

    hamiltonian = _build_lattice_hamiltonian(package, settings)
    hamil = AutoHFHamiltonian(hamiltonian)

    # check that all *values* are present
    for name in ["t", "U", "U1", "U2", "J"]:
      input_data = locals().get(name)
      if input_data:
        # by convntion
        if name == "t":
          sign = -1
        else:
          sign = 1
        # check if all unique values are present
        unique_input_values = sign * np.unique(np.array(input_data))
        convertedTerm = hamil.get(name.upper(), [])[0]
        for val in unique_input_values:
          if name != "t":
            assert val in convertedTerm
          else:
            assert np.any([val in term for term in convertedTerm])
      else:
        assert hamil.get(name, None) is None


class TestConvertInitialSlaterDet:
  """Test conversion of initial Slater determinants between formats."""

  def test_collinear_conversion(self, rng):
    """Test conversion of collinear Slater determinant."""
    nmo = 10
    nelec = (3, 4)

    # Create mock initial guess in afqmctools format: [nmo, nelec[0] + nelec[1]]
    initial_guess = rng.standard_normal(size=(nmo, nelec[0] + nelec[1])) \
                    + 1j * rng.standard_normal(size=(nmo, nelec[0] + nelec[1]))

    result = convertInitialSlaterDet(initial_guess, nelec, spin_symm=SlaterType.COLLINEAR)

    # Result should be a tuple of two arrays
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Check shapes
    orb_up, orb_dn = result
    assert orb_up.shape == (nmo, nelec[0])
    assert orb_dn.shape == (nmo, nelec[1])

    # Check values match input
    assert jnp.allclose(orb_up, initial_guess[:, :nelec[0]])
    assert jnp.allclose(orb_dn, initial_guess[:, nelec[0]:])

  def test_noncollinear_conversion(self, rng):
    """Test conversion of noncollinear Slater determinant."""
    nmo = 10
    nelec = (3, 4)
    n_total = nelec[0] + nelec[1]

    # Create mock initial guess
    initial_guess = rng.standard_normal(size=(nmo, n_total)) + 1j * rng.standard_normal(size=(nmo, n_total))

    result = convertInitialSlaterDet(initial_guess, nelec, spin_symm=SlaterType.NONCOLLINEAR)

    # Result should be a 1-tuple containing a 2D array
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (nmo, n_total)

    # Check values match input
    assert jnp.allclose(result[0], initial_guess[:, :n_total])

  def test_invalid_slater_type(self, rng):
    """Test that invalid SlaterType raises ValueError."""
    nmo = 10
    nelec = (3, 4)
    initial_guess = rng.standard_normal(size=(nmo, nelec[0] + nelec[1]))

    # SlaterType.CLOSED is not valid for conversion
    with pytest.raises(ValueError, match="Could not convert"):
      convertInitialSlaterDet(initial_guess, nelec, spin_symm=SlaterType.CLOSED)
