import pytest
import numpy as np

afqmctools = pytest.importorskip("afqmctools")
from afqmctools.hamiltonian.model.director import HamiltonianDirector
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder

from autohf.converters import AutoHFHamiltonian


class TestSFQMCHamiltonians:
  @pytest.mark.dev
  @pytest.mark.parametrize(
    "t,U,U1,U2,J",
    [(None, 2, 1.5, 1.0, 0.5), (None, 2, -1.5, 1.0, 0.5), ([1, 0.5], 2, -1.5, 1.0, None)],
  )
  def test_correct_hamiltonian(self, t, U, U1, U2, J):
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

    hamiltonian_director = HamiltonianDirector(source=settings)
    hamiltonian = hamiltonian_director.build()
    lattice = hamiltonian_director.builder.lattice

    # TODO: update to convert to AutoHFHamiltonian first, then pass along.
    hamil = AutoHFHamiltonian(hamiltonian, L=lattice.L)

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
