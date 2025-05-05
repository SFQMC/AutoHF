import pytest
from autohf.solver import lattice_hf


class TestAutoHFSolver:
  """
  Tests for the behavior of the lattice_hf solver.
  This does not include tests which check the correctness
  of the Hartree-Fock energy for specific systems.
  """

  def test_starts(self, hubbard_4x4):
    """
    Tests that the lattice_hf solver can be started with
    a variety of initial conditions.
    """

    hf_settings = dict(
      ansatz="SD_ROT", numSteps=10, nelec=(8, 8), numTrials=3, seed=12345, noncollinear=False
    )

    results = lattice_hf(hamiltonian=hubbard_4x4, settings=hf_settings)

    assert results is not None

  def test_restart(self, hubbard_4x4):
    """
    Tests that the lattice_hf solver
    can be restarted from a previous calculation.
    This allows settings (or even the Hamiltonian) to be changed
    after some initial optimization.

    Parameters
    ----------
    hubbard_4x4 : AutoHFHamiltonian (from pytest fixture)
        A simple 4x4 Hubbard model with periodic boundary conditions.
    """

    hf_settings = dict(
      ansatz="SD_ROT", numSteps=10, nelec=(8, 8), numTrials=3, seed=12345, noncollinear=False
    )

    results = lattice_hf(hubbard_4x4, settings=hf_settings)

    hf_settings["ansatz"] = "SD"
    hf_settings["riemannian"] = True

    results = lattice_hf(hubbard_4x4, settings=hf_settings, start_from=results)

    # TODO: need to think about how to test that this
    #         did what we wanted (this only checks that it runs
    #         after being given a results dictionary from a previous run)
    assert results is not None

  def test_expected_errors(self):
    """
    Tests that the lattice_hf solver raises the expected errors.

    TODO: use pytest mark parameterize to test many cases here.
    """
    with pytest.raises(ValueError):
      lattice_hf(None, settings={})
