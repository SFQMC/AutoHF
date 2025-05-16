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
      ansatz="SD_ROT",
      numSteps=10,
      nelec=(8, 8),
      numTrials=3,
      seed=12345,
      noncollinear=False,
      dumpTrials=True,
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

  @pytest.mark.parametrize("ansatz", ["SD_ROT", "SD", "DIAG", "SDC", "SD_ROTC"])
  @pytest.mark.parametrize("nelec", [(3, 3), (6, 2)])
  def test_shapes_collinear(self, ansatz, nelec, hubbard_4x4):
    N = 16
    hf_settings = dict(
      ansatz=ansatz, numSteps=-1, nelec=nelec, numTrials=3, seed=12345, noncollinear=False
    )

    results = lattice_hf(hubbard_4x4, hf_settings)
    assert results[0]["nelec"] == nelec
    assert results[0]["args"].ansatz == ansatz
    oout = results[0]["orbitals"]
    assert oout[0].shape == (N, nelec[0]) and oout[1].shape == (N, nelec[1])

    rdms = results[1]["makeRDMs"](results[0]["state"])
    assert rdms[0].shape == (N, N) and rdms[1].shape == (N, N)
    o = results[1]["state2Orbitals"](results[0]["state"])
    assert o[0].shape == (N, nelec[0]) and o[1].shape == (N, nelec[1])

  @pytest.mark.parametrize("ansatz", ["SD_ROT", "SD", "DIAG", "SDC", "SD_ROTC"])
  @pytest.mark.parametrize("nelec", [(3, 3), (6, 2)])
  def test_shapes_noncollinear(self, ansatz, nelec, hubbard_4x4):
    N = 16
    hf_settings = dict(
      ansatz=ansatz, numSteps=-1, nelec=nelec, numTrials=3, seed=12345, noncollinear=True
    )

    results = lattice_hf(hubbard_4x4, hf_settings)
    assert results[0]["nelec"] == nelec
    assert results[0]["args"].ansatz == ansatz
    oout = results[0]["orbitals"][0]
    assert oout.shape == (2 * N, sum(nelec))

    rdms = results[1]["makeRDMs"](results[0]["state"])
    assert rdms.shape == (2 * N, 2 * N)
    o = results[1]["state2Orbitals"](results[0]["state"])[0]
    assert o.shape == (2 * N, sum(nelec))
