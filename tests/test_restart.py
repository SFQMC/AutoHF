import pytest
from autohf.solver import solve_hf
import numpy as np


class TestSolverRestart:
  @pytest.mark.parametrize("ansatz", ["SD_ROT", "SD", "DIAG", "SDC", "SD_ROTC"])
  @pytest.mark.parametrize("nelec", [(3, 3), (6, 2)])
  def test_restart_collinear(self, ansatz, nelec, hubbard_4x4):
    hf_settings = dict(
      ansatz=ansatz, steps=-1, nelec=nelec, batch_size=3, seed=12345, noncollinear=False
    )

    results = solve_hf(hubbard_4x4, hf_settings)
    results_restart = solve_hf(hubbard_4x4, hf_settings | {"batch_size": 2}, start_from=results)

    assert np.allclose(results[0]["E"], results_restart[0]["E"])

  @pytest.mark.parametrize("ansatz", ["SD_ROT", "SD", "DIAG", "SDC", "SD_ROTC"])
  @pytest.mark.parametrize("nelec", [(3, 3), (6, 2)])
  def test_restart_noncollinear(self, ansatz, nelec, hubbard_4x4):
    hf_settings = dict(
      ansatz=ansatz, steps=-1, nelec=nelec, batch_size=3, seed=12345, noncollinear=True
    )

    results = solve_hf(hubbard_4x4, hf_settings)
    results_restart = solve_hf(
      hubbard_4x4, hf_settings | {"batch_size": 2, "seed": 42}, start_from=results
    )

    assert np.allclose(results[0]["E"], results_restart[0]["E"])

  @pytest.mark.parametrize("ansatz", ["SD_ROT", "DIAG", "SD_ROTC", "SD"])
  @pytest.mark.parametrize("final_ansatz", ["SD", "SDC"])
  @pytest.mark.parametrize("nelec", [(3, 3), (6, 2)])
  @pytest.mark.parametrize("nc", [False, True])
  def test_restart_convert(self, ansatz, final_ansatz, nelec, nc, hubbard_4x4):
    hf_settings = dict(
      ansatz=ansatz,
      steps=-1,
      nelec=nelec,
      batch_size=3,
      seed=12345,
      noncollinear=nc,
      dump_batch=True,
    )

    results = solve_hf(hubbard_4x4, hf_settings)
    results_restart = solve_hf(
      hubbard_4x4,
      hf_settings | {"ansatz": final_ansatz, "batch_size": 2, "seed": 42},
      start_from=results,
    )

    assert np.allclose(results[0]["E"], results_restart[0]["E"])
