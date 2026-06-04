import pytest
import numpy as np


from autohf.hamiltonian import AutoHFHamiltonian


def pytest_addoption(parser):
  parser.addoption(
    "--seed",
    action="store",
    default="42",
    help="RNG seed for tests. Use an integer for a fixed seed, or 'random' for a random seed.",
  )

@pytest.fixture(scope="session")
def hubbard_4x4():
  """
  A simple 4x4 Hubbard model with periodic boundary conditions.
  """
  t = -1.0
  U = 4.0
  L = 4

  T = t * np.array([[0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.]]
               )  # fmt: skip

  H_U = np.full((L), U)

  # TODO: make this more smooth
  settings = {"L": (L, L)}
  hamiltonian = AutoHFHamiltonian(settings=settings)
  hamiltonian.add_terms(T=(T, T), U=H_U)
  return hamiltonian

@pytest.fixture(scope="session")
def rng(request):
  """Fixture to provide a random number generator with a configurable seed.
  
  Uses session scope to ensure the same RNG is used across all tests in the session, 
  allowing for reproducibility, but without generating the same random samples in every test.
  """
  raw = request.config.getoption("--seed")
  seed = None if raw == "random" else int(raw)
  return np.random.default_rng(seed)
