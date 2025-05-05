import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
# import matplotlib.pyplot as plt


import autohf


def makeT(Nx, Ny, xperiodic=True, yperiodic=True):
  N = Nx * Ny
  T = np.zeros([N, N])
  # fill hopping matrix
  for i in range(Nx):
    for j in range(Ny):
      a = i * Ny + j
      if i + 1 < Nx or yperiodic:
        b = ((i + 1) % Nx) * Ny + j

        T[a, b] = -1
        T[b, a] = -1
      if i + 1 < Ny or xperiodic:
        b = i * Ny + (j + 1) % Ny
        T[a, b] = -1
        T[b, a] = -1
  return T


Nx, Ny = 4, 4
N = Nx * Ny
T = makeT(Nx, Ny)

settings = {"L": (Nx, Ny)}  # change to shape?
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=T, U=1.5)

hf_settings = {
  "verbose": False,
  "numSteps": 100,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "numTrials": 4,
  "gpu": False,
  "nelec": (7, 7),  # should this be moved?
  # "dumpTrials"
}
autohf.solver.lattice_hf(
  H,
  settings=hf_settings,
)
