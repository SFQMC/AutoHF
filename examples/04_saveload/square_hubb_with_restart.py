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


#### Define Parameters
Nx, Ny = 4, 4
U = 2.5
Ne = 7
numSteps = 100
numTrials = 4
output_base = "autohf_output"
hf_settings = {
  "verbose": False,
  "numSteps": numSteps,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "numTrials": numTrials,
  "gpu": False,
  "nelec": (Ne, Ne),
  "noncollinear": False,
  "output": output_base + "_0.pkl",
  "dumpTrials": True,  # needed to convert from SD_ROT to SD
  "seed": 42,
}

#### Setup calculation
N = Nx * Ny
T = makeT(Nx, Ny)

settings = {"L": (Nx, Ny)}  # change to shape?
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=T, U=U)

autohf.solver.lattice_hf(
  H,
  settings=hf_settings,
)


hf_settings = {
  "verbose": False,
  "numSteps": numSteps,
  "opt_method": "lbfgs",
  "ansatz": "SD",
  "riemannian": True,
  "numTrials": numTrials // 2,  # pick the best half
  "gpu": False,
  "nelec": (Ne, Ne),
  "noncollinear": False,
  "output": output_base + "_1.pkl",
  "seed": 1234,
}
autohf.solver.lattice_hf(
  H,
  settings=hf_settings,
  start_from=output_base + "_0.pkl",
  jaxoptargs=dict(tol=1e-14),  # secret sauce
)
