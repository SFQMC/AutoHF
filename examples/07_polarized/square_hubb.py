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
U = 1.5
Ne = 7
steps = 100
trials = 4

hf_settings = {
  "verbose": False,
  "steps": steps,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "trials": trials,
  "gpu": False,
  "nelec": (6, 2),
  "noncollinear": False,
}

#### Setup calculation
N = Nx * Ny
T = makeT(Nx, Ny)

settings = {"L": (Nx, Ny)}  # change to shape?
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=T, U=U)

data, data_F = autohf.lattice_hf(
  H,
  settings=hf_settings,
)

data, data_F = autohf.lattice_hf(
  H, settings=hf_settings | {"ansatz": "SD", "riemannian": True}, restart_from=(data, data_F)
)

print([x.shape for x in data["orbitals"]])
rdms = data_F["makeRDMs"](data["state"])
print("<Sz> =", np.round((rdms[0].diagonal() - rdms[1].diagonal()).sum() / 2, 4))
print("<n> =", np.round((rdms[0].diagonal() + rdms[1].diagonal()).sum(), 4))
