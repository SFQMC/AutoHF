import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
# import matplotlib.pyplot as plt


import autohf


def makeTPin(Nx, Ny, h_pin, yperiodic=False):
  N = Nx * Ny
  T = np.zeros((N, N))
  for i in range(Nx):
    for j in range(Ny):
      a = i * Ny + j
      if i + 1 < Nx or yperiodic:
        b = ((i + 1) % Nx) * Ny + j
        T[a, b] = -1
        T[b, a] = -1
      b = i * Ny + (j + 1) % Ny
      T[a, b] = -1
      T[b, a] = -1
  Sz = 1 / 2 * h_pin
  pin = np.zeros(N)
  pin[np.arange(N - Ny, N)] = Sz * (-1) ** (np.arange(Ny))
  pin[np.arange(Ny)] = Sz * (-1) ** (1 + np.arange(Ny))
  print(pin)
  print(np.sum(T, axis=0))
  assert np.allclose(T, T.T)
  return T, pin


h_pin = 0.5  # h_pin*Sz_i = h_pin/2*Z
Nx, Ny = 4, 4
N = Nx * Ny
T_, pin = makeTPin(Nx, Ny, h_pin)
T_up, T_dn = T_ + np.diag(pin), T_ - np.diag(pin)

settings = {"L": (Nx, Ny)}
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=(T_up, T_dn), U=1.5)

hf_settings = {
  "verbose": False,
  "steps": 100,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "trials": 4,
  "gpu": False,
  "nelec": (7, 7),
}
autohf.lattice_hf(
  H,
  settings=hf_settings,
)
