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


### set this
fname = "input.toml"

# cheating to set Hamiltonian with input
input_settings = autohf.io.read_input_params(fname)
lattice_settings = input_settings["lattice"]
Nx, Ny = lattice_settings["L1"], lattice_settings["L2"]
U = input_settings["hamiltonian"]["U"]
N = Nx * Ny
T = makeT(Nx, Ny, lattice_settings["boundary1"] == "PBC", lattice_settings["boundary2"] == "PBC")

H = autohf.AutoHFHamiltonian(T=(T, T), U=1.5)

autohf.solve_hf(H, inputfile=fname)
