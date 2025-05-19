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
output_fname = "autohf_output.pkl"
hf_settings = {
  "verbose": False,
  "steps": steps,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "trials": trials,
  "gpu": False,
  "nelec": (Ne, Ne),
  "noncollinear": False,
  "output": output_fname,
}

#### Setup calculation
N = Nx * Ny
T = makeT(Nx, Ny)

settings = {"L": (Nx, Ny)}  # change to shape?
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=T, U=U)

autohf.lattice_hf(
  H,
  settings=hf_settings,
)

# the following is an example of what you can do with the output file API
print("####### Reading output file ######")
import pickle

with open(output_fname, "rb") as f:
  output_data = pickle.load(f)

print(f"Best E={output_data['E']}")
print(f"Ansatz was {output_data['args'].ansatz}")
orbitals = output_data["orbitals"]
print("Orbital overlap matrix:")
Ov = np.swapaxes(orbitals, -1, -2) @ np.conj(orbitals)
print(np.round(Ov, 4))
