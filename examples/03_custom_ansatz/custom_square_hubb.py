import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

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
Ne = 8
T = makeT(Nx, Ny)

settings = {"L": (Nx, Ny)}  # change to shape?
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=T, U=4)

TGHF_ = autohf.tools.uhfToGhf([T, T])
signs = (-1) ** (np.arange(N) // Nx + np.arange(N) % Nx)


def orbitalFunc(state):
  # take the hopping matrix TGHF, and "set" M on the off-diagonal blocks
  TGHF = TGHF_ + 0.0
  TGHF = TGHF.at[N + np.arange(N), np.arange(N)].set(signs * state)
  TGHF = TGHF.at[np.arange(N), N + np.arange(N)].set(signs * state)
  lams, vecs = jnp.linalg.eigh(TGHF)
  return jnp.stack([vecs[:, : 2 * Ne]])


def rdmFunc(state):
  o = orbitalFunc(state)
  return autohf.solver.sdToRDM_noncollinear(o)


hf_settings = {
  "verbose": False,
  "numSteps": 2000,
  "opt_method": "lbfgs",
  "ansatz": "CUSTOM",
  "numTrials": 16,
  "gpu": False,
  "nelec": (Ne, Ne),
  "noncollinear": True,
}

rng = np.random.default_rng(42)
state0_ref = 0.0 * np.arange(N).reshape(N)
state0 = state0_ref + rng.normal(scale=0.01, size=(hf_settings["numTrials"], N))

autohf.lattice_hf(
  H,
  settings=hf_settings,
  state2orbitals=orbitalFunc,
  state2rdm=rdmFunc,
  state0=state0,
  state0_ref=state0_ref,
  jaxoptargs=dict(tol=1e-14),  # secret sauce
)
