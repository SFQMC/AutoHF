import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax
# import matplotlib.pyplot as plt


import autohf


hops = (np.array([ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,
                        2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
                        5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  8,  8,  8,
                        8,  8,  8,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 11, 11,
                       11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14,
                       14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,
                       17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19,
                       19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
                       22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25,
                       25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28,
                       28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31,
                       31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33,
                       34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35]),
         np.array([ 1,  5,  6, 11, 30, 31,  0,  2,  6,  7, 31, 32,  1,  3,  7,  8, 32,
                        33,  2,  4,  8,  9, 33, 34,  3,  5,  9, 10, 34, 35,  0,  4, 10, 11,
                        30, 35,  0,  1,  7, 11, 12, 17,  1,  2,  6,  8, 12, 13,  2,  3,  7,
                         9, 13, 14,  3,  4,  8, 10, 14, 15,  4,  5,  9, 11, 15, 16,  0,  5,
                         6, 10, 16, 17,  6,  7, 13, 17, 18, 23,  7,  8, 12, 14, 18, 19,  8,
                         9, 13, 15, 19, 20,  9, 10, 14, 16, 20, 21, 10, 11, 15, 17, 21, 22,
                         6, 11, 12, 16, 22, 23, 12, 13, 19, 23, 24, 29, 13, 14, 18, 20, 24,
                        25, 14, 15, 19, 21, 25, 26, 15, 16, 20, 22, 26, 27, 16, 17, 21, 23,
                        27, 28, 12, 17, 18, 22, 28, 29, 18, 19, 25, 29, 30, 35, 19, 20, 24,
                        26, 30, 31, 20, 21, 25, 27, 31, 32, 21, 22, 26, 28, 32, 33, 22, 23,
                        27, 29, 33, 34, 18, 23, 24, 28, 34, 35,  0,  5, 24, 25, 31, 35,  0,
                         1, 25, 26, 30, 32,  1,  2, 26, 27, 31, 33,  2,  3, 27, 28, 32, 34,
                         3,  4, 28, 29, 33, 35,  4,  5, 24, 29, 30, 34])
        )  # fmt: skip


#### Define Parameters
Nx, Ny = 6, 6
U = 4
Ne = 18
numSteps = 2000
numTrials = 4

hf_settings = {
  "verbose": False,
  "numSteps": numSteps,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "numTrials": numTrials,
  "gpu": False,
  "nelec": (Ne, Ne),
  "noncollinear": True,
  "state0_scale": 0.01,
  "seed": 1234,
}


# Build a lagrange multiplier for total Sz
# We'll use this so that our energy is E+lambda*|\sum_i Sz_i|^2+100
# Importante note: as an example, let's shift the energy
# so we understand when its the special loss vs real energy
lam = 10.0


def Esz2(state, rdms):
  r_uu, r_dd = rdms[0], rdms[1]
  return lam * jnp.abs(jnp.sum(r_uu.diagonal() - r_dd.diagonal())) ** 2 + 100


##
## We'll do the following steps:
## 1. run a calculation with no steps to get an energy function
##    because our energy function will really be a loss,
##    polluted with the lagrange multiplier
##
## 2. For illustration, we'll run without the multiplier,
##    to see how the the GHF state can rotate arbitrarily
##    You may naturally do this step in order to realize
##    the state needs constraining more explicitly
##
## 3. We run AutoHf with the custom energy term
##
## 4. Compare 2 and 3
##

#### Setup calculation
N = Nx * Ny
assert Nx == 6 and Ny == 6  # harded coded here
T = np.zeros((N, N))
T[hops] = -1

settings = {"L": (Nx, Ny)}
H = autohf.hamiltonian.AutoHFHamiltonian(settings=settings)
H.add_terms(T=T, U=U)

####### Step 1 ######
# First get energy function
temp_settings = hf_settings | {"numSteps": -1}
data, data_functions = autohf.solver.lattice_hf(
  H,
  settings=temp_settings,
)
energy_func = data_functions["energy_func"]

####### Step 2 ######
# run with default settings, no extra term
data, data_functions = autohf.solver.lattice_hf(
  H,
  settings=hf_settings,
)

final_rdm = data_functions["makeRDMs"](data["state"])
r_uu, r_dd = final_rdm[:N, :N], final_rdm[N:, N:]

####### Step 3 ######
# run with addition term
# for illustration, energy is shifted by 100
data_sz, data_functions_sz = autohf.solver.lattice_hf(
  H,
  settings=hf_settings,
  custom_energy_term=Esz2,
)

final_rdm_sz = data_functions_sz["makeRDMs"](data_sz["state"])
r_uu_sz, r_dd_sz = final_rdm_sz[:N, :N], final_rdm_sz[N:, N:]

####### Step 4 ######
print("True energy:        ", energy_func(data["state"]))
print("True energy w/ term:", energy_func(data_sz["state"]))
print("Final sz:        ", np.sum(r_uu.diagonal() - r_dd.diagonal()))
print("Final sz w/ term:", np.sum(r_uu_sz.diagonal() - r_dd_sz.diagonal()))
