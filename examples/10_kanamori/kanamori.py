r"""
Hubbard-Kanamori model on a 1-D chain with 2-bands per site.

If site indices are :math:`i`, :math:`j` and band indices is :math:`m`, :math:`n`,
The Hamiltonian is given by:

.. math::
  H = -t \sum_m \sum_{\langle i, j \rangle, \sigma} (c_{i\sigma}^\dagger c_{j\sigma} + h.c.)
      + \sum_m U_m \sum_i n_{i\uparrow} n_{i\downarrow}
      + \sum_{m < n} U^1_{mn} \sum_i (n_{im\uparrow} n_{in\downarrow} + n_{im\downarrow} n_{in\uparrow}) )
      + \sum_{m < n} U^2_{mn} \sum_i (n_{im\uparrow} n_{in\uparrow} + n_{im\downarrow} n_{in\downarrow})
      + \sum_{m < n} J_{mn} \sum_i (c_{im\uparrow}^\dagger c_{in\uparrow} c_{in\downarrow}^\dagger c_{im\downarrow} + h.c.) 

In this example, we have 2 bands, and make the following choices for the interaction parameters. Note that
only the upper-triangular part of the interaction matrices are specified, and the lower-triangular part is inferred from symmetry.

U_m = [ 2.0, 4.0]
U1_mn = [[ 0.0, 1.5
          0.0, 0.0]]
U2_mn = [[ 0.0, 1.0
          0.0, 0.0]]
J_mn = [[ 0.0, 0.5
          0.0, 0.0]]
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

import numpy as np

import autohf

def make_T_multiband(Nx, Ny, Nbands, t, xperiodic=True, yperiodic=True):
  Nsites = Nx * Ny

  # Build single-band hopping on (Nsites, Nsites)
  T_band = np.zeros((Nsites, Nsites))
  for ix in range(Nx):
    for iy in range(Ny):
      s = ix * Ny + iy
      if ix + 1 < Nx or xperiodic:
        s_nb = ((ix + 1) % Nx) * Ny + iy
        T_band[s, s_nb] = -t
        T_band[s_nb, s] = -t
      if iy + 1 < Ny or yperiodic:
        s_nb = ix * Ny + (iy + 1) % Ny
        T_band[s, s_nb] = -t
        T_band[s_nb, s] = -t

  # Uniform across bands: shape (Nbands, Nsites, Nsites)
  T_intraband = np.stack([T_band] * Nbands, axis=0)

  N = Nbands * Nsites
  T = np.zeros((N, N))
  for b in range(Nbands):
    T[b::Nbands, b::Nbands] = T_intraband[b]
  return T


def make_interband_matrix(Nsites, Nbands, val):
  """NxN upper-triangular matrix with `val` at every same-site inter-band pair."""
  M_band = np.zeros((Nbands, Nbands))
  M_band[np.triu_indices(Nbands, k=1)] = val
  M = np.kron(np.eye(Nsites), M_band)
  return M


#### Define Parameters
Nx, Ny = 6, 1
Nbands = 2
Nsites = Nx * Ny
N = Nsites * Nbands
xperiodic = True
yperiodic = False

t = 1.0
U_val = 2.0
U1_val = 1.5
U2_val = 1.0
J_val = 0.5

Ne = 6
steps = 500
batch_size = 10


T = make_T_multiband(Nx, Ny, Nbands, t, xperiodic=xperiodic, yperiodic=yperiodic)

U1_mat = make_interband_matrix(Nsites, Nbands, U1_val)
U2_mat = make_interband_matrix(Nsites, Nbands, U2_val)
J_mat = make_interband_matrix(Nsites, Nbands, J_val)

H = autohf.AutoHFHamiltonian(
  T=(T, T),         
  U=U_val,          # scalar → uniform on-orbital Hubbard
  U1=U1_mat,        # inter-orbital, opposite-spin density-density
  U2=U2_mat,        # inter-orbital, same-spin density-density
  J=J_mat,          # Hund's coupling
  Nx=Nx,
  Ny=Ny,
  Nbands=Nbands,
)

hf_settings = {
  "verbose": False,
  "steps": steps,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "batch_size": batch_size,
  "gpu": False,
  "nelec": (Ne, Ne),
  "noncollinear": False,
}

autohf.lattice_hf(H, settings=hf_settings)
