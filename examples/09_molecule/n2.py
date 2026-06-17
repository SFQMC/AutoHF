import importlib
import sys
import time

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

if importlib.util.find_spec("pyscf") is None:
  print("pyscf is required to run this example, please install it")
  sys.exit(0)

import pyscf
import numpy as np
import autohf

# helper functions
def format_fixed_width_strings(strings):
  """Format a list of strings to be fixed width."""
  return ' '.join('{:>17}'.format(s) for s in strings)

def format_fixed_width_floats(floats):
  """Format a list of floats to be fixed width."""
  return ' '.join('{: .10e}'.format(f) for f in floats)

def modified_cholesky_direct(M, tol=1e-5, verbose=False, cmax=20):
  """Modified cholesky decomposition of matrix.

  See, e.g. [Motta17]_

  Parameters
  ----------
  M : :class:`np.ndarray`
      Positive semi-definite, symmetric matrix.
  tol : float
      Accuracy desired. Optional. Default : False.
  verbose : bool
      If true print out convergence progress. Optional. Default : False.
  cmax : int
      Number of cholesky vectors to store N_gamma = cmax M.

  Returns
  -------
  chol_vecs : :class:`np.ndarray`
      Matrix of cholesky vectors.
  """
  # matrix of residuals.
  delta = np.copy(M.diagonal())
  nchol_max = min(int(cmax*M.shape[0]**0.5),M.shape[1])
  # index of largest diagonal element of residual matrix.
  nu = np.argmax(np.abs(delta))
  delta_max = delta[nu]
  if verbose:
    print(f"Performing Cholesky decomposition with tolerance {tol}")
    print ("# max number of cholesky vectors = %d"%nchol_max)
    header = ['iteration', 'max_residual', 'time']
    print(format_fixed_width_strings(header))
    init = [delta_max]
    print('{:17d} '.format(0)+format_fixed_width_floats(init))
  # Store for current approximation to input matrix.
  Mapprox = np.zeros(M.shape[0], dtype=M.dtype)
  chol_vecs = np.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
  nchol = 0
  chol_vecs[0] = np.copy(M[:,nu])/delta_max**0.5
  while abs(delta_max) > tol and nchol<nchol_max-1: # -1 because we already have one vector.
    # Update cholesky vector
    start = time.time()
    Mapprox += chol_vecs[nchol]*chol_vecs[nchol].conj()
    delta = M.diagonal() - Mapprox
    nu = np.argmax(np.abs(delta))
    delta_max = np.abs(delta[nu])
    nchol += 1
    Munu0 = np.dot(chol_vecs[:nchol,nu].conj(), chol_vecs[:nchol,:])
    chol_vecs[nchol] = (M[:,nu] - Munu0) / (delta_max)**0.5
    if verbose:
      step_time = time.time() - start
      out = [delta_max, step_time]
      print('{:17d} '.format(nchol)+format_fixed_width_floats(out))

  return np.array(chol_vecs[:nchol])

def get_ortho_ao_mol(S, LINDEP_CUTOFF=0):
  """Generate canonical orthogonalization transformation matrix.
  Parameters
  ----------
  S : :class:`numpy.ndarray`
      Overlap matrix.
  LINDEP_CUTOFF : float
      Linear dependency cutoff. Basis functions whose eigenvalues lie below
      this value are removed from the basis set. Should be set in accordance
      with value in :func:`pyscf.scf.addons.remove_linear_dep`.
  
  Returns
  -------
  X : :class:`numpy.array`
      Transformation matrix.
  """
  sdiag, Us = np.linalg.eigh(S)
  X = Us[:,sdiag>LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
  return X

## End helper functions


mol = pyscf.M(
  atom="""
    N 0 0 0.5488
    N 0 0 -0.5488
""",
  basis="sto-3g",
)

onebody = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
overlap = mol.intor("int1e_ovlp")
eri = mol.intor("int2e")
nuclear_repulsion = mol.energy_nuc()

X = get_ortho_ao_mol(overlap)

norb = onebody.shape[0]
print(f"{norb=}")

chol_tol = 1e-8
cholesky_operators = modified_cholesky_direct(
  eri.reshape(norb**2, norb**2), tol=chol_tol, verbose=False
).reshape(-1, norb, norb)

onebody = X.T @ onebody @ X
cholesky_operators = 1j * np.einsum("ij,njk,kl->nil", X.T, cholesky_operators, X)


onebody += 0.5 * np.einsum("nij,njk", cholesky_operators, cholesky_operators).real

# double along spin dimension
cholesky_operators = np.stack([cholesky_operators, cholesky_operators], axis=1)
onebody = np.stack([onebody, onebody], axis=0)


mf = mol.UHF.run()


T = np.zeros((norb, norb))

H = autohf.AutoHFHamiltonian(T=(T, T), U=0)
nelec = np.sum(mf.get_occ(), axis=1).astype(int)

hf_settings = {
  "verbose": False,
  "steps": 2000,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "batch_size": 4,
  "gpu": False,
  "nelec": nelec,
  "noncollinear": False,
  "state0_scale": 0.01,
  "seed": 1234,
  "measure_spin": True,
}


def molecular_energy(state, rdms):
  return (
    autohf.terms.cholesky.hf_onebody(rdms, onebody)
    + autohf.terms.cholesky.hf_cholesky(rdms, cholesky_operators)
    + nuclear_repulsion
  )


data, data_functions = autohf.solve_hf(
  H, settings=hf_settings, custom_energy_term=molecular_energy
)
rounding_level = -np.log10(chol_tol).astype(int)
# recall we have a cholesky cutoff, so we will be slightly off in evaluating the energy
# practically this isn't a huge issue
print("PySCF energy:", np.round(mf.energy_tot(), rounding_level))
print("AutoHF      :", np.round(data["E"], rounding_level))
