import importlib
import sys

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

if importlib.util.find_spec("afqmctools") is None:
  print("afqmctools is required to run this example, please install it")
  sys.exit(0)
if importlib.util.find_spec("pyscf") is None:
  print("pyscf is required to run this example, please install it")
  sys.exit(0)

import pyscf
import afqmctools.utils.linalg
import numpy as np
import scipy.linalg
import autohf

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

X = afqmctools.utils.linalg.get_ortho_ao_mol(overlap)

norb = onebody.shape[0]
print(f"{norb=}")

chol_tol = 1e-8
cholesky_operators = afqmctools.utils.linalg.modified_cholesky_direct(
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


data, data_functions = autohf.solve_hf(H, settings=hf_settings, custom_energy_term=molecular_energy)
rounding_level = -np.log10(chol_tol).astype(int)
# recall we have a cholesky cutoff, so we will be slightly off in evaluating the energy
# practically this isn't a huge issue
print("PySCF energy:", np.round(mf.energy_tot(), rounding_level))
print("AutoHF      :", np.round(data["E"], rounding_level))
