"""
Interface to the lattice Hamiltonians of safiretools.

The helpers only use accessors that the legacy afqmctools Hamiltonian shares, so they read both.
"""

import importlib

import numpy as np

# optionally import safiretools
if importlib.util.find_spec("safiretools") is not None:
  HAS_SAFIRETOOLS = True
  from safiretools import LatticeHamiltonian
else:
  HAS_SAFIRETOOLS = False

  class LatticeHamiltonian:
    pass


def _get_T(hamiltonian: LatticeHamiltonian):
  """
  get the T matrix from the lattice Hamiltonian instanace, `hamiltonian`
  """

  T = hamiltonian.get_one_body()

  if T is None:
    return None

  N = hamiltonian.nbands * hamiltonian.nsites
  # compare by name to also accept the legacy afqmctools SpinSymm
  if hamiltonian.spin_symm.name == "COLLINEAR":
    # TODO: RHF symmetry / batched up&dn
    # if not np.allclose(T[:N,:],T[N:,:]):
    #      Tup, Tdown = T[:N,:],T[N:,:]
    #      T = [Tup,Tdown]
    #  else:
    #      T = [T[:N,:]]
    Tup, Tdown = T[:N, :], T[N:, :]
    T = [Tup, Tdown]
  else:
    T = [T[: N * 2, :]]

  return T


def _get_U(hamiltonian: LatticeHamiltonian):
  """
  get U, U1, and U2 from hamiltonian.
  """
  if "Uij" not in hamiltonian.keys():
    U = U1 = U2 = None
  else:
    fullU = hamiltonian.get_U().toarray()
    M = hamiltonian.nsites * hamiltonian.nbands
    U = np.diag(fullU[:M, :M].diagonal())
    U1 = fullU[:M, :M].copy() - U
    if np.allclose(U1, np.zeros_like(U1)):
      U1 = None

    if fullU.shape[0] == 2 * M:
      U2 = fullU[M:, :M]
    else:
      U2 = None

  return U, U1, U2


def _get_J(hamiltonian: LatticeHamiltonian):
  if "Jij" in hamiltonian.keys():
    return hamiltonian.get_J().toarray()
  else:
    return None


def _get_heisenberg(hamiltonian: LatticeHamiltonian):
  if "J_heisenberg" in hamiltonian.keys():
    return hamiltonian.get_heisenberg().toarray()
  else:
    return None


def _get_terms(hamiltonian):
  """
  given a lattice model Hamiltonian instance, return all terms.

  Input:
  - hamiltonian: safiretools.LatticeHamiltonian instance

  Returns:
  - ? we can't jsut return a list of csr matrices; that can't be interpretted.

  Notes:
  the Hamiltonian may have multiple of each type of term.
  For example, when we have terms with different signs,
  we may keep them separate so that we can use different
  Hubbard-Stratonovich transformation (HST) types for each term.
  For HF, we can combine these into one term of each type (no HST),
  but we still need to find all terms.
  """
  raise NotImplementedError
