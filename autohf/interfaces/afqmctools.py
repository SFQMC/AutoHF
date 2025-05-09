import importlib
import logging as log

import numpy as np
from ..options import RepOptions, SlaterType
import jax.numpy as jnp

# optionally import afqmctools
if importlib.util.find_spec("afqmctools") is not None:
  HAS_AFQMCTOOLS = True
  from afqmctools.hamiltonian.model.ham_class import SpinSymm, Hamiltonian
else:
  HAS_AFQMCTOOLS = False

  class Hamiltonian:
    def __init__(self):
      pass

  class SpinSymm:
    COLLINEAR = 0
    NONCOLLINEAR = 1
    CLOSED = 2


def fail_without_afqmctools(func):
  def wrapper(*args, **kwargs):
    if HAS_AFQMCTOOLS:
      return func(*args, **kwargs)
    else:
      raise RuntimeError(
        f"called function {func.__name__} which requires afqmctools. "
        "Install afqmctools to use this function."
      )

  return wrapper


def skip_without_afqmctools(func):
  if not HAS_AFQMCTOOLS:
    log.warning(f"Skipping {func.__name__} because afqmctools is not installed.")
    return lambda *args, **kwargs: None
  return func


if HAS_AFQMCTOOLS:
  from afqmctools.observables.spin import spin_squared, spin_z
else:
  spin_squared = skip_without_afqmctools(lambda *args, **kwargs: None)
  spin_z = skip_without_afqmctools(lambda *args, **kwargs: None)


@fail_without_afqmctools
def _get_T(hamiltonian: Hamiltonian):
  """
  get the T matrix from the lattice Hamiltonian instanace, `hamiltonian`
  """

  T = hamiltonian.get_one_body()

  if T is None:
    return None

  N = hamiltonian.nbands * hamiltonian.nsites
  if hamiltonian.spin_symm == SpinSymm.COLLINEAR:
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


@fail_without_afqmctools
def _get_U(hamiltonian: Hamiltonian):
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


@fail_without_afqmctools
def _get_J(hamiltonian: Hamiltonian):
  if "Jij" in hamiltonian.keys():
    return hamiltonian.get_J().toarray()
  else:
    return None


@fail_without_afqmctools
def _get_heisenberg(hamiltonian: Hamiltonian):
  if "J_heisenberg" in hamiltonian.keys():
    return hamiltonian.get_heisenberg().toarray()
  else:
    return None


@fail_without_afqmctools
def _get_terms(hamiltonian):
  """
  given a lattice model Hamiltonian instance, return all terms.

  Input:
  - hamiltonian: afqmctools.hamiltonian.model.Hamiltonian instance

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
