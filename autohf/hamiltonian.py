import importlib

import numpy as np
from .options import RepOptions, SlaterType
from .rotation_tools import makeRot
import jax.numpy as jnp

from .terms.uhf import known_terms as uhf_known_terms
from .terms.ghf import known_terms as ghf_known_terms

# optionally import afqmctools
if importlib.util.find_spec("afqmctools") is not None:
  HAS_AFQMCTOOLS = True
  from afqmctools.hamiltonian.model.ham_class import SpinSymm, Hamiltonian

  from .converters import _get_T, _get_U, _get_J, _get_heisenberg
else:
  HAS_AFQMCTOOLS = False

  class Hamiltonian:
    def __init__(self):
      pass

  class SpinSymm:
    COLLINEAR = 0
    NONCOLLINEAR = 1
    CLOSED = 2


class AutoHFHamiltonian:
  """
  Conatiner class to hold Hamiltonian terms for AutoHF
      in internal format.
  """

  def __init__(self, source=None, settings=None) -> None:
    self._terms = {}
    self._one_body = None

    if isinstance(source, Hamiltonian):
      L = settings["L"]  # TODO: make this nicer?
      self._init_from_afqmctools(source, L)
    elif isinstance(source, AutoHFHamiltonian):
      raise NotImplementedError
    elif source is None:
      if settings is not None:
        self.Nbands = settings.get("Nbands", 1)
        self.Nx, self.Ny = settings["L"]
        self.N = self.Nx * self.Ny * self.Nbands
      else:
        # in this case, we can still use .set() to add terms
        pass
    else:
      raise ValueError(
        f"[For Developers] Can't initialize AutoHF Hamiltonian  \
                                from source of type {type(source)}"
      )

  def add_terms(self, **kwargs):
    self._add_T(kwargs.get("T", None))
    self._add_U(kwargs.get("U", None), kwargs.get("U1", None), kwargs.get("U2", None))
    self._add_J(kwargs.get("J", None))
    if "Jheisenberg" in kwargs:  # alias
      self._add_heisenberg(kwargs["Jheisenberg"])
    self._add_heisenberg(kwargs.get("Jheis", None))

  def _init_from_afqmctools(self, hamiltonian, L):
    self.Nbands = hamiltonian.nbands
    self.N = hamiltonian.nsites * self.Nbands
    self.nsites = hamiltonian.nsites
    self.Nx, self.Ny = L
    # self.spin_symm = hamiltonian.spin_symm

    T = _get_T(hamiltonian)
    self._add_T(T)
    U, U1, U2 = _get_U(hamiltonian)
    self._add_U(U, U1, U2)
    J = _get_J(hamiltonian)
    self._add_J(hamiltonian)
    J = _get_heisenberg(hamiltonian)
    self._add_heisenberg(J)

  def _add_T(self, T):
    # assume [up,dn] for collinear
    # and [total] for NC

    if T is not None:
      Tvals = []
      TSites = []

      # clean up shaping
      if isinstance(T, np.ndarray):
        if len(T.shape) == 2:
          T = np.stack([T, T])

        if len(T.shape) != 3:
          raise RuntimeError(
            f"Passed T matrix has shape {T.shape}; needs to be (nspin,N,N) instead"
          )

      if (isinstance(T, list) or isinstance(T, tuple)) and len(T) == 1:
        T = (T[0], T[0])

      # up to one T matrix per spin sector
      for sigma, T_sigma in enumerate(T):
        # TODO: make this output cleaner!
        print(f"Reading hopping from spin sector {sigma}")
        Tvals.append(np.unique(T_sigma[T_sigma != 0.0]))
        TSites.append([_sites_where(T_sigma, t) for t in Tvals[sigma]])

      self._terms["T"] = (Tvals, TSites)
      print(f"Added unique T terms: {Tvals}")

    self._one_body = T

  # def _add_T_from_afqmctools(self,hamiltonian):
  #    T = _get_T(hamiltonian)
  #    self._add_T(T)

  def _add_U(self, U, U1, U2):
    if U is not None:
      if isinstance(U, float) or isinstance(U, int):
        U = np.ones(self.N) * U
      if isinstance(U, list) or isinstance(U, tuple):
        U = np.array(U)
      if len(U.shape) == 1:
        U = np.diag(U)
      Uvals = np.unique(U[U != 0.0])
      USites = [_sites_where(U, u)[0] for u in Uvals]

      self._terms["U"] = (Uvals, USites)
      print(f"Added unique U terms: {Uvals}")

    if U1 is not None:
      U1vals = np.unique(U1[U1 != 0.0])
      U1Sites = [_sites_where(U1, u, i_less_than_j=True) for u in U1vals]
      self._terms["U1"] = (U1vals, U1Sites)
      print(f"Added unique U1 terms: {U1vals}")

    if U2 is not None:
      U2vals = np.unique(U2[U2 != 0.0])
      U2Sites = [_sites_where(U2, u, i_less_than_j=True) for u in U2vals]
      self._terms["U2"] = (U2vals, U2Sites)
      print(f"Added unique U2 terms: {U2vals}")

  def _add_J(self, J):
    if J is not None:
      Jvals = np.unique(J[J != 0.0])
      JSites = [_sites_where(J, j, i_less_than_j=True) for j in Jvals]
      self._terms["J"] = (Jvals, JSites)
      print(f"Added unique J terms: {Jvals}")

  def _add_heisenberg(self, J):
    if J is not None:
      Jvals = np.unique(J[J != 0.0])
      JSites = [_sites_where(J, j, i_less_than_j=True) for j in Jvals]
      self._terms["J_heisenberg"] = (Jvals, JSites)
      print(f"Added unique J_heisenberg terms: {Jvals}")

  def keys(self):
    return self._terms.keys()

  def __getitem__(self, key):
    return self._terms[key]

  def get(self, key, default=None):
    return self._terms.get(key, default)

  def set(self, key, value):
    # TODO: filter based on allowed keys
    if key not in uhf_known_terms and key not in ghf_known_terms:
      raise ValueError(f"Unknown term {key}")

    self._terms[key] = value

  def get_one_body(self):
    return self._one_body


def _sites_where(M, val, i_less_than_j=False):
  inds = np.argwhere(M == val)
  if i_less_than_j:
    inds_i_less_than_j = np.argwhere(inds[:, 0] < inds[:, 1])
    return (inds.T[0, inds_i_less_than_j], inds.T[1, inds_i_less_than_j])
  else:
    return (inds.T[0, :], inds.T[1, :])
