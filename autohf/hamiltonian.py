import importlib
import copy

import numpy as np
from .options import RepOptions, SlaterType
from .rotation_tools import makeRot
import jax.numpy as jnp

from .terms.uhf import known_terms as uhf_known_terms
from .terms.ghf import known_terms as ghf_known_terms
from .terms.cholesky import known_terms as cholesky_known_terms

known_terms = uhf_known_terms + ghf_known_terms + cholesky_known_terms

# optionally import afqmctools
if importlib.util.find_spec("afqmctools") is not None:
  HAS_AFQMCTOOLS = True
  from afqmctools.hamiltonian.model.ham_class import SpinSymm, Hamiltonian

  from .interfaces.afqmctools import _get_T, _get_U, _get_J, _get_heisenberg
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
  r"""
  Container class to hold Hamiltonian terms for AutoHF in internal format.

  Hamiltonian terms are stored as ``(vals, sites)`` pairs, where ``vals``
  is a list of unique non-zero coupling values and ``sites`` is a list of
  the corresponding site-index arrays at which each value appears.

  Parameters
  ----------
  source : AutoHFHamiltonian or afqmctools.Hamiltonian, optional
      If source is an ``AutoHFHamiltonian``, performs a deep-copy (copy constructor).
      If source is an afqmctools ``Hamiltonian``, reads T, U, U1, U2, J, and
      Heisenberg terms from it.  If ``None`` (default), builds from
      keyword arguments.
  settings : dict, optional
      Convenience dict for lattice geometry.  Expected keys:
      * ``"L"`` — ``(Nx, Ny)`` tuple of lattice dimensions (required).
      * ``"Nbands"`` — number of orbitals per site (default 1).
      Takes precedence over the ``Nx``, ``Ny``, and ``Nbands`` kwargs.
  **kwargs
      Hamiltonian terms and lattice metadata passed directly.  See
      ``add_terms`` for the accepted term keywords.  Additional keys:
      * ``N`` — total number of orbitals (sites × Nbands; inferred from T or settings if omitted).
      * ``Nx``, ``Ny`` — lattice dimensions (used for metadata only).
      * ``Nbands`` — number of orbitals per site (default 1).
  
  Supported Hamiltonian Terms
  ---------------------------
  Supply each term as a keyword argument, and supply input arrays in the following formats.
    
  T : array-like, shape (nspin, N, N)

    One-body hopping matrix. Implements 
    :math:`\\sum_{ij,\\sigma} T_{ij} c_{i\\sigma}^\\dagger c_{j\\sigma}`.
      
    Must be a list/tuple/ndarray with one
    NxN matrix per spin sector: ``(T_up, T_dn)`` for collinear (UHF)
    or ``[T_total]`` for non-collinear (GHF).

  U : scalar, 1-D array of length N, or NxN matrix

    On-site Hubbard interaction.  Implements
    :math:`\\sum_i U_i n_{i\\uparrow} n_{i\\downarrow}`.

      Accepted forms:

      * scalar — expanded to ``np.ones(N) * U`` then diagonalized.
      * 1-D array of length N — treated as diagonal of the coupling matrix.
      * NxN matrix — all non-zero entries are used as per-orbital strengths
        (row indices are stored; off-diagonal entries contribute their row
        index, which is the intended behavior for orbital-dependent U).

  U1 : NxN array

    Inter-orbital mixed-spin density-density coupling.  Implements
    :math:`\\sum_{i<j,\\sigma} U1_{ij}\, n_{i\\sigma} n_{j\\bar\\sigma}`.

    Stored as ``_terms["U1"] = (U1vals, U1Sites)``
    where ``U1Sites[k]`` is a tuple ``(i_array, j_array)`` of pairs
    with ``i < j`` and coupling value ``U1vals[k]``.

    Only the upper-triangular part of the matrix is used.

  U2 : NxN array

    Inter-orbital same-spin density-density coupling (Wick-contracted with
    the exchange correction).  Implements
    :math:`\\sum_{i<j,\\sigma} U2_{ij}\, n_{i\\sigma} n_{j\\sigma}`.

    Uses same format as U1, stored under ``_terms["U2"]``.

    Only the upper-triangular part of the matrix is used.

  J : NxN array

    Hund's coupling (HK-model form).  Encodes pair-hopping and
    spin-flip processes between orbitals ``i`` and ``j``. Implements
    :math:`\\sum_{i<j} J_{ij} (c_{i\\uparrow}^\\dagger c_{j\\uparrow} c_{j\\downarrow}^\\dagger c_{i\\downarrow} + h.c.)`

    Uses same format as U1, stored under ``_terms["J"]``.

    Only the upper-triangular part of the matrix is used.
    
  Jheis, Jheisenberg : NxN array

    Heisenberg spin-spin coupling. Implements
    :math:`\\sum_{i<j} J_{ij}\, \\vec{S}_i \\cdot \\vec{S}_j`.
    
    Uses same format as U1, stored under ``_terms["J_heisenberg"]``.

    ``Jheisenberg`` is an alias for ``Jheis``.
  """

  def __init__(self, source=None, settings=None, **kwargs) -> None:
    self._terms = {}
    self._one_body = None
    self.N = None
    self.Nx, self.Ny = None, None
    self.Nbands = None

    if isinstance(source, Hamiltonian):
      self._init_from_afqmctools(source)
    elif isinstance(source, AutoHFHamiltonian):
      # TODO copy constructor
      self._terms = copy.deepcopy(source._terms)
      self._one_body = copy.deepcopy(source._one_body)
      self.Nx, self.Ny = source.Nx, source.Ny
      self.N = source.N
      self.Nbands = source.Nbands
    elif source is None:
      if settings is not None:
        self.Nbands = settings.get("Nbands", 1)
        self.Nx, self.Ny = settings["L"]
        self.N = self.Nx * self.Ny * self.Nbands
      elif "T" in kwargs:
        # T will set N later when we sort out what it is
        self.N = kwargs.get("N", None)
        self.Nx, self.Ny = kwargs.get("Nx", None), kwargs.get("Ny", None)
        self.Nbands = kwargs.get("Nbands", 1)
      else:
        # in this case, we can still use .set() to add terms
        pass
    else:
      raise ValueError(
        f"[For Developers] Can't initialize AutoHF Hamiltonian  \
                                from source of type {type(source)}"
      )
    self.add_terms(**kwargs)

  def add_terms(self, **kwargs):
    r"""
    Add one or more Hamiltonian terms.
    
    Parameters
    ----------
    **kwargs
        Hamiltonian terms and lattice metadata passed directly.  See
        ``Supported Hamiltonian Terms`` in the docstring of AutoHFHamiltonian
        for the accepted term keywords and array formats.
    """
    self._add_T(kwargs.get("T", None))
    # NOTE: T must go first as it sets N for potential scalar U
    self._add_U(kwargs.get("U", None), kwargs.get("U1", None), kwargs.get("U2", None))
    self._add_J(kwargs.get("J", None))
    if "Jheisenberg" in kwargs:  # alias
      self._add_heisenberg(kwargs["Jheisenberg"])
    self._add_heisenberg(kwargs.get("Jheis", None))

  def _init_from_afqmctools(self, hamiltonian):
    self.Nbands = hamiltonian.nbands
    self.nsites = hamiltonian.nsites
    self.N = hamiltonian.nsites * self.Nbands
    # self.spin_symm = hamiltonian.spin_symm

    T = _get_T(hamiltonian)
    self._add_T(T)
    U, U1, U2 = _get_U(hamiltonian)
    self._add_U(U, U1, U2)
    J = _get_J(hamiltonian)
    self._add_J(J)
    J = _get_heisenberg(hamiltonian)
    self._add_heisenberg(J)

  def _add_T(self, T):
    """
    Parse and store the one-body hopping matrix.

    see docstring of AutoHFHamiltonian for accepted formats.

    T is stored both as the raw ``_one_body`` array and in the compressed
    ``_terms["T"]`` representation ``(Tvals, TSites)`` where

    * ``Tvals[sigma]`` — 1-D array of unique non-zero hopping values for
      spin sector ``sigma``.
    * ``TSites[sigma]`` — list of ``(i_array, j_array)`` index pairs, one
      entry per unique value in ``Tvals[sigma]``.

    Also sets ``self.N`` from ``len(T[0])`` when it has not been set yet.
    """
    # assume [up,dn] for collinear
    # and [total] for NC

    if T is not None:
      Tvals = []
      TSites = []

      # clean up shaping
      if isinstance(T, np.ndarray):
        # if len(T.shape) == 2:
        #  T = np.stack([T, T])

        if len(T.shape) != 3:
          raise RuntimeError(
            f"Passed T matrix has shape {T.shape}; needs to be (nspin,N,N) instead"
          )

      # Assume this is GHF, let's enforce that two T need to be passed for collinear
      # if (isinstance(T, list) or isinstance(T, tuple)) and len(T) == 1:
      #  T = (T[0], T[0])

      # up to one T matrix per spin sector
      for sigma, T_sigma in enumerate(T):
        # TODO: make this output cleaner!
        print(f"Reading hopping from spin sector {sigma}")
        Tvals.append(np.unique(T_sigma[T_sigma != 0.0]))
        TSites.append([_sites_where(T_sigma, t) for t in Tvals[sigma]])

      self._terms["T"] = (Tvals, TSites)
      print(f"Added unique T terms: {Tvals}")
      if self.N is None:
        self.N = len(T[0])

      self._one_body = T

  def _add_U(self, U, U1, U2):
    """
    Parse and store density-density interaction terms.

    see docstring of AutoHFHamiltonian for accepted formats.
    """
    if U is not None:
      if isinstance(U, float) or isinstance(U, int):
        if self.N is None:
          raise RuntimeError("N must be set before passing scalar U")
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
    """
    Parse and store the Hund's coupling term.

    see docstring of AutoHFHamiltonian for accepted formats.
    """
    if J is not None:
      Jvals = np.unique(J[J != 0.0])
      JSites = [_sites_where(J, j, i_less_than_j=True) for j in Jvals]
      self._terms["J"] = (Jvals, JSites)
      print(f"Added unique J terms: {Jvals}")

  def _add_heisenberg(self, J):
    """
    Parse and store the Heisenberg spin-spin coupling term.

    see docstring of AutoHFHamiltonian for accepted formats.
    """
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
    if key not in known_terms:
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
