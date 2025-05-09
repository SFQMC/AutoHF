import os
import time

import numpy as np

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax
import optax
import jaxopt

import warnings

import scipy.sparse as sps  # TODO: Remove

# TODO: make optional
from autohf.interfaces.afqmctools import fail_without_afqmctools, spin_squared, spin_z


from .io import read_input_params
from .tools import sdToRDM, sdToRDM_noncollinear
from .options import SlaterType, RepOptions, get_ansatz, process_args
from .rotation_tools import (
  jexpm,
  get_nonz,
  get_nonz_herm,
)
from .converters import convertInitialSlaterDet
from .hamiltonian import AutoHFHamiltonian
from .terms.uhf import (
  uhf_U,
  uhf_U1,
  uhf_U2,
  uhf_J,
  uhf_Jheis,
)
from .terms.ghf import (
  ghf_U,
  ghf_U1,
  ghf_U2,
  ghf_J,
  ghf_Jheis,
)


def saveStates(states, orbFunc, energies, phi0, args, nelec, fname, dumpTrials):
  """Save all the trial states in an extra file
  Will automatically save based on npy/pkl/pickle file extension
  Does nothing if fname is empty/None
  """
  import pickle

  Es = np.array(energies)
  best = np.argmin(Es)
  o = orbFunc(states[best])
  data = {
    "state": states[best],
    "Es": Es,
    "E": Es[best],
    "orbitals": o,
    "args": args,
    "phi0": phi0,
    "states": np.array(states),
    "nelec": nelec,
  }
  if dumpTrials:
    orbs = [np.array(o) for o in jax.vmap(orbFunc)(states)]
    data["all_orbitals"] = orbs

  if fname == "" or fname is None:
    return data
  if fname[-4:] == ".npy":
    print("Saving states as npy:", fname)
    np.save(fname, [np.array(o) for o in orbs])
  elif fname[-3:] == ".h5":
    raise NotImplementedError("Dumping states to h5 not yet supported")
  elif fname[-4:] == ".pkl" or fname[-7:] == ".pickle":
    print("Saving Pickle:", fname)
    with open(fname, "wb") as f:
      pickle.dump(data, f)
  else:
    print("Ignoring file extension. Saving states with pickle extension", fname + ".pkl")
    print("Saving Pickle:", fname + ".pkl")
    with open(fname + ".pkl", "wb") as f:
      pickle.dump(data, f)

  return data


def loadStates(fname, args, nelec):
  """
  Loads saved pkl from saveStates
  Supports using the same ansatz or any ansatz -> SD/SDC
  """
  import pickle
  # TODO: support SD/SDC ->SD_ROT

  print(f"Loading saved data from {fname}")
  if fname[-4:] != ".pkl" and fname[-7:] != ".pickle":
    print("Assuming pkl/pickle format, other formats currently unsupported")

  with open(fname, "rb") as f:
    original_data = pickle.load(f)

  Es = original_data["Es"]
  order = np.argsort(Es)
  phi0 = original_data["phi0"]

  if original_data["args"].numTrials < args.numTrials:
    raise NotImplementedError(
      f"Old data had {original_data['args'].numTrials} trials "
      f"but requesting more on restart {args.numTrials}. This is currently unsupported"
    )

  if args.ansatz == original_data["args"].ansatz:
    # easy case
    state0_ref = original_data["state"]
    state0 = original_data["states"][order[: args.numTrials]]

  elif args.ansatz == "SD" or args.ansatz == "SDC":
    state0_ref = jnp.array(original_data["orbitals"])
    if "all_orbitals" not in original_data:
      raise RuntimeError(
        "All orbitals were not saved (i.e. dumpTrials was not set to True). "
        "Please rerun original calculation or keep the same trial"
      )
    # orbitals leading dim is (nspin,nwalkers,...)
    # so need a reshape
    original_nelec = original_data["nelec"]
    if original_nelec[0] != original_nelec[1]:
      raise NotImplementedError("Polarized calculation not implemented (please raise issue)")

    state0 = jnp.stack([o[order[: args.numTrials]] for o in original_data["all_orbitals"]])
    state0 = jnp.swapaxes(state0, 0, 1)
    # if originally not SDC and now is SDC
    if original_data["args"].ansatz != "SDC" and args.ansatz == "SDC":
      state0 = jnp.stack([state0.real, state0.imag])
  print("Loaded state with energy:", original_data["E"])

  return state0, state0_ref, phi0


def construct_random_Phi(rng, N, Nes, spin_symm=SlaterType.COLLINEAR, delta=0.00001, Phi=None):
  """
  construct a random Slater determinant as

  e^{-i*delta*X} Phi_{ia}

  where Phi_{ia} is a Slater determinant, X is a random symmetric matrix,
  and delta is a real number used to control the amplitude of the random noise.

  Inputs:
  - rng: initialized numpy random number generator
  - N : number of basis set functions (number of sites * number of bands)
  - Nes : number of electrons as an iterable of length 2 (Nup,Ndown)
  - spin_symm : the desired spin symmetry of the Slater determinant
  """
  print("Making a randomized initial guess")
  if spin_symm == SlaterType.NONCOLLINEAR:
    slater_shape = (2 * N, sum(Nes))
    shape = (2 * N, 2 * N)
  else:
    slater_shape = (N, sum(Nes))
    shape = (N, N)

  if Phi is None:
    Phi = np.eye(*slater_shape)

  X = rng.random(shape)
  X = 0.5 * (X + X.T)

  Phi = np.exp(-1j * delta * X) @ Phi

  if spin_symm == SlaterType.NONCOLLINEAR:
    return jnp.stack([Phi])
  elif spin_symm == SlaterType.COLLINEAR:
    return (Phi[:, : Nes[0]], Phi[:, : Nes[1]])


def constructPhi0(Tlist, N, Nes, spin_symm=SlaterType.COLLINEAR, averageSpins=False):
  # assume Tlist is a list of either [NC] or [up,dn]
  # if averageSpins, combine up and down
  # eventually replace that with hamiltonian symmetry system
  if Tlist is None:
    return None

  # NC T, NC walkers
  if Tlist[0].shape[0] != N:
    if spin_symm != SlaterType.NONCOLLINEAR:
      raise RuntimeError(
        "If Hamiltonian is noncollinear then HF ansatzalso need to be noncollinear"
      )
    assert Tlist[0].shape[0] == 2 * N
    if len(Tlist) > 1:
      warnings.warn(
        "Construsting phi0 with noncollinear Hamiltonian"
        "with multiple hopping T, only taking the first",
        RuntimeWarning,
        stacklevel=2,
      )
    T = Tlist[0]
    if isinstance(T, sps.sparray):
      T = T.toarray()
    _, orbitals = np.linalg.eigh(T)
    return jnp.stack([orbitals[:, : sum(Nes)]])

  # collinear T version
  assert np.all([T.shape[0] == N for T in Tlist])
  Tup = Tlist[0]
  if len(Tlist) == 1:  # collinear sym
    Tdn = Tup
  elif averageSpins:
    Tavg = sum(Tlist) / len(Tlist)
    Tup = Tdn = Tavg
  else:
    Tdn = Tlist[1]

  # in case we have sparse arrays
  if isinstance(Tup, sps.sparray):
    Tup = Tup.toarray()
    Tdn = Tdn.toarray()

  if spin_symm == SlaterType.COLLINEAR:
    _, orbitals_up = np.linalg.eigh(Tup)
    _, orbitals_down = np.linalg.eigh(Tdn)

    return (orbitals_up[:, : Nes[0]], orbitals_down[:, : Nes[1]])

  elif spin_symm == SlaterType.NONCOLLINEAR:
    # collinear Hamiltonian, NC walkers
    _, orbitals = np.linalg.eigh(np.block([[Tup, np.zeros_like(Tdn)], [np.zeros_like(Tup), Tdn]]))
    return jnp.stack([orbitals[:, : Nes[0] + Nes[1]]])


###########################################################################


# TODO: simplify energy call factory once things are working, AND we have unit tests
# TODO: use 'if' statements, not this dispatching function!
def makeEnergyCall(hamil: AutoHFHamiltonian, spin_symm, **kwargs):
  N = kwargs.get("N", hamil.N)
  debug_mode = kwargs.get("debug_mode", False)
  rep_type = kwargs.get("rep_type", RepOptions.SLATERDET)

  if rep_type is None:  # legacy
    rep_type = RepOptions.SLATERDET

  def _sites_to_ii_jj(sites):
    """
    convert from (i_vals,j_vals) to (ii_vals,jj_vals)
    """
    i = sites[0]
    j = sites[1]

    ii = np.zeros((2, i.shape[0]), dtype=np.int64)
    jj = np.zeros((2, j.shape[0]), dtype=np.int64)

    ii[0, :] = ii[1, :] = i[:, 0]
    jj[0, :] = jj[1, :] = j[:, 0]

    return ii, jj

  def _sites_to_ij_ji(sites):
    i = sites[0]
    j = sites[1]
    ij = np.zeros((2, i.shape[0]), dtype=np.int64)
    ij[0, :], ij[1, :] = i[:, 0], j[:, 0]
    ji = np.zeros((2, i.shape[0]), dtype=np.int64)
    ji[0, :], ji[1, :] = j[:, 0], i[:, 0]

    return ij, ji

  ######## call logic
  if "ref_state" in kwargs:
    phi0 = kwargs["ref_state"]  # should be 2*N x sum(nelec)
    if spin_symm != SlaterType.GHF:
      phi0_up, phi0_dn = phi0
  riemannian = kwargs.get("riemannian", False)

  found_interaction = False

  if "T" in hamil.keys():
    Tcoeffs, ThopList = hamil["T"]
  else:
    Tcoeffs = ThopList = None

  if "U" in hamil.keys():
    Ucoeffs, USites = hamil["U"]
    found_interaction = True
  else:
    Ucoeffs, USites = [], []

  if "U1" in hamil.keys():
    U1coeffs, U1Sites = hamil["U1"]
    found_interaction = True
  else:
    U1coeffs, U1Sites = [], []
  ii_jj_for_U1 = [_sites_to_ii_jj(sites) for sites in U1Sites]
  ij_ji_for_U1 = [_sites_to_ij_ji(sites) for sites in U1Sites]

  if "U2" in hamil.keys():
    U2coeffs, U2Sites = hamil["U2"]
    found_interaction = True
  else:
    U2coeffs, U2Sites = [], []
  ii_jj_for_U2 = [_sites_to_ii_jj(sites) for sites in U2Sites]
  ij_ji_for_U2 = [_sites_to_ij_ji(sites) for sites in U2Sites]

  if "J" in hamil.keys():
    Jcoeffs, JSites = hamil["J"]
    found_interaction = True
  else:
    Jcoeffs, JSites = [], []
  ij_ji_for_J = [_sites_to_ij_ji(sites) for sites in JSites]
  ii_jj_for_J = [_sites_to_ii_jj(sites) for sites in JSites]

  if "J_heisenberg" in hamil.keys():
    J_heisenberg_coeffs, J_heisenberg_Sites = hamil["J_heisenberg"]
    found_interaction = True
  else:
    J_heisenberg_coeffs, J_heisenberg_Sites = [], []
  ij_ji_for_Jheis = [_sites_to_ij_ji(sites) for sites in J_heisenberg_Sites]
  ii_jj_for_Jheis = [_sites_to_ii_jj(sites) for sites in J_heisenberg_Sites]

  if not found_interaction:
    raise ValueError("Have to define an interaction!")

  ############

  if rep_type == RepOptions.SLATERDET:
    orbFunc = lambda x: x
    if spin_symm == SlaterType.GHF:
      rdmFunc = sdToRDM_noncollinear  # could make safe
    elif spin_symm == SlaterType.UHF:
      rdmFunc = sdToRDM  # could make safe
    else:
      raise NotImplementedError("Unkown rdmfunc for SLATERDET")
  elif rep_type == RepOptions.SLATERDETC:

    def orbFunc(state):
      return state[0] + state[1] * 1j

    def rdmFunc(state):
      o = orbFunc(state)
      if spin_symm == SlaterType.GHF:
        return sdToRDM_noncollinear(o)
      elif spin_symm == SlaterType.UHF:
        return sdToRDM(o)
      else:
        raise NotImplementedError("Unkown rdmfunc for SLATERDET")
  elif rep_type == RepOptions.SLATERDET_ROT:
    if "approx_expm" in kwargs:
      use_approx_expm = kwargs["approx_expm"]
    else:
      use_approx_expm = False

    if use_approx_expm:
      print("Using approx expm")

    def makeRot(v):
      if spin_symm == SlaterType.GHF:
        xc = jnp.concatenate([v, v[2 * N :][::-1]])
        skewHerm = jnp.triu(xc.reshape([2 * N, 2 * N]), k=0)
      else:
        xc = jnp.concatenate([v, v[N:][::-1]])
        skewHerm = jnp.triu(xc.reshape([N, N]), k=0)
      skewHerm -= skewHerm.conj().T
      if use_approx_expm:
        Rot = jexpm(skewHerm)
      else:
        Rot = jax.scipy.linalg.expm(skewHerm)
      return Rot

    if spin_symm == SlaterType.GHF:
      # note, phi0 has shape (1,2*N,sum(nelec))
      # so we pick up the correct (1,) from numpy
      # broadcasting rules
      def orbFunc(state):
        R = makeRot(state[0])
        orbs = R @ phi0
        return orbs

      def rdmFunc(state):
        R = makeRot(state[0])
        orbs = R @ phi0
        return sdToRDM_noncollinear(orbs)
    else:

      def orbFunc(state):
        R_up, R_dn = makeRot(state[0]), makeRot(state[1])
        orbs = (R_up @ phi0_up, R_dn @ phi0_dn)
        return orbs

      def rdmFunc(state):
        R_up, R_dn = makeRot(state[0]), makeRot(state[1])
        orbs = (R_up @ phi0_up, R_dn @ phi0_dn)
        return sdToRDM(orbs)
  elif rep_type == RepOptions.SLATERDET_ROTC:
    if "approx_expm" in kwargs:
      use_approx_expm = kwargs["approx_expm"]
    else:
      use_approx_expm = False

    if use_approx_expm:
      print("Using approx expm")

    def makeRot(v):
      if spin_symm == SlaterType.GHF:
        xc = jnp.concatenate([v, v[2 * N :][::-1]])
        Heff = jnp.triu(xc.reshape([2 * N, 2 * N]), k=0)
      else:
        xc = jnp.concatenate([v, v[N:][::-1]])
        Heff = jnp.triu(xc.reshape([N, N]), k=0)
      Heff += jnp.tril(Heff.conj().T, k=-1)
      if use_approx_expm:
        Rot = jexpm(1j * Heff)
      else:
        Rot = jax.scipy.linalg.expm(1j * Heff)
      return Rot

    if spin_symm == SlaterType.GHF:
      # note, phi0 has shape (1,2*N,sum(nelec))
      # so we pick up the correct (1,) from numpy
      # broadcasting rules
      def orbFunc(state):
        R = makeRot(state[0])
        orbs = R @ phi0
        return orbs

      def rdmFunc(state):
        R = makeRot(state[0])
        orbs = R @ phi0
        return sdToRDM_noncollinear(orbs)
    else:

      def orbFunc(state):
        R_up, R_dn = makeRot(state[0]), makeRot(state[1])
        orbs = (R_up @ phi0_up, R_dn @ phi0_dn)
        return orbs

      def rdmFunc(state):
        R_up, R_dn = makeRot(state[0]), makeRot(state[1])
        orbs = (R_up @ phi0_up, R_dn @ phi0_dn)
        return sdToRDM(orbs)
  elif rep_type == RepOptions.DIAG:
    Ne = kwargs["Ne"]
    Tlist_ = hamil.get_one_body()
    if spin_symm == SlaterType.GHF:
      if len(Tlist_) == 2:
        Tup, Tdn = Tlist_[0].toarray(), Tlist_[1].toarray()
        T_ = np.block([[Tup, np.zeros_like(Tdn)], [np.zeros_like(Tup), Tdn]])
      else:
        # TODO account for potential RHF like Hamiltonian
        T_ = Tlist_[0].toarray()
    else:
      T_ = np.zeros((2, N, N))
      T_[0, :] = Tlist_[0].toarray()

      if len(Tlist_) == 1:
        T_[0, :] = T_[1, :]
      else:
        T_[1, :] = Tlist_[1].toarray()
    print("Hopping used:", T_)

    def makefunc(Ne, T_):
      Ne2 = np.sum(Ne)

      def orbFunc(state):
        Heff = T_ + (state + jnp.swapaxes(state, -1, -2).conj())  # symmetrized
        if spin_symm == SlaterType.GHF:
          orbs = (jnp.linalg.eigh(Heff)[1][:, :Ne2],)
          return orbs
        else:
          all_orbs = jnp.linalg.eigh(Heff)[1]
          orbs = [all_orbs[0, :, : Ne[0]], all_orbs[1, :, : Ne[1]]]
          return orbs

      def rdmFunc(state):
        if spin_symm == SlaterType.GHF:
          orbs = orbFunc(state)
          return sdToRDM_noncollinear(orbs)
        else:
          orbs = orbFunc(state)
          return sdToRDM(orbs)

      return orbFunc, rdmFunc

    orbFunc, rdmFunc = makefunc(Ne, T_)
  elif rep_type == RepOptions.CUSTOM:
    # a representation needs the following:
    # 1. orbFunc : state->orbitals
    # 2. rdmFunc : state->RDM
    # 3. state0 (initial state)
    # 4. state0_ref (initial state for RHF, default: 0*state0)
    orbFunc = kwargs["state2orbitals"]
    rdmFunc = kwargs["state2rdm"]
    # state0 = kwargs["state0"]
  else:
    raise NotImplementedError(f"Rep type {rep_type} Not Implemented (try CUSTOM)?")

  if spin_symm == SlaterType.GHF:
    if Tcoeffs is None and ThopList is None:

      def hopCall(rdms):
        return 0
    elif len(Tcoeffs) == len(ThopList) == 2:

      def hopCall(rdms):
        E = 0
        rdm_uu, rdm_dd = rdms[0], rdms[1]
        for t_up, hops in zip(Tcoeffs[0], ThopList[0]):
          E += t_up * jnp.sum(rdm_uu[hops[0], hops[1]])
        for t_down, hops in zip(Tcoeffs[1], ThopList[1]):
          E += t_down * jnp.sum(rdm_dd[hops[0], hops[1]])
        return E
    elif len(Tcoeffs) == len(ThopList) == 1:
      # assuming NC setup, TODO: RHF like
      # TODO: split hops into block list
      def hopCall(rdms):
        E = 0
        rdm = jnp.block([[rdms[0], rdms[2]], [rdms[3], rdms[1]]])
        for t, hops in zip(Tcoeffs[0], ThopList[0]):
          E += t * jnp.sum(rdm[hops[0], hops[1]])
        return E
    else:
      warnings.warn(
        "Unknown GHF hopping case, skipping hopCall."
        "Please tell developers what you're trying to do",
        RuntimeWarning,
        stacklevel=2,
      )

      def hopCall(rdms):
        return 0
  else:  # UHF version
    if Tcoeffs is None and ThopList is None:

      def hopCall(rdms):
        return 0
    elif isinstance(Tcoeffs, list) and len(Tcoeffs) == len(ThopList) == 2:

      def hopCall(rdms):
        E = 0
        for t_up, hops in zip(Tcoeffs[0], ThopList[0]):
          E += t_up * jnp.sum(rdms[0][hops[0], hops[1]])
        for t_down, hops in zip(Tcoeffs[1], ThopList[1]):
          E += t_down * jnp.sum(rdms[1][hops[0], hops[1]])
        return E
    else:  # no-hopping case TODO: just make a csr-matrix with all zeros!
      warnings.warn(
        "Unknown UHF hopping case, skipping hopCall."
        "Please tell developers what you're trying to do",
        RuntimeWarning,
        stacklevel=2,
      )

      def hopCall(rdms):
        return 0
    # elif Tcoeffs is not None and ThopList is not None:
    #    def hopCall(rdms):
    #        E = 0
    #        for t,hops in zip(Tcoeffs,ThopList):
    #            E += t*(jnp.sum(rdms[0][hops[0],hops[1]])+jnp.sum(rdms[1][hops[0],hops[1]]))
    #        return E
  # interaction terms
  if spin_symm == SlaterType.GHF:
    E_U = ghf_U
    E_U1 = ghf_U1
    E_U2 = ghf_U2
    E_J = ghf_J
    E_Jheis = ghf_Jheis
  else:
    E_U = uhf_U
    E_U1 = uhf_U1
    E_U2 = uhf_U2
    E_J = uhf_J
    E_Jheis = uhf_Jheis

  def UCall(rdms):
    E = 0
    for U, sites in zip(Ucoeffs, USites):
      E += U * E_U(rdms, sites)
    return E

  def U1call(rdms):
    E = 0
    for U1, diag_sites, off_diag_sites in zip(U1coeffs, ii_jj_for_U1, ij_ji_for_U1):
      ii, jj = diag_sites
      ij, ji = off_diag_sites
      E += U1 * E_U1(rdms, ii, jj, ij, ji)
    return E

  def U2call(rdms):
    E = 0
    for U2, diag_sites, off_diag_sites in zip(U2coeffs, ii_jj_for_U2, ij_ji_for_U2):
      ii, jj = diag_sites
      ij, ji = off_diag_sites
      E += U2 * E_U2(rdms, ii, jj, ij, ji)
    return E

  def Jcall(rdms):
    E = 0
    for J, diag_sites, off_diag_sites in zip(Jcoeffs, ii_jj_for_J, ij_ji_for_J):
      ii, jj = diag_sites
      ij, ji = off_diag_sites
      E += J * E_J(rdms, ii, jj, ij, ji)
    return E

  def Jheis_call(rdms):
    E = 0
    for J, diag_sites, off_diag_sites in zip(J_heisenberg_coeffs, ii_jj_for_Jheis, ij_ji_for_Jheis):
      ii, jj = diag_sites
      ij, ji = off_diag_sites
      E += J * E_Jheis(rdms, ii, jj, ij, ji)
    return E

  if riemannian:
    # using ideas from https://arxiv.org/abs/2110.08986
    # the only manifold implemented is a Stiefel manifold (SD/SDC)
    # check that we are SD only
    assert rep_type in [RepOptions.SLATERDET, RepOptions.SLATERDETC]
    beta = 1e-8  # TODO: calc this
    Ne = kwargs["Ne"]
    print(f"Using Riemannian CD mode, {beta=}")

    def A(X):
      XX = jnp.swapaxes(X.conj(), -1, -2) @ X
      return 1.5 * X - X @ (XX / 2)

    def rdmFunc_E(state):
      orbs = orbFunc(state)
      orbs = A(orbs)
      if spin_symm == SlaterType.GHF:
        return sdToRDM_noncollinear(orbs)
      elif spin_symm == SlaterType.UHF:
        return sdToRDM(orbs)

    def shift(state):
      orbs = orbFunc(state)
      Z = A(orbs)
      if spin_symm == SlaterType.GHF:
        return (
          beta
          / 4
          * jnp.linalg.norm((jnp.swapaxes(Z[0].conj(), -1, -2) @ Z[0]) - np.eye(sum(Ne)), "fro")
          ** 2
        )
      else:
        n1 = jnp.linalg.norm((jnp.swapaxes(Z[0].conj(), -1, -2) @ Z[0]) - np.eye(Ne[0]), "fro") ** 2
        n2 = jnp.linalg.norm((jnp.swapaxes(Z[1].conj(), -1, -2) @ Z[1]) - np.eye(Ne[1]), "fro") ** 2
        return beta / 4 * (n1 + n2)

  else:

    def shift(X):
      return 0.0

    rdmFunc_E = rdmFunc

  custom_energy_term = kwargs.get("custom_energy_term", None)
  if custom_energy_term is not None:
    print("Using passed custom energy term")

  def energyCall(state):
    # TODO spit into orbitals then rdm
    rdms = rdmFunc_E(state)
    # reshape if GHF
    if spin_symm == SlaterType.GHF:
      rdms = (rdms[:N, :N], rdms[N:, N:], rdms[:N, N:], rdms[N:, :N])

    EK = hopCall(rdms)
    EU = UCall(rdms)
    EU1 = U1call(rdms)
    EU2 = U2call(rdms)
    EJ = Jcall(rdms)
    Eheis = Jheis_call(rdms)

    if debug_mode:
      jax.debug.print(
        "energyCall: Etotal={Etotal} with EK={EK} EU={EU} EU1={EU1}"
        " EU2={EU2} EJ={EJ} Eheisenber={Eheis}",
        Etotal=EK + EU + EU1 + EU2 + EJ,
        EK=EK,
        EU=EU,
        EU1=EU1,
        EU2=EU2,
        EJ=EJ,
        Eheis=Eheis,
      )
    if custom_energy_term is not None:
      E_extra = custom_energy_term(state, rdms)
    else:
      E_extra = 0.0
    return jnp.real(EK + EU + EU1 + EU2 + EJ + Eheis + E_extra) + shift(state)

  return jax.jit(energyCall), jax.jit(rdmFunc), orbFunc


def lattice_hf(
  hamiltonian: AutoHFHamiltonian,
  settings=None,
  initial_guess=None,
  start_from=None,
  inputfile=None,
  custom_energy_term=None,
  suppress_logo=False,
  **kwargs,
):
  """
  Run Hartree-Fock on lattice Model Hamiltonians
  """

  banner = """
    -----------------------
    - ╔═╗ ------- ╖ ╓╔═╕ --
    - ╠═╣ ╖╖╒╦╕╔╗ ╠═╣╠╕ ---
    - ╜ ╙ ╚╝ ╜ ╚╝ ╜ ╙╨ ----
    -----------------------
    """
  if not suppress_logo:
    print(banner)

  if hamiltonian is None:
    raise ValueError(
      "Hartree-Fock requies a Hamiltonian to run. "
      "Please provide a Hamiltonian instance as an argument\
            i.e. `lattice_hf(hamiltonian=hamiltonian)`"
    )

  if settings is None:
    infile_dict = read_input_params(inputfile)
    settings = infile_dict["autohf"] | infile_dict["misc_params"]
    settings["inputfile"] = inputfile

  args = process_args(settings, not suppress_logo)

  eta = float(args.eta)
  numOptSteps = int(args.numSteps)
  state0_scale = float(args.state0_scale)

  if not args.gpu:
    print("adding flags")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # just in case

  print(jax.local_devices())
  print("default dtype:", jnp.zeros(1).dtype)

  if args.plot:
    import matplotlib.pyplot as plt

  nelec = settings["nelec"]  # TODO: this is in an awk place
  force_complex = args.force_complex

  print("nelec:", nelec)

  Nbands = hamiltonian.Nbands
  N = hamiltonian.N
  Nx, Ny = hamiltonian.Nx, hamiltonian.Ny

  rng = np.random.default_rng(args.seed)
  rep_type = get_ansatz(args.ansatz)

  numTrials = int(args.numTrials)

  #### MAGIC ZONE ###########
  if args.noncollinear:
    spin_symm = SlaterType.NONCOLLINEAR
  else:
    spin_symm = SlaterType.COLLINEAR

  if start_from is not None:
    state0, state0_ref, phi0 = loadStates(start_from, args, nelec)
  elif args.random_initial_phi0 and initial_guess is None:
    print("Using a random initial guess")
    phi0 = construct_random_Phi(
      rng, N, nelec, spin_symm=spin_symm, delta=getattr(args, "delta", 0.001)
    )
  elif initial_guess is None:
    print("Using a free electron initial guess")
    phi0 = constructPhi0(
      hamiltonian.get_one_body(),
      N,
      nelec,
      spin_symm=spin_symm,
    )
    # TODO: consider if this is a good or a bad idea
    if phi0 is None:
      phi0 = construct_random_Phi(rng, N, nelec, spin_symm=spin_symm)
  else:
    phi0 = convertInitialSlaterDet(initial_guess, nelec, spin_symm=spin_symm)

  # TODO: clean up: put Hamiltonian terms in a dictionary and unpack below!
  energy, makeRDMs, state2Orbitals = makeEnergyCall(
    hamiltonian,
    spin_symm,
    # these are all kwargs now
    rep_type=rep_type,
    debug_mode=args.verbose,
    ref_state=phi0,
    noncollinear=args.noncollinear,
    Ne=nelec,
    riemannian=args.riemannian,
    custom_energy_term=custom_energy_term,
    **kwargs,
  )
  loss = jax.jit(jax.value_and_grad(energy))
  loss_v = jax.jit(jax.vmap(loss))
  #### END MAGIC ZONE ###########

  if start_from is not None:
    pass  # taken care of above
  elif rep_type == RepOptions.SLATERDET_ROT:
    print("Running in Rotation Mode")
    if force_complex:
      raise NotImplementedError("complex rotation not implemented yet")
    if not args.noncollinear:
      num_params = N * (N + 1) // 2
      nonz = get_nonz(N)
      state0 = np.zeros((numTrials, 2, num_params))
      state0[:, 0, nonz] = rng.normal(scale=state0_scale, size=(numTrials, len(nonz)))
      state0[:, 1, nonz] = rng.normal(scale=state0_scale, size=(numTrials, len(nonz)))
    else:
      num_params = (2 * N) * (2 * N + 1) // 2
      nonz = get_nonz(2 * N)
      state0 = np.zeros((numTrials, 1, num_params))
      state0[:, 0, nonz] = rng.normal(scale=state0_scale, size=(numTrials, len(nonz)))
    state0_ref = state0[0] * 0.0
  elif rep_type == RepOptions.SLATERDET_ROTC:
    print("Running in Hermitian (Complex) Rotation Mode")
    if force_complex:
      print("Requested force_complex, but already complex!")
    if not args.noncollinear:
      num_params = N * (N + 1) // 2
      nonz = get_nonz_herm(N)
      state0 = np.zeros((numTrials, 2, num_params))
      state0[:, 0, nonz] = rng.normal(scale=state0_scale, size=(numTrials, len(nonz)))
      state0[:, 1, nonz] = rng.normal(scale=state0_scale, size=(numTrials, len(nonz)))
    else:
      num_params = (2 * N) * (2 * N + 1) // 2
      nonz = get_nonz_herm(2 * N)
      state0 = np.zeros((numTrials, 1, num_params))
      state0[:, 0, nonz] = rng.normal(scale=state0_scale, size=(numTrials, len(nonz)))
    state0_ref = state0[0] * 0.0
  elif rep_type == RepOptions.SLATERDET:
    print("Running in Slater Determinant Mode")
    # TODO: take care of padding ala afqmctools
    if isinstance(phi0, tuple) or isinstance(phi0, list):
      phi0 = np.stack(phi0)
    state0 = phi0.copy()
    state0_ref = state0.copy()
    state0 = state0 + rng.normal(scale=state0_scale, size=(numTrials, *state0.shape))
    if force_complex:
      print("Forcing complex orbitals")
      state0 += 1j * rng.normal(scale=state0_scale / 100, size=(numTrials, *state0.shape))
  elif rep_type == RepOptions.SLATERDETC:
    print("Running in Complex Slater Determinant Mode")
    # TODO: take care of padding ala afqmctools
    if isinstance(phi0, tuple) or isinstance(phi0, list):
      phi0 = np.stack(phi0)
    state0 = jnp.stack([phi0.real, phi0.imag])
    state0_ref = state0.copy()
    state0 = state0 + rng.normal(scale=state0_scale, size=(numTrials, *state0.shape))
    if force_complex:
      print("Forcing complex orbitals")
      state0 += 1j * rng.normal(scale=state0_scale / 100, size=(numTrials, *state0.shape))
  elif rep_type == RepOptions.DIAG:
    print("Running in Diagonalization Mode")
    if not args.noncollinear:
      state0 = rng.normal(scale=state0_scale / 10, size=((numTrials, 2, N, N)))
    else:
      state0 = rng.normal(scale=state0_scale / 10, size=((numTrials, 2 * N, 2 * N)))
    state0_ref = state0[0] * 0.0
  elif rep_type == RepOptions.CUSTOM:
    state0 = kwargs["state0"]
    state0_ref = kwargs.get("state0_ref", state0 * 0.0)

  E_ref = energy(state0_ref)
  print(f"Reference HF Energy = {E_ref}")

  if numOptSteps < 0:  # skip optimization
    state = np.stack([state0_ref])  # how to build phi0 in the ansatz
    E_final = E_ref
    Es = [E_final]
    output_data = saveStates(
      state, state2Orbitals, np.array(Es), phi0, args, nelec, args.output, args.dumpTrials
    )

  else:
    if args.opt_method == "grad":
      # Optimizer setup
      _optax_settings = {"scale_by": optax.scale_by_belief}

      optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        # optax.scale_by_adam(),
        # optax.scale_by_belief(),
        _optax_settings["scale_by"](),
        # Scale updates by -1 since optax.apply_updates is additive
        # and we want to descend on the loss.
        optax.scale(-eta),
      )

      state = state0.copy()
      opt_state = jax.vmap(optimizer.init)(state)
      Es = []
      bestState = 0
      if args.plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        (line1,) = ax[0].plot(np.arange(len(Es)), Es, "o-")
        ax[0].axhline(y=E_ref, ls=":", c="k", alpha=0.5)
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("E")

        rdms = makeRDMs(state[bestState])
        # TODO: make a function that returns nup,ndn
        # and accounts for collinear vs. noncollinear
        nup, ndn = rdms[0].diagonal(), rdms[1].diagonal()
        # we are plotting the "bands" in the y-direction!
        line2, *_ = ax[1].plot(np.arange(Nx), (nup - ndn).reshape(Nx, Ny * Nbands) / 2)
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("<Sz>")
      for _ in range(numOptSteps):
        # TODO: this fails on complex-valued t-matrix elements
        allE, derivs = loss_v(state)
        bestState = np.argmin(allE)

        Es.append(allE[bestState])
        print(Es[-1], allE)  # TODO: print more info!!
        # find out density
        # TODO, loss should output
        if args.plot:
          rdms = makeRDMs(state[bestState])
          nup, ndn = rdms[0].diagonal(), rdms[1].diagonal()

        # update params
        # TODO, clean up vmap
        updates, opt_state = jax.vmap(optimizer.update)(derivs, opt_state, state)
        state = jax.vmap(optax.apply_updates)(state, updates)

        if args.plot:
          ax[0].clear()
          ax[1].clear()
          ax[0].plot(Es, "o-")
          ax[1].plot((nup - ndn).reshape(Nx, Ny * Nbands) / 2)

          ax[0].axhline(y=E_ref, ls=":", c="k", alpha=0.5)
          ax[0].set_xlabel("Step")
          ax[0].set_ylabel("E")
          ax[1].set_xlabel("Step")
          ax[1].set_ylabel("<Sz>")
          fig.canvas.draw()
          fig.canvas.flush_events()
          plt.draw()

      if args.plot:
        plt.show()

      output_data = saveStates(
        state, state2Orbitals, allE, phi0, args, nelec, args.output, args.dumpTrials
      )
      state = state[bestState]
    elif args.opt_method == "lbfgs":
      # TODO: have this be the first run of a more useful function
      print(f"Initial Energy: {loss_v(state0)[0]}", flush=True)
      extraops = kwargs.get("jaxoptargs", {})
      solver = jaxopt.LBFGS(fun=loss, value_and_grad=True, maxiter=numOptSteps, **extraops)
      run_v = jax.vmap(solver.run)
      res = run_v(state0)

      bestState = np.argmin(res.state.value)
      state = res.params[bestState]
      print(
        "# iterations:", res.state.iter_num, "frac of possible:", res.state.iter_num / numOptSteps
      )
      print("Final energy:", res.state.value[bestState], res.state.value, flush=True)

      if res.state.iter_num[bestState] == numOptSteps:
        warnings.warn(
          f"The best variational state did not converge in {numOptSteps} steps. "
          "Inspect result carefully!",
          RuntimeWarning,
          stacklevel=2,
        )

      output_data = saveStates(
        res.params, state2Orbitals, res.state.value, phi0, args, nelec, args.output, args.dumpTrials
      )
      if args.plot:
        rdms = makeRDMs(state)
        nup, ndn = rdms[0].diagonal(), rdms[1].diagonal()
        Es = [energy(state)]

    elif args.opt_method == "basin":
      from scipy.optimize import basinhopping

      if numTrials != 1:
        warnings.warn(
          "Basin hopping does not use repetitions, modify the settings instead!",
          RuntimeWarning,
          stacklevel=2,
        )
      if E_ref.dtype != "float64":
        raise RuntimeError("""Attempted to use basin hopping with a dtype that isnt float64.
                                 Please run with
                                 import jax
                                 jax.config.update("jax_enable_x64", True)
                                 """)

      state0 = state0[0]  # ensure single example

      def makeLossSP(state0, loss):
        @jax.jit
        def loss2(x):
          res = loss(x.reshape(state0.shape))
          return res[0], res[1].flatten()

        return loss2

      lossSP = makeLossSP(state0, loss)
      minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}

      # TODO: make a verbose option
      def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))

      ret = basinhopping(
        lossSP,
        state0.flatten(),
        minimizer_kwargs=minimizer_kwargs,
        niter=numOptSteps,
        callback=print_fun,
        seed=rng,
        T=10,
        stepsize=0.1,
      )
      print(ret.message, flush=True)
      state = ret.x.reshape(state0.shape)
      output_data = saveStates(
        state, state2Orbitals, ret.fun, phi0, args, nelec, args.output, args.dumpTrials
      )
      if args.plot:
        rdms = makeRDMs(state)
        nup, ndn = rdms[0].diagonal(), rdms[1].diagonal()
        Es = [energy(state)]
    else:
      msg = f"{args.opt_method} is not implemented! try grad, lbfgs, or basin"
      raise NotImplementedError(msg)

    if args.plot:
      fig1, ax1 = plt.subplots(1, 1)

      rho_spin = (nup - ndn).reshape(Nx, Ny * Nbands).T / 2
      rho_charge = (nup + ndn).reshape(Nx, Ny * Nbands).T

      # TODO: we need to decide how to plot densities with multiple bands
      mat1 = ax1.matshow(rho_spin, vmin=-0.5, vmax=0.5, cmap="seismic")  # ,fignum=2)
      ax1.set_title(f"{Nx}x{Ny} nbands={Nbands} n={sum(nelec) / N:0.4f} E={Es[-1]:0.4f} Sz")
      fig1.colorbar(mat1)

      fig2, ax2 = plt.subplots(1, 1)
      mat2 = ax2.matshow(rho_charge, cmap="plasma")
      ax2.set_title(f"{Nx}x{Ny} nbands={Nbands} n={sum(nelec) / N:0.4f} E={Es[-1]:0.4f} Charge")
      fig2.colorbar(mat2)

      plt.show()

    # We could potentially cache this from the optimization output
    # but I like the idea that this is a double check that the state is sane
    E_final = energy(state)
    print(f"Variational energy of final HF state: {E_final}")
    if E_final > E_ref:
      dE = E_final - E_ref
      warnings.warn(
        f"Final energy ({E_final}) is higher than reference energy ({E_ref}) by {dE}",
        RuntimeWarning,
        stacklevel=1,
      )

  if args.measure_spin:
    rdms = makeRDMs(state)
    if args.noncollinear:
      spin_symm = "noncollinear"
      rdm_list = [rdms]
    else:
      spin_symm = "collinear"
      rdm_list = rdms
    print(
      "<Sz>",
      spin_z(rdm=rdm_list, spin_symm=spin_symm),
      end=" ",
    )
    print(
      "<S^2>",
      spin_squared(rdm=rdm_list, spin_symm=spin_symm),
    )

  # data_functions can't be pickled, so split off but useful
  data_functions = {
    "makeRDMs": makeRDMs,
    "state2Orbitals": state2Orbitals,
    "energy_func": energy,
    "loss": loss,
    "loss_v": loss_v,
  }
  return output_data, data_functions
