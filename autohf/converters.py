from autohf.interfaces.afqmctools import fail_without_afqmctools, Hamiltonian, SpinSymm

import numpy as np
from .options import RepOptions,SlaterType
import jax.numpy as jnp

@fail_without_afqmctools
def autoHF_2_afqmc_wfn(state,state2Orbitals,N,Ne,
    rep=RepOptions.SLATERDET_ROT,
    spin_symm=SlaterType.COLLINEAR):

    orbs = state2Orbitals(state)

    if spin_symm == SlaterType.CLOSED:
        raise NotImplementedError("Not implemented for Closed Walkers")
    elif spin_symm == SlaterType.COLLINEAR:
        return autoHF_2_afqmc_wfn_collinear(orbs,N,Ne,rep=rep)
    elif spin_symm == SlaterType.NONCOLLINEAR:
        return autoHF_2_afqmc_wfn_noncollinear(orbs,N,Ne,rep=rep)
    else:
        raise ValueError("Unknown spin symmetry requested")

@fail_without_afqmctools
def autoHF_2_afqmc_wfn_noncollinear(orbs,N,nelec,
    rep=RepOptions.SLATERDET_ROT):
    '''
    '''
    state_afqmc = orbs[0][:,:nelec[0]+nelec[1]]
    return state_afqmc

@fail_without_afqmctools
def autoHF_2_afqmc_wfn_collinear(orbs,N,nelec,rep=RepOptions.SLATERDET_ROT):
    '''
    '''

    state_afqmc = np.zeros((N,sum(nelec)),dtype=orbs[0].dtype)
    state_afqmc[:,:nelec[0]] = orbs[0][:,:nelec[0]]
    state_afqmc[:,nelec[0]:] = orbs[1][:,:nelec[1]]

    return state_afqmc

@fail_without_afqmctools
def _get_T(hamiltonian:Hamiltonian):
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
        Tup, Tdown = T[:N,:],T[N:,:]
        T = [Tup,Tdown]
    else:
        T = [T[:N*2,:]]

    return T

@fail_without_afqmctools
def _get_U(hamiltonian:Hamiltonian):
    """
    get U, U1, and U2 from hamiltonian.
    """
    if 'Uij' not in hamiltonian.keys():
        U = U1 = U2 = None
    else:
        fullU = hamiltonian.get_U().toarray()
        M = hamiltonian.nsites*hamiltonian.nbands
        U = np.diag(fullU[:M,:M].diagonal())
        U1 = fullU[:M,:M].copy() - U
        if np.allclose(U1,np.zeros_like(U1)):
            U1 = None

        if fullU.shape[0] == 2*M:
            U2 = fullU[M:,:M]
        else:
            U2 = None

    return U,U1,U2

@fail_without_afqmctools
def _get_J(hamiltonian:Hamiltonian):
    if 'Jij' in hamiltonian.keys():
        return hamiltonian.get_J().toarray()
    else:
        return None

@fail_without_afqmctools
def _get_heisenberg(hamiltonian:Hamiltonian):
    if 'J_heisenberg' in hamiltonian.keys():
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

def convertInitialSlaterDet(initial_guess,Nes,spin_symm=SlaterType.COLLINEAR):
    """
    map from an initial guess in the afqmctools format to the autoHF format.

    Note: generates the initial Slater determint in RepOptions.SLATERDET mode
            or the reference Slater determinant in RepOptions.SLATERDET_ROT mode

    For Collinear Slater determinants:
        afqmctools will provide Phi[nmo,Nes[0]+Nes[1]] which should be converted to
        jnp.stack(Phi[:,:Nes[0]], Phi[nmo][:,Nes[0]:])
    For Noncollinear Slater determinants:
        simply converts to a jnp array with appropriate dimensions.
    """

    if spin_symm == SlaterType.COLLINEAR:
        return jnp.stack([initial_guess[:,:Nes[0]],initial_guess[:,Nes[0]:]])
    elif spin_symm == SlaterType.NONCOLLINEAR:
        return jnp.stack([initial_guess[:,:Nes[0]+Nes[1]]])
    else:
        raise ValueError("Could not convert the given initial guess using SlaterType ", spin_symm)

