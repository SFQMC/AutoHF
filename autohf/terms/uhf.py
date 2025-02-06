import numpy as np
import jax.numpy as jnp
import jax


#from .options import (
#    SlaterType,
#    RepOptions,
#    processes_args
#)
#from .rotation_tools import (
#    jexpm,
#    get_nonz,
#    get_nonz_herm,
#)
#from .converters import (
#    AutoHFHamiltonian,
#    convertInitialSlaterDet,
#    autoHF_2_afqmc_wfn
#)

known_terms = ["T","U","U1","U2","J","Jheis"]

'''
Library of wick contractions for UHF wave functions
This implies the RDMs (one body Green's function)
are in the form [G_up, G_dn]

'''

# one body hop - c^\dagger_i c_j
# slightly special, assumes only one spin species
def rhf_hop(rdm,hops):
    return jnp.sum(rdm[hops[0], hops[1]])

# Hubbard U n_{i\uparrow} n_{j\downarrow}
def uhf_U(rdms,sites):
  dup,ddn = rdms[0].diagonal(),rdms[1].diagonal()
  return dup[sites]@ddn[sites]

# U1 mixed spin or \sum_\sigma n_{i \sigma}n_{j \sigma'}
def uhf_U1(rdms,ii,jj,ij,ji):
  rho_up,rho_dn = rdms[0],rdms[1]
  return (rho_up[ii[0],ii[1]]@rho_dn[jj[0],jj[1]] \
         + rho_up[jj[0],jj[1]]@rho_dn[ii[0],ii[1]])

# U2 same spin or \sum_\sigma n{i \sigma} n_{j \sigma}
# split into ii,jj for (i,i),(j,j) and ij,ji for (i,j),(j,i)
def uhf_U2(rdms,ii,jj,ij,ji):
  rho_up,rho_dn = rdms[0],rdms[1]
  E = (rho_up[ii[0],ii[1]]@rho_up[jj[0],jj[1]] \
      + rho_dn[ii[0],ii[1]]@rho_dn[jj[0],jj[1]])
  # off-diag in spatial
  E -= (rho_up[ij[0],ij[1]]@rho_up[ji[0],ji[1]] \
      + rho_dn[ij[0],ij[1]]@rho_dn[ji[0],ji[1]])
  return E

# J term (see HK model)
# split into ij,ji
def uhf_J(rdms,ii,jj,ij,ji):
  rho_up,rho_down = rdms[0],rdms[1]
  return (rho_up[ij[0],ij[1]]@rho_down[ji[0],ji[1]]
          + rho_up[ij[0],ij[1]]@rho_down[ij[0],ij[1]]
          + rho_up[ji[0],ji[1]]@rho_down[ij[0],ij[1]]
          + rho_up[ji[0],ji[1]]@rho_down[ji[0],ji[1]])

def uhf_Jheis(rdms,ii,jj,ij,ji):
  rho_up,rho_down = rdms[0],rdms[1]
  return (rho_up[ii[0],ii[1]]@rho_down[jj[0],jj[1]]
         + rho_up[ji[0],ji[1]]@rho_down[ij[0],ij[1]])
