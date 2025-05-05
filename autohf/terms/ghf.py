import numpy as np
import jax.numpy as jnp

"""
Library of wick contractions for UHF wave functions
This implies the RDMs (one body Green's function)
are split into up-up, up-dn, dn-up, dn-dn

"""

known_terms = ["T", "U", "U1", "U2", "J", "Jheis"]


# Hubbard U n_{i\uparrow} n_{j\downarrow}
def ghf_U(rdms, sites):
  rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
  dup, ddn = rdm_uu.diagonal(), rdm_dd.diagonal()
  dud, ddu = rdm_ud.diagonal(), rdm_du.diagonal()
  return dup[sites] @ ddn[sites] - dud[sites] @ ddu[sites]


# U1 mixed spin or \sum_\sigma n_{i \sigma}n_{j \sigma'}
def ghf_U1(rdms, ii, jj, ij, ji):
  rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
  E = rdm_uu[ii[0], ii[1]] @ rdm_dd[jj[0], jj[1]] + rdm_uu[jj[0], jj[1]] @ rdm_dd[ii[0], ii[1]]
  # spin terms
  E -= rdm_ud[ij[0], ij[1]] @ rdm_du[ji[0], ji[1]] + rdm_du[ij[0], ij[1]] @ rdm_ud[ji[0], ji[1]]
  return E


# U2 same spin or \sum_\sigma n{i \sigma} n_{j \sigma}
# split into ii,jj for (i,i),(j,j) and ij,ji for (i,j),(j,i)
def ghf_U2(rdms, ii, jj, ij, ji):
  rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
  E = rdm_uu[ii[0], ii[1]] @ rdm_uu[jj[0], jj[1]] + rdm_dd[ii[0], ii[1]] @ rdm_dd[jj[0], jj[1]]
  # off-diag in spatial
  E -= rdm_uu[ij[0], ij[1]] @ rdm_uu[ji[0], ji[1]] + rdm_dd[ij[0], ij[1]] @ rdm_dd[ji[0], ji[1]]
  return E


# J term (see HK model)
# split into ij,ji
def ghf_J(rdms, ii, jj, ij, ji):
  rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
  E = (
    rdm_uu[ij[0], ij[1]] @ rdm_dd[ji[0], ji[1]]
    + rdm_uu[ij[0], ij[1]] @ rdm_dd[ij[0], ij[1]]
    + rdm_uu[ji[0], ji[1]] @ rdm_dd[ij[0], ij[1]]
    + rdm_uu[ji[0], ji[1]] @ rdm_dd[ji[0], ji[1]]
  )
  # spin off-diagonal terms
  E -= (
    rdm_ud[ii[0], ii[1]] @ rdm_du[jj[0], jj[1]]
    + rdm_ud[ij[0], ij[1]] @ rdm_du[ij[0], ij[1]]
    + rdm_ud[jj[0], jj[1]] @ rdm_du[ii[0], ii[1]]
    + rdm_ud[ji[0], ji[1]] @ rdm_du[ji[0], ji[1]]
  )
  return E


def ghf_Jheis(rdms, ii, jj, ij, ji):
  rdm_uu, rdm_dd, rdm_ud, rdm_du = rdms
  E = rdm_uu[ii[0], ii[1]] @ rdm_dd[jj[0], jj[1]] + rdm_uu[ji[0], ji[1]] @ rdm_dd[ij[0], ij[1]]
  # spin off-diagonal terms
  E -= rdm_ud[ij[0], ij[1]] @ rdm_du[ji[0], ji[1]] + rdm_ud[jj[0], jj[1]] @ rdm_du[ii[0], ii[1]]
  return E
