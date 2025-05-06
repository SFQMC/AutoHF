import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def sdToRDM(sds):
  r"""takes (2,N,Ne) orbitals
  returns <\psi|c^dag c | \psi>/<\psi|\psi>
  """
  # (2,N,Ne)
  # because we could have different number of particles, let's not batch
  # but TODO: make batched later
  phi_up, phi_dn = sds
  rdm_up = phi_up.conj() @ jnp.linalg.solve(phi_up.T @ phi_up.conj(), phi_up.T)
  rdm_dn = phi_dn.conj() @ jnp.linalg.solve(phi_dn.T @ phi_dn.conj(), phi_dn.T)
  return rdm_up, rdm_dn


@jax.jit
def sdToRDM_noncollinear(sds):
  r"""takes (1,N,Ne_total) orbitals
  returns <\psi|c^dag c | \psi>/<\psi|\psi>
  """
  # (1,N,Ne)
  phi = sds[0]
  rdm = phi.conj() @ jnp.linalg.solve(phi.T @ phi.conj(), phi.T)
  return rdm  # tuple for consistency with the collinear case!
