import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def sd_to_rdm(sds):
  r"""takes (2,N,Ne) orbitals
  returns <\psi|c^dag c | \psi>/<\psi|\psi>
  """
  # (2,N,Ne)
  # because we could have different number of particles, let's not batch
  phi_up, phi_dn = sds
  rdm_up = phi_up.conj() @ jnp.linalg.solve(phi_up.T @ phi_up.conj(), phi_up.T)
  rdm_dn = phi_dn.conj() @ jnp.linalg.solve(phi_dn.T @ phi_dn.conj(), phi_dn.T)
  return rdm_up, rdm_dn


def create_polarized_func(phi0, f):
  r"""takes collinear phi0 and constructs sliced input to function f"""
  N, Nup = phi0[0].shape
  _, Ndn = phi0[1].shape

  def rdm_func(state):
    return f((state[0][0:N, 0:Nup], state[1][0:N, 0:Ndn]))

  return rdm_func


@jax.jit
def sd_to_rdm_noncollinear(sds):
  r"""takes (1,N,Ne_total) orbitals
  returns <\psi|c^dag c | \psi>/<\psi|\psi>
  """
  # (1,N,Ne)
  phi = sds[0]
  rdm = phi.conj() @ jnp.linalg.solve(phi.T @ phi.conj(), phi.T)
  return rdm  # tuple for consistency with the collinear case!


def uhf_to_ghf(x):
  """Converts [o_up, o_dn] UHF like input to a
  [ o_up 0    ]
  [  0   o_dn ]
  block diagonal GHF form
  """
  if isinstance(x, np.ndarray):
    return np.block([[x[0], np.zeros_like(x[1])], [np.zeros_like(x[0]), x[1]]])
  return jnp.block([[x[0], jnp.zeros_like(x[1])], [jnp.zeros_like(x[0]), x[1]]])
