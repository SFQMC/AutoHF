import numpy as np
from .options import RepOptions, SlaterType
import jax.numpy as jnp


def convertInitialSlaterDet(initial_guess, Nes, spin_symm=SlaterType.COLLINEAR):
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
    return (jnp.array(initial_guess[:, : Nes[0]]), jnp.array(initial_guess[:, Nes[0] :]))
  elif spin_symm == SlaterType.NONCOLLINEAR:
    return (jnp.array(initial_guess[:, : Nes[0] + Nes[1]]),)
  else:
    raise ValueError(f"Could not convert the given initial guess using SlaterType {spin_symm}")

