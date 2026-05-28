from .solver import solve_hf
from .hamiltonian import AutoHFHamiltonian

import warnings
def lattice_hf(*args, **kwargs):
    warnings.warn(
        "autohf.lattice_hf() is deprecated and will be removed in a future version. "
        "Please use autohf.solve_hf() instead",
        category=DeprecationWarning,
        stacklevel=2
    )
    return solve_hf(*args, **kwargs)
