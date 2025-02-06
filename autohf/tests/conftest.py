import pytest
import numpy as np


from autohf.hamiltonian import AutoHFHamiltonian

@pytest.fixture(scope="session")
def hubbard_4x4():
    """
    A simple 4x4 Hubbard model with periodic boundary conditions.
    """
    t = -1.0
    U = 4.0
    L = 4

    #TODO: dump these from afqmctools for test purposes
    T = np.zeros((L,L))
    H_U = np.full((L), U)

    # TODO: make this more smooth
    hamiltonian = AutoHFHamiltonian()
    hamiltonian.set("T",T)
    hamiltonian.set("U",H_U)
    return hamiltonian

 