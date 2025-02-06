import pytest
import numpy as np

from afqmctools.hamiltonian.model.director import HamiltonianDirector
from autohf.solver import lattice_hf

class TestAutoHFSolver:

    @pytest.mark.dev
    @pytest.mark.parametrize( # TODO: add more cases
        "t,J_heisenberg,L,boundaries,Nes,seed,expected_energy",
        [
            (0,1.0,(4,1),("open","open"),(4,0),42,0.75)
        ]
    )
    def test_heisenberg_collinear(self,t,J_heisenberg,L,boundaries,
                                  Nes,seed,expected_energy,tmp_path):
        """
        Tests that the Hamiltonian is properly converted to
        autoHF's format.

        This test is indirect. It only checks that each of the
        unique values in the input parameters t,U,U1,U2,J appear
        *somewhere* in the appropriate "term".
        """
        settings = {
            "hamiltonian" : dict(
                t = t,
                J_heisenberg = J_heisenberg
            ),
            "lattice" : dict(
                L1 = L[0],
                L2 = L[1],
                boundary1 = boundaries[0],
                boundary2 = boundaries[1]
                )
        }

        hamiltonian_director = HamiltonianDirector(source=settings)
        hamiltonian = hamiltonian_director.build()
        lattice = hamiltonian_director.builder.lattice

        hf_settings = dict(
            ansatz = 'SD_ROT',
            numSteps = 100,
            output = tmp_path/"afqmc.h5",
            opt_method="grad",
            nelec = Nes,
            numTrials = 10,
            seed = seed,
            noncollinear = False
        )
        E_hf = lattice_hf(
            hamiltonian=hamiltonian,
            lattice=lattice,
            settings=hf_settings
        )["E_final"]

        assert np.isclose(E_hf,expected_energy, atol=1e-4)
