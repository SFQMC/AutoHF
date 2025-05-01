import pytest
import numpy as np

from autohf.hamiltonian import AutoHFHamiltonian

class TestAutoHFHamiltonian:
    """
    Tests for the AutoHFHamiltonian class.
    """

    # 3x3 square lattice with periodic boundary conditions
    T = [[ 0., -1., -1., -1.,  0.,  0., -1.,  0.,  0.],
        [-1.,  0., -1.,  0., -1.,  0.,  0., -1.,  0.],
        [-1., -1.,  0.,  0.,  0., -1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0., -1., -1., -1.,  0.,  0.],
        [ 0., -1.,  0., -1.,  0., -1.,  0., -1.,  0.],
        [ 0.,  0., -1., -1., -1.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0., -1.,  0.,  0.,  0., -1., -1.],
        [ 0., -1.,  0.,  0., -1.,  0., -1.,  0., -1.],
        [ 0.,  0., -1.,  0.,  0., -1., -1., -1.,  0.],
        [ 0., -1., -1., -1.,  0.,  0., -1.,  0.,  0.],
        [-1.,  0., -1.,  0., -1.,  0.,  0., -1.,  0.],
        [-1., -1.,  0.,  0.,  0., -1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0., -1., -1., -1.,  0.,  0.],
        [ 0., -1.,  0., -1.,  0., -1.,  0., -1.,  0.],
        [ 0.,  0., -1., -1., -1.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0., -1.,  0.,  0.,  0., -1., -1.],
        [ 0., -1.,  0.,  0., -1.,  0., -1.,  0., -1.],
        [ 0.,  0., -1.,  0.,  0., -1., -1., -1.,  0.]]

    T_auto_HF = ([np.array([-1.]), np.array([-1.])],
                 [[(np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
       5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]),
                    np.array([1, 2, 3, 6, 0, 2, 4, 7, 0, 1, 5, 8, 0, 4, 5, 6, 1, 3, 5, 7, 2, 3,
       4, 8, 0, 3, 7, 8, 1, 4, 6, 8, 2, 5, 6, 7]))],
                  [(np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
       5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]),
                    np.array([1, 2, 3, 6, 0, 2, 4, 7, 0, 1, 5, 8, 0, 4, 5, 6, 1, 3, 5, 7, 2, 3,
       4, 8, 0, 3, 7, 8, 1, 4, 6, 8, 2, 5, 6, 7]))]])

    U = [[4., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 4., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 4., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 4., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 4., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 4., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 4., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 4., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 4.]]

    U_auto_HF = (np.array([4.]), [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])])

    @pytest.fixture(scope="class")
    def dummy_hamiltonian(self):
        """
        Returns a dummy Hamiltonian for testing purposes.
        """
        return AutoHFHamiltonian()

    @pytest.mark.parametrize(("term","value","expected"),[("T",T,T_auto_HF),("U",U,U_auto_HF)])
    def test_terms(self,term,value,expected,dummy_hamiltonian):
        """
        Tests that the set method works as expected.

        Q: how will we check that the output is correct?

        I expect both a term to be added to the Hamiltonian, AND for the
        term to actually be correct. The former can be checkde with no
        additional information, but the latter will use to just save the correct
        term.
        """
        # TODO: need to convert from the value to the correct format.
        dummy_hamiltonian.set(term,value)
        assert dummy_hamiltonian.get(term) == expected

