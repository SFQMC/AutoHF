import pytest
import numpy as np

from autohf.hamiltonian import AutoHFHamiltonian


class TestAutoHFHamiltonianShort:
  """
  Tests for the AutoHFHamiltonian class.
  """

  # 3x3 square lattice with periodic boundary conditions
  T = np.array(
    [
      [0.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
      [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
      [-1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
      [-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0],
      [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0],
      [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0],
      [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0],
      [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0],
    ]
  )

  T_auto_HF = ([np.array([-1.]), np.array([-1.])],
                 [[(np.array([i//4 for i in range(4*9)]),
                    np.array([1, 2, 3, 6, 0, 2, 4, 7,
                              0, 1, 5, 8, 0, 4, 5, 6,
                              1, 3, 5, 7, 2, 3,4, 8,
                              0, 3, 7, 8, 1, 4, 6, 8, 2, 5, 6, 7]))],
                  [(np.array([i//4 for i in range(4*9)]),
                    np.array([1, 2, 3, 6, 0, 2, 4, 7,
                              0, 1, 5, 8, 0, 4, 5, 6,
                              1, 3, 5, 7, 2, 3, 4, 8,
                              0, 3, 7, 8, 1, 4, 6, 8,
                              2, 5, 6, 7]))]]
                 )  # fmt: skip

  U = [
    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
  ]

  U_auto_HF = (np.array([4.0]), [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])])

  def test_add_terms(self):
    T = self.T
    U = self.U
    dummy_hamiltonian = AutoHFHamiltonian(T=(T, T), U=U)
    for term, value, expected in [("T", T, self.T_auto_HF), ("U", U, self.U_auto_HF)]:
      unique_vals, sites = dummy_hamiltonian.get(term)
      assert np.allclose(expected[0], unique_vals)
      assert np.allclose(expected[1], sites)

  @pytest.mark.parametrize(("Tmat"), [[T], [T, T], (T,), (T, T), np.stack([T, T])])
  def test_hopping_types(self, Tmat):
    term = "T"
    expected = self.T_auto_HF

    H = AutoHFHamiltonian(**{term: Tmat})

    unique_vals, sites = H.get(term)
    assert np.allclose(expected[0], unique_vals)
    assert np.allclose(expected[1], sites)

  def test_hopping_pinned(self):
    T = self.T
    term = "T"

    # pass list
    H = AutoHFHamiltonian(T=[T + 0.1 * np.eye(9), T - 0.1 * np.eye(9)])

    unique_vals, sites = H.get(term)
    assert len(unique_vals) == 2
    assert len(unique_vals[0]) == 2

  @pytest.mark.parametrize(
    ("U"),
    [
      4,
      4.0,
      np.diag(U),
      U,
    ],
  )
  def test_U_types(self, U):
    term = "U"
    expected = self.U_auto_HF

    H = AutoHFHamiltonian(N=len(self.U),T=(self.T,self.T),**{term: U})

    unique_vals, sites = H.get(term)
    assert np.allclose(expected[0], unique_vals)
    assert np.allclose(expected[1], sites)


class TestAutoHFHamiltonianVerbose:
  """
  Tests for the AutoHFHamiltonian class.
  """

  # 3x3 square lattice with periodic boundary conditions
  T = np.array(
    [
      [0.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
      [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
      [-1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
      [-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0],
      [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0],
      [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0],
      [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0],
      [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0],
    ]
  )

  T_auto_HF = ([np.array([-1.]), np.array([-1.])],
                 [[(np.array([i//4 for i in range(4*9)]),
                    np.array([1, 2, 3, 6, 0, 2, 4, 7,
                              0, 1, 5, 8, 0, 4, 5, 6,
                              1, 3, 5, 7, 2, 3,4, 8,
                              0, 3, 7, 8, 1, 4, 6, 8, 2, 5, 6, 7]))],
                  [(np.array([i//4 for i in range(4*9)]),
                    np.array([1, 2, 3, 6, 0, 2, 4, 7,
                              0, 1, 5, 8, 0, 4, 5, 6,
                              1, 3, 5, 7, 2, 3, 4, 8,
                              0, 3, 7, 8, 1, 4, 6, 8,
                              2, 5, 6, 7]))]]
                 )  # fmt: skip

  U = [
    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
  ]

  U_auto_HF = (np.array([4.0]), [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])])

  @pytest.fixture(scope="class")
  def dummy_hamiltonian(self):
    """
    Returns a dummy Hamiltonian for testing purposes.
    """
    return AutoHFHamiltonian(settings={"L": (3, 3)})

  @pytest.mark.parametrize(
    ("term", "value", "expected"), [("T", (T, T), T_auto_HF), ("U", U, U_auto_HF)]
  )
  def test_add_terms(self, term, value, expected, dummy_hamiltonian):
    """
    Tests that the add_terms method works as expected.

    Q: how will we check that the output is correct?

    I expect both a term to be added to the Hamiltonian, AND for the
    term to actually be correct. The former can be checkde with no
    additional information, but the latter will use to just save the correct
    term.
    """
    dummy_hamiltonian.add_terms(**{term: value})
    unique_vals, sites = dummy_hamiltonian.get(term)
    assert np.allclose(expected[0], unique_vals)
    assert np.allclose(expected[1], sites)

  @pytest.mark.parametrize(("Tmat"), [[T, T], (T, T), np.stack([T, T])])
  def test_hopping_types(self, Tmat):
    term = "T"
    expected = self.T_auto_HF

    H = AutoHFHamiltonian(settings={"L": (3, 3)})
    # Pass direct matrix (up=down)
    H.add_terms(**{term: Tmat})

    unique_vals, sites = H.get(term)
    assert np.allclose(expected[0], unique_vals)
    assert np.allclose(expected[1], sites)

  def test_hopping_pinned(self):
    T = self.T
    term = "T"

    # pass list
    H = AutoHFHamiltonian(settings={"L": (3, 3)})
    H.add_terms(T=[T + 0.1 * np.eye(9), T - 0.1 * np.eye(9)])

    unique_vals, sites = H.get(term)
    assert len(unique_vals) == 2
    assert len(unique_vals[0]) == 2

  @pytest.mark.parametrize(
    ("U"),
    [
      4,
      4.0,
      np.diag(U),
      U,
    ],
  )
  def test_U_types(self, U):
    term = "U"
    expected = self.U_auto_HF

    H = AutoHFHamiltonian(settings={"L": (3, 3)})
    # Pass direct matrix (up=down)
    H.add_terms(**{term: U})

    unique_vals, sites = H.get(term)
    assert np.allclose(expected[0], unique_vals)
    assert np.allclose(expected[1], sites)
