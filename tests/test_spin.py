import numpy as np
import pytest
from scipy.linalg import block_diag

from autohf.spin import spin_squared, spin_z


def test_polarized_collinear():
  """Three up electrons: S = Sz = 3/2."""
  rdm = block_diag(np.diag([1.0, 1.0, 1.0, 0.0]), np.zeros((4, 4)))
  assert spin_z(rdm) == pytest.approx(1.5)
  assert spin_squared(rdm) == pytest.approx(3.75)


def test_singlet_collinear():
  """Closed shell: S = Sz = 0."""
  rdm = block_diag(np.diag([1.0, 1.0, 0.0, 0.0]), np.diag([1.0, 1.0, 0.0, 0.0]))
  assert spin_z(rdm) == pytest.approx(0.0)
  assert spin_squared(rdm) == pytest.approx(0.0)


def test_polarized_along_x_noncollinear():
  """Three electrons polarized along x: S = 3/2 but Sz = 0."""
  M = 4
  phi = np.zeros((2 * M, 3))
  for k in range(3):
    phi[k, k] = phi[M + k, k] = 1 / np.sqrt(2)
  rdm = phi @ phi.T
  assert spin_z(rdm) == pytest.approx(0.0)
  assert spin_squared(rdm) == pytest.approx(3.75)
