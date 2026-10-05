"""
Legacy interface to afqmctools, superseded by :mod:`autohf.interfaces.safiretools`.
"""

import importlib

from .safiretools import _get_T, _get_U, _get_J, _get_heisenberg, _get_terms

# optionally import afqmctools
if importlib.util.find_spec("afqmctools") is not None:
  HAS_AFQMCTOOLS = True
  from afqmctools.hamiltonian.model.ham_class import SpinSymm, Hamiltonian
else:
  HAS_AFQMCTOOLS = False

  class Hamiltonian:
    def __init__(self):
      pass
