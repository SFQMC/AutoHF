import importlib
import logging as log

# optionally import afqmctools
if importlib.util.find_spec("afqmctools") is not None:
  HAS_AFQMCTOOLS = True
  from afqmctools.hamiltonian.model.ham_class import SpinSymm, Hamiltonian
else:
  HAS_AFQMCTOOLS = False

  class Hamiltonian:
    def __init__(self):
      pass

  class SpinSymm:
    COLLINEAR = 0
    NONCOLLINEAR = 1
    CLOSED = 2


def fail_without_afqmctools(func):
  def wrapper(*args, **kwargs):
    if HAS_AFQMCTOOLS:
      return func(*args, **kwargs)
    else:
      raise RuntimeError(
        f"called function {func.__name__} which requires afqmctools. "
        "Install afqmctools to use this function."
      )

  return wrapper


def skip_without_afqmctools(func):
  if not HAS_AFQMCTOOLS:
    log.warning(f"Skipping {func.__name__} because afqmctools is not installed.")
    return lambda *args, **kwargs: None
  return func


if HAS_AFQMCTOOLS:
  from afqmctools.wavefunction.model import write_wfn
  from afqmctools.observables.spin import spin_squared, spin_z
else:
  write_wfn = skip_without_afqmctools(lambda *args, **kwargs: None)
  spin_squared = skip_without_afqmctools(lambda *args, **kwargs: None)
  spin_z = skip_without_afqmctools(lambda *args, **kwargs: None)
