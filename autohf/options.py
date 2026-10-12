import time
import enum
from types import SimpleNamespace
import warnings


class RepOptions(enum.Enum):
  NONE = enum.auto()
  SLATERDET = enum.auto()
  SLATERDETC = enum.auto()
  SLATERDET_ROT = enum.auto()
  SLATERDET_ROTC = enum.auto()
  DIAG = enum.auto()
  CUSTOM = enum.auto()
  # todo: multiSD


def get_ansatz(ansatz: str) -> RepOptions:
  """
  Given a string, return the correct RepOptions,
  or representation options enum
  """

  # Later this should be styled after afqmctools processing
  ansatz = ansatz.upper()
  if ansatz == "SD_ROT":
    rep_type = RepOptions.SLATERDET_ROT
  elif ansatz == "SD_ROTC":
    rep_type = RepOptions.SLATERDET_ROTC
  elif ansatz == "SD":
    rep_type = RepOptions.SLATERDET
  elif ansatz == "SDC":
    rep_type = RepOptions.SLATERDETC
  elif ansatz == "DIAG":
    rep_type = RepOptions.DIAG
  elif ansatz == "CUSTOM":
    rep_type = RepOptions.CUSTOM
  else:
    raise NotImplementedError(f"Ansatz {ansatz} not implemented!")
  return rep_type


class SlaterType(enum.Enum):
  NONE = enum.auto()
  RHF = CLOSED = enum.auto()
  UHF = COLLINEAR = enum.auto()
  GHF = NONCOLLINEAR = enum.auto()


def process_args(args_dict: dict = None, display: bool = True) -> SimpleNamespace:
  """
  Processes autoHF arguments into a single namespace using
  default values if not specified. Accepts arguments from either
  a namespace `args`, a dictionary `args_dict`, or a combination.
  If both are provided and set the same values, the dictionary takes
  priority. If neither are provided, then a namespace containing default
  settings is returned.

  Will ignore any arguments provided which do not correspond to valid keyword arguments.

  Parameters
  ----------
  args : Namespace
    a Namespace instance containing arguments (i.e. similar to the output of argparse)
  args_dict : dict
    a dict containing arguments.
    If both `args` and `args_dict` set the same argument, the value in `args_dict` takes precendence.
  """
  _old_args = dict(
    numTrials="batch_size",
    trials="batch_size",
    numSteps="steps",
    dumpTrials="dump_batch",
    dump_trials="dump_batch",
  )
  _args = dict(
    noncollinear=False,
    gpu=False,
    approx_expm=False,
    force_complex=False,
    ansatz="SD_ROT",
    opt_method="lbfgs",
    riemannian=False,
    seed=round(time.time()),
    eta=0.1,
    steps=100,
    batch_size=1,
    plot=False,
    verbose=False,
    output=None,
    dump_batch=False,
    measure_spin=False,
    random_initial_phi0=False,
    state0_scale=0.01,
  )

  if args_dict is not None:
    for key in args_dict:
      if key in _old_args:
        # deprecations are hidden by default, lets be a bit loud
        print(f'"{key}" is deprecated, please use "{_old_args[key]}" instead')
        warnings.warn(
          f"{key} is deprecated, please use {_old_args[key]} instead",
          DeprecationWarning,
          stacklevel=2,
        )
        _args[_old_args[key]] = args_dict[key]
    for key in _args.keys():
      if key in args_dict:
        _args[key] = args_dict[key]
    if "inputfile" in args_dict:
      _args["inputfile"] = args_dict["inputfile"]

  if display:
    print("\n ===== AutoHF Settings ===== \n")
    for key, value in _args.items():
      print(" " * 8 + f"{key:30s} {str(value):>30s}")

  return SimpleNamespace(**_args)
