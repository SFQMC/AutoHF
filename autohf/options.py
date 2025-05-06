import time
import enum
from types import SimpleNamespace


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


def processes_args(args_dict: dict = None) -> SimpleNamespace:
  """
  Processes autoHF arguments into a single namespace using
     default values if not specified. Accepts arguments from either
     a namespace `args`, a dictionary `args_dict`, or a combination.
     If both are provided and set the same values, the dictionary takes
     priority. If neither are provided, then a namespace containing default
     settings is returned.

     Will ignore any arguments provided which do not correspond to valid keyword arguments.

  Inputs:
  - args : a Namespace instance containing arguments (i.e. similar to the output of argparse)
  - args_dict : a dict contiains arguments. If both `args` and `args_dict` set the same arguement,
                  the value in `args_dict` takes precendence.

  """
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
    numSteps=100,
    numTrials=1,
    plot=False,
    verbose=False,
    output=None,
    dumpTrials=False,
    measure_spin=False,
    random_initial_phi0=False,
  )

  if args_dict is not None:
    for key in _args.keys():
      if key in args_dict:
        _args[key] = args_dict[key]
    if "inputfile" in args_dict:
      _args["inputfile"] = args_dict["inputfile"]

  print("\n ===== AutoHF Settings ===== \n")
  for key, value in _args.items():
    print(" " * 8 + f"{key:30s} {str(value):>30s}")

  return SimpleNamespace(**_args)
