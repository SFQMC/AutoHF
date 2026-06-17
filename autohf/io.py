import warnings

import toml
import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np

def read_input_params(infile: str):
  """
  Read parameters from a human-readable input file in TOML format.
  Returns a Dictionary containing the parameters.
  """
  with open(infile, "r") as f:
    params = toml.loads(f.read())
  return params

def saveStates(states, orbFunc, energies, phi0, args, nelec, fname, dump_batch):
  """Save all the trial states in an extra file
  Will automatically save based on npy/pkl/pickle file extension
  Does nothing if fname is empty/None
  """
  import pickle
  Es = np.array(energies)
  best = np.argmin(Es)
  o = orbFunc(states[best])
  data = {
    "state": states[best],
    "Es": Es,
    "E": Es[best],
    "orbitals": o,
    "args": args,
    "phi0": phi0,
    "states": np.array(states),
    "nelec": nelec,
  }
  if dump_batch:
    orbs = [np.array(o) for o in jax.vmap(orbFunc)(states)]
    data["all_orbitals"] = orbs
  if fname == "" or fname is None:
    return data
  if fname[-4:] == ".npy":
    print("Saving states as npy:", fname)
    np.save(fname, [np.array(o) for o in orbs])
  elif fname[-3:] == ".h5":
    print("Saving states as h5:", fname)
    _save_to_h5(data, fname)
  elif fname[-4:] == ".pkl" or fname[-7:] == ".pickle":
    print("Saving Pickle:", fname)
    warnings.warn("Saving states with pickle extension. Make sure this is intentional, \n"
                  "as unpickling can be unsafe if the file is from an untrusted source.\n"
                  "consider using the h5 extension instead, which is safer and more portable.",
                  RuntimeWarning, stacklevel=2)
    with open(fname, "wb") as f:
      pickle.dump(data, f)
  else:
    print("Ignoring file extension. Saving states with pickle extension", fname + ".pkl")
    print("Saving Pickle:", fname + ".pkl")
    with open(fname + ".pkl", "wb") as f:
      pickle.dump(data, f)

  return data


def loadStates(input_, args, nelec):
  """
  Loads saved pkl from saveStates
  Supports using the same ansatz or any ansatz -> SD/SDC
  """

  if isinstance(input_, str):
    import pickle

    # TODO: support SD/SDC ->SD_ROT
    fname = input_
    print(f"Loading saved data from {fname}")
    if fname.endswith(".h5"):
      print("Reading from h5 format")
      original_data = _load_from_h5(fname)
    elif fname[-4:] == ".pkl" or fname[-7:] == ".pickle":
      print("Reading from pkl/pickle format")
      warnings.warn("The pickle format is deprecated and can be unsafe if the file \n"
                    "is from an untrusted source. Consider using the h5 format instead,\n"
                    "which is safer and more portable.", RuntimeWarning, stacklevel=2)
      warnings.warn("Be careful with loading pickle files from untrusted sources!",
                    RuntimeWarning, stacklevel=2)
      with open(fname, "rb") as f:
        original_data = pickle.load(f)
    else:
      raise RuntimeError(f"Unknown file extension for restart data: {fname}")
  elif isinstance(input_, tuple) or isinstance(input_, list):
    original_data = input_[0]
  elif isinstance(input_, dict):
    original_data = input_
  else:
    raise RuntimeError(f"Unknown restart data ({type(input_)}")

  Es = original_data["Es"]
  order = np.argsort(Es)
  phi0 = original_data["phi0"]

  if original_data["args"].batch_size < args.batch_size:
    raise NotImplementedError(
      f"Old data had {original_data['args'].batch_size} batch_size "
      f"but requesting more on restart {args.batch_size}. This is currently unsupported"
    )

  if args.ansatz == original_data["args"].ansatz:
    # easy case
    state0_ref = original_data["state"]
    state0 = original_data["states"][order[: args.batch_size]]

  elif args.ansatz == "SD" or args.ansatz == "SDC":
    if "all_orbitals" not in original_data:
      raise RuntimeError(
        "All orbitals were not saved (i.e. dump_batch was not set to True). "
        "Please rerun original calculation or keep the same trial"
      )
    # orbitals leading dim is (nspin,nwalkers,...)
    # so need a reshape

    old_orb = original_data["orbitals"]
    maxdim = max([p.shape[-1] for p in old_orb])
    dtype = old_orb[0].dtype
    state0_ref = np.zeros((len(old_orb), old_orb[0].shape[0], maxdim), dtype=dtype)
    for i, p in enumerate(old_orb):
      x, y = p.shape
      state0_ref[i, 0:x, 0:y] = old_orb[i][:]
    # now deal with picking the best of the old orbitals
    old_orbs = original_data["all_orbitals"]  # (nspin,ntrial,N,Ne)
    state0 = np.zeros((args.batch_size, *state0_ref.shape), dtype=dtype)
    for i, p in enumerate(old_orbs):
      x, y = p[0].shape
      state0[:, i, 0:x, 0:y] = old_orbs[i][order[: args.batch_size], 0:x, :y]
    ## if originally not SDC and now is SDC
    if args.ansatz == "SDC":
      state0 = jnp.stack([state0.real, state0.imag])
      state0_ref = jnp.stack([state0_ref.real, state0_ref.imag])
  print("Loaded state with energy:", original_data["E"])

  return state0, state0_ref, phi0


def _save_to_h5(data: dict, fname: str):
  with h5.File(fname, "w") as f:
      # iterate over data and save all arrays as datasets
      for key, value in data.items():
        if isinstance(value, (np.ndarray, jnp.ndarray)):
          f.create_dataset(key, data=value)
        elif key=="args":
          # for namespace objects, save attributes in a group
          grp = f.create_group(key)
          for attr_name, attr_value in vars(value).items():
            if isinstance(attr_value, (np.ndarray, jnp.ndarray)):
              grp.create_dataset(attr_name, data=attr_value)
            else:
              try:
                grp.attrs[attr_name] = attr_value
              except (TypeError, ValueError):
                # If the attribute can't be saved, convert to string
                grp.attrs[attr_name] = str(attr_value)
        else:
          # for non-array data, save as an attribute
          try:
            f.attrs[key] = value
          except (TypeError, ValueError):
            # If the value can't be saved as attribute, convert to string
            f.attrs[key] = str(value)

def _load_from_h5(fname: str) -> dict:
  """Load data from HDF5 file created by _save_to_h5"""
  from argparse import Namespace

  data = {}
  with h5.File(fname, "r") as f:
    # Load root-level attributes
    for key, value in f.attrs.items():
      data[key] = value

    # Load datasets and groups
    for key in f.keys():
      item = f[key]
      if isinstance(item, h5.Dataset):
        # Regular dataset - load as numpy array
        data[key] = item[:]
      elif key == "args" and isinstance(item, h5.Group):
        # Group - reconstruct as namespace object
        ns = Namespace()
        # Load attributes from the group
        for attr_name, attr_value in item.attrs.items():
          setattr(ns, attr_name, attr_value)
        # Load datasets from the group
        for subkey in item.keys():
          setattr(ns, subkey, item[subkey][:])
        data[key] = ns

  return data
