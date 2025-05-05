import toml


def read_input_params(infile: str):
  """
  Read parameters from a human-readable input file in TOML format.
      Returns a Dictionary containing the parameters.
  """
  with open(infile, "r") as f:
    params = toml.loads(f.read())
  return params
