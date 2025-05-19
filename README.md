# AutoHF

AutoHF is an automatic-differention-based lattice model Hartree-Fock solver.

In contrast to typical self-consistent-field Hartree-Fock implementations, AutoHF leverages JAX and gradient-descend algorithms to find the Hartree-Fock solution variationally. This approach can be more robust against both saddle-points or local minima and allows flexible user-supplied generalizations beyond the standard Hartree-Fock ansatz.

## Install

JAX is central to AutoHF and a required dependency.
A build based on a GPU-enabled version of JAX is highly recommended
if you intend to run AutoHF for large systems.
See the [official installation instructions for JAX](https://jax.readthedocs.io/en/latest/installation.html).

It is also recommended, but not required to install `afqmctools` to utilize the lattice model Hamiltonian tools that are implemented there.
As of now, the release of `afqmctools` is pending.

### With GPU verion of JAX

```bash
$ cd /path/to/autohf/
$ module load cuda/12 cudnn python
$ python -m venv venv
$ . venv/bin/activate
$ pip install .[cuda12]
```

CUDA 11 is also available, by replacing `cuda12` with `cuda11` in the above;
however, CUDA 12 is recommended if available.

### With CPU version of JAX

```bash
$ cd /path/to/autohf/
$ python -m venv venv
$ . venv/bin/activate
$ pip install .
```

