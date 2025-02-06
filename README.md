# AutoHF

AutoHF is an autodifferention-based Lattice model Hatree-Fock solver.
While the foundational formulation of Hartree-Fock theory is based on a variational anstaz, practical Hartree-Fock is typically based on a self-consistent field algorithm.
Here, we use autodiff to directly optimize varitional ansätze in order to directly obtain HF results.
This implementation requires only that a function to evaluate the energy of a specific ansatz be written.
Then, one of many possible optimzers may be applied with all gradients, hessians, etc. computed automatically with no need to write and maintain complicated code.

## Install

JAX is central to autoHF and is, of course, a required dependency.
A build based on a GPU-enabled version of JAX is highly recommended
if you intend to run autohf for large systems.
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

Cuda 11 is also available, by replacing `cuda12` with `cuda11` in the above;
however, Cuda 12 is recommended if available.

### With CPU version of JAX

```bash
$ cd /path/to/autohf/
$ python -m venv venv
$ . venv/bin/activate
$ pip install .
```

