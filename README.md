# AutoHF
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://safire.flatironinstitute.org/autohf/main)
[![CI](https://jenkins.flatironinstitute.org/buildStatus/icon?job=SAFIRE%2Fmain&subject=CI)](https://jenkins.flatironinstitute.org/job/AutoHF/job/main/)

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

## Compiling the Documentation

To compile the documentation, you must install Sphinx.
You can either do this directly via pip as,

```bash
$ pip install sphinx sphinxcontrib-bibtex sphinx-rtd-theme myst_nb sphinx-autodoc-typehints numpydoc
```

or, when installing autohf, you can install the `DOCS` optional dependency as,

```bash
$ pip install .[DOCS]
```

Once sphinx is installed, you can compile the docs using `make` with,

```bash
$ cd /path/to/autohf/docs
$ make html
```

which will generate the documentation as html in the directory, `/path/to/autohf/docs/_build/html`.
You can view the documentation locally in a web browser using the url, `file:///path/to/autohf/docs/_build/html/index.html`
(where you must replace `/path/to/autohf` with your local path to `autohf`.

