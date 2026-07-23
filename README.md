# ![AutoHF](docs/_static/logo.svg)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://safire.flatironinstitute.org/autohf/dev/)
[![CI Status](https://jenkins-new.flatironinstitute.org/job/CCQ/job/AutoHF/job/main/badge/icon?subject=CI)](https://jenkins-new.flatironinstitute.org/job/CCQ/job/AutoHF/job/main/)
[![arXiv-badge](https://img.shields.io/badge/arXiv-2607.14263-red.svg)](https://arxiv.org/abs/2607.14263) 
[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/SFQMC/AutoHF/blob/main/LICENSE)

AutoHF is an automatic-differention-based lattice model Hartree-Fock solver.

In contrast to typical self-consistent-field Hartree-Fock implementations, AutoHF leverages JAX and gradient-descend algorithms to find the Hartree-Fock solution variationally. This approach can be more robust against both saddle-points or local minima and allows flexible user-supplied generalizations beyond the standard Hartree-Fock ansatz.

## Getting started

You can install autohf via pip

```
pip install sfqmc-autohf
```

For more instractions on how to install and run AutoHF, see the [Documentation](https://safire.flatironinstitute.org/autohf/dev/). Also have a look at the scripts in the [`examples/`](https://github.com/SFQMC/AutoHF/tree/main/examples) folder.

## Citation

If you use AutoHF in a scientific publication please consider citing the paper
```bibtex
@misc{autohf_arxiv,
  doi = {10.48550/ARXIV.2607.14263},
  url = {https://arxiv.org/abs/2607.14263},
  author = {Levy,  Ryan and Eskridge,  Brandon and Weber,  Lukas and Morales,  Miguel A. and Zhang,  Shiwei},
  title = {AutoHF: a general Hartree-Fock solver utilizing direct energy minimization with automatic differentiation},
  publisher = {arXiv},
  year = {2026}
}
```
