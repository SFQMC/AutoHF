# ![AutoHF](docs/_static/logo.svg)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://safire.flatironinstitute.org/autohf/dev/)
[![CI](https://jenkins.flatironinstitute.org/buildStatus/icon?job=SAFIRE%2Fmain&subject=CI)](https://jenkins-new.flatironinstitute.org/job/CCQ/job/AutoHF/job/main/)

AutoHF is an automatic-differention-based lattice model Hartree-Fock solver.

In contrast to typical self-consistent-field Hartree-Fock implementations, AutoHF leverages JAX and gradient-descend algorithms to find the Hartree-Fock solution variationally. This approach can be more robust against both saddle-points or local minima and allows flexible user-supplied generalizations beyond the standard Hartree-Fock ansatz.

## Getting started

For instractions on how to install and run AutoHF, see the [Documentation](https://safire.flatironinstitute.org/autohf/dev/). Also have a look at the scripts in the `examples/` folder.
