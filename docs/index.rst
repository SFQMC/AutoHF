.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:
   
   user/installation
   user/quickstart
   user/examples_include

   api/autohf
.. toctree::
   :maxdepth: 3
   :caption: API Reference
   :hidden:
   


User's Guide and API Reference
==============================

AutoHF is an automatic-differentiation-based lattice model Hartree-Fock solver.

In contrast to typical self-consistent-field Hartree-Fock implementations, AutoHF leverages JAX and gradient-descent algorithms to find the Hartree-Fock solution variationally. This approach can be more robust against both saddle-points and local minima and allows flexible user-supplied generalizations beyond the standard Hartree-Fock ansatz.

See the `Quickstart Guide <user/quickstart.html>`_ for more information on how to use AutoHF.

See the `API Reference <api/modules.html>`_ for detailed information on the modules and functions available in AutoHF.

