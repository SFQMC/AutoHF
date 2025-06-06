.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:
   
   user/installation
   user/quickstart
   user/hams
   user/manual
   user/examples_include

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   :hidden:
   
   api/modules


User's Guide and API Reference
==============================

AutoHF is an automatic-differention-based lattice model Hartree-Fock solver.

In contrast to typical self-consistent-field Hartree-Fock implementations, AutoHF leverages JAX and gradient-descend algorithms to find the Hartree-Fock solution variationally. This approach can be more robust against both saddle-points or local minima and allows flexible user-supplied generalizations beyond the standard Hartree-Fock ansatz.

See the `Quickstart Guide <user/quickstart.html>`_ for more information on how to use AutoHF.

See the `API Reference <api/modules.html>`_ for detailed information on the modules and functions available in AutoHF.

