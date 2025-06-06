.. _hams:

Implemented Hamiltonians
========================


Lattice Models
""""""""""""""

Consider a model with ``N`` sites 

* ``T`` 

  * :math:`t_{ij} c^\dagger_{i\sigma}c_{j\sigma}`
  * size: ``(2,N,N)`` for UHF, ``(1,2*N,2*N)`` for GHF

* ``U``

  * :math:`U_i n_{i\uparrow}n_{i\downarrow}`
  * size: ``(N,)`` vector
  * A single float will automatically be used for all sites 
    (``U=1.5`` :math:`\to` ``1.5*np.ones(N)``)

* ``V`` 

  * :math:`V_i n_{i}n_{j}`
  * size: ``(N,N)`` matrix
  * implemented as ``U1+U2``


* ``U1``

  * :math:`U^1_{ij}(\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow})`
  * size: ``(N,N)`` matrix


* ``U2``

  * :math:`U^2_{ij}(\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow})`
  * size: ``(N,N)`` matrix

* ``J``

  * :math:`J_{ij} (\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}+\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow}+\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}  +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow})`
  * size: ``(N,N)`` matrix

Cholesky Operators
""""""""""""""""""

See Example of UHF N2

.. code-block:: python

   # Obtain one-body T (2,NMO,NMO) and cholesky_operators (nchol,2,NMO,NMO)
   H = autohf.AutoHFHamiltonian(T=(T, T), U=0) # hack
   def molecular_energy(state, rdms):
     return (
         autohf.terms.cholesky.hf_onebody(rdms, onebody)
         + autohf.terms.cholesky.hf_cholesky(rdms, cholesky_operators)
         + nuclear_repulsion
           )
