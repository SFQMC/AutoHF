.. _autohf:

User's Guide and Developer's Manual
===================================

Overview
--------

Here is some text. Be vary careful with the formatting, as it is very easy to break the
sphinx formatting!

.. math::
  :name: simple-eq
  :label: simple-eq

  X = Y + Z

where :math:`X`,  :math:`Y`, and :math:`Z` are all made up!

.. _section:

A Section
---------

Here is some text.
A code block is shown below:


.. code-block:: python
  :caption: Sample use of AutoHF
  :name: Listing 1

  import os

  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  from jax import config

  config.update("jax_enable_x64", True)

  import numpy as np
  # import matplotlib.pyplot as plt


  import autohf


  def makeTPin(Nx, Ny, h_pin, yperiodic=False):
    N = Nx * Ny
    T = np.zeros((N, N))
    for i in range(Nx):
      for j in range(Ny):
        a = i * Ny + j
        if i + 1 < Nx or yperiodic:
          b = ((i + 1) % Nx) * Ny + j
          T[a, b] = -1
          T[b, a] = -1
        b = i * Ny + (j + 1) % Ny
        T[a, b] = -1
        T[b, a] = -1
    Sz = 1 / 2 * h_pin
    pin = np.zeros(N)
    pin[np.arange(N - Ny, N)] = Sz * (-1) ** (np.arange(Ny))
    pin[np.arange(Ny)] = Sz * (-1) ** (1 + np.arange(Ny))
    print(pin)
    print(np.sum(T, axis=0))
    assert np.allclose(T, T.T)
    return T, pin


  h_pin = 0.5  # h_pin*Sz_i = h_pin/2*Z
  Nx, Ny = 4, 4
  N = Nx * Ny
  T_, pin = makeTPin(Nx, Ny, h_pin)
  T_up, T_dn = T_ + np.diag(pin), T_ - np.diag(pin)

  H = autohf.AutoHFHamiltonian(T=(T_up, T_dn), U=1.5)

  hf_settings = {
    "verbose": False,
    "steps": 100,
    "opt_method": "lbfgs",
    "ansatz": "SD_ROT",
    "batch_size": 4,
    "gpu": False,
    "nelec": (7, 7),
  }
  autohf.solve_hf(
    H,
    settings=hf_settings,
  )

.. _another_section:

Another Section
---------------

We can refer to the equation above as :eq:`simple-eq`.
We can also refer to the code block above as :ref:`Listing 1`.
We even have a link to the :ref:`autohf` page, and to other 
sections such as :ref:`section` and :ref:`another_section`.
Please also see the footnotes [1]_ [2]_.


More on writing with rst / sphinx
---------------------------------

.. note::

   This is a note that will be shown in the documentation.

.. warning::

    This is a warning that will be shown in the documentation.

.. tip::

    This is a tip that will be shown in the documentation.
  
.. seealso::

    For more information, see the ...

.. rubric:: Footnotes

.. [1] This is a footnote that will be shown in the documentation.
   It can be used to provide additional information or references.
   Footnotes are useful for providing context or citations without cluttering the main text.

.. [2] Another footnote can be added here.

