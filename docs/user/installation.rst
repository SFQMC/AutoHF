.. _installation:

Installation
------------

AutoHF is a Python based tool, and so you should have a copy of your favorite version of Python around. 
If there isn't a good version already installed, we recommend using `miniforge <https://github.com/conda-forge/miniforge>`_ to get started.

.. note::
  Currently we recommend Python version 3.12, however any version >=3.9 will work

JAX is essential to AutoHF. Installing JAX with or without GPU support can be tricky however. 
When in doubt, create a new virtual environment (venv) and start from scratch.

Installation (GPU Support)
""""""""""""""""""""""""""

To install with GPU support, create a new virtual environment (recommended) and using pip

.. code-block:: bash
  
  $ cd /path/to/autohf/
  $ python -m venv venv
  $ . venv/bin/activate
  $ pip install .[cuda12]

CUDA 13, 12, and 11 are supported using `cuda13`, `cuda12`, and `cuda11`, respectively.

.. tip::
   JAX will crash if there is no GPU found when installed with GPU support. 
   To circumvent this, AutoHF will by default can enforce a CPU mode. Use ``gpu=False`` in the settings for this behavior. 
   Likewise, be sure to use ``gpu=True`` when you want to use the GPU.  

   Look for ``[CudaDevice(id=0)]`` in AutoHF output to ensure GPU support is working properly

Installation (CPU Only)
"""""""""""""""""""""""

.. code-block:: bash
  
   $ cd /path/to/autohf/
   $ python -m venv venv
   $ . venv/bin/activate
   $ pip install .
  

Common Pitfalls
"""""""""""""""

* Install the CPU version and want to swap the GPU version? You can run ``pip install -U jax[cuda12]`` and ensure that ``jaxlib`` is the correct version
* The examples all assume CPU only. This is enforced by the code

  .. code-block:: python
    
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    
  Remove this line to enable possible CPU support
