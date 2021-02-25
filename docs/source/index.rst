.. statOT documentation master file, created by
   sphinx-quickstart on Wed Jan 20 12:48:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to statOT's documentation!
==================================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

statOT package
==============

Core implementation
-------------------
.. automodule:: statot.inference
    :members:
    :undoc-members:
    :show-inheritance:

CellRank wrapper
----------------
.. automodule:: statot.cr
    :members:
    :undoc-members:
    :show-inheritance:

pyKeOps-numpy implementation
----------------------------
We offer an implementation of entropic (Sinkhorn) OT-based StationaryOT using the [pyKeOps](http://kernel-operations.io/keops/index.html) library. 
This utilises the `LazyTensor` class which evaluates `N x N` kernels on the fly rather than storing precomputed kernels, allowing the method to 
scale for datasets of up to `100,000` cells. This module generally requires a GPU.

.. automodule:: statot.keops
    :members:
    :undoc-members:
    :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
