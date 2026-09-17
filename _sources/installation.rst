Installation
============

Requirements
------------

- numpy>=1.10.0
- scipy>=1.1.0
- matplotlib>=1.3.1
- scikit-learn>=1.5

Procedure
---------

You can install Optifik using `uv`:

.. code-block:: bash

    uv pip install optifik

Alternatively, you can install directly from source:

* `Releases on github <https://github.com/sciunto-org/optifik/releases>`_

.. code-block:: bash

    cd optifik
    uv pip install -e .

or from the main branch (in development):

.. code-block:: bash

    git clone https://github.com/sciunto-org/optifik.git
    cd optifik
    uv sync


Check the installation and the version:

.. code-block:: bash

    uv run python -c 'import optifik; print(optifik.__version__)'


Tests
-----

Install

.. code-block:: bash

    uv sync


To run the test suite:

.. code-block:: bash

    uv run pytest
