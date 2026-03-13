pulserver
=========

This scaffold contains bridge/interface contracts and host stubs.

Installation
------------

Install pulserver from GitHub (dev branch):

.. code-block:: bash

   pip install "git+ssh://git@github.com/pulserver/pulserver.git@dev"

Development install with test tools:

.. code-block:: bash

   pip install "git+ssh://git@github.com/pulserver/pulserver.git@dev#egg=pulserver[dev,test]"

Included from monorepo
----------------------

- bridge/
- python/pulserver/core/_base.py
- python/pulserver/core/_params.py
- python/pulserver/__init__.py
- LICENSE.txt

Testing
-------

Run Python tests:

.. code-block:: bash

   pytest tests/python

Run Nim bridge tests:

.. code-block:: bash

   bash tests/nim/run_tests.sh

Bridge plugin loading
---------------------

pypulseq_host loads plugins directly from the full script path passed via
--script.

Example:

.. code-block:: bash

   ./bridge/pypulseq_host --script bridge/tests/test_plugin.py --validate-only

Fast Sequence Path
------------------

pulserver.pulseq.Sequence is a sequential-only pypulseq.Sequence replacement
for production bridge execution. It disables positional set_block() and can
skip per-block deduplication and continuity checks during build.

Use pulserver.io.write(seq, output=...) to write either to disk or binary
blobs.

.. code-block:: python

   import pypulseq as pp
   import pulserver
   import pulserver.pulseq as ps

   seq = ps.Sequence()
   lab = ps.make_label("MYLAB", "SET", 1)
   seq.add_block(lab)
   # seq.add_block(...)

   payload = pulserver.io.write(seq, output=None, check_timing=False)  # bytes

Notes
-----

- Intended as a standalone interface package for bridge/plugin contracts.
- MATLAB host stubs are kept under bridge/.
