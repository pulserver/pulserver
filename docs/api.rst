API
===

Plugin contract
---------------

The plugin contract is exposed directly from :mod:`pulserver`. The internal
``pulserver._core`` is not a user import location.

Enhanced pypulseq
-----------------

Use this namespace as a drop-in replacement for upstream :mod:`pypulseq`.
Pulserver overrides ``Sequence`` and selected helpers while re-exporting the
complete upstream namespace.

.. automodule:: pulserver.pypulseq
   :members:
   :imported-members:

Sequence design
---------------

Concrete RF, preparation, encoding, and readout modules are created through
these factories. :class:`pulserver.Module` is the only concrete-module
base class intended for direct import. RF, readout, sampling, gradient,
schedule, and system implementations are private; import their public
factories and sampling helpers directly from :mod:`pulserver` or
:mod:`pulserver.pypulseq`.

.. automodule:: pulserver
   :members:
   :imported-members:

Sampling
--------

Sampling helpers such as :func:`pulserver.from_mask`,
:func:`pulserver.radial_2d`, and :func:`pulserver.slice_groups` are part of
the flat package API. See :doc:`reference/sampling` for examples.

Sequence I/O
------------

.. automodule:: pulserver.io
   :members:
