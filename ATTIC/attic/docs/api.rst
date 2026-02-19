API References
==============

Core
----
Core sequence representation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.Sequence
   pulserver.Opts 

Blocks
------
Subroutines for the generation of sequence blocks, e.g., 
preparation modules, rf pulses, phase encoding, readout.

RF Pulses
^^^^^^^^^
Non-adiabatic RF pulses blocks, including both the RF events
and (for spatially-selective pulses), the accompanying gradient event.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.blocks.make_hard_pulse
   pulserver.blocks.make_slr_pulse
   pulserver.blocks.make_spsp_pulse
   
Readout
^^^^^^^
Readout blocks, including both the gradient (either trapezoidal or arbitrary, e.g., for spiral imaging), 
and the accompanying adc events.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.blocks.make_line_readout
   pulserver.blocks.make_spiral_readout
   
Phase Encoding
^^^^^^^^^^^^^^
Phase encoding blocks, including phase blips for EPI sampling.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.blocks.make_phase_encoding
   
Miscellaneous
^^^^^^^^^^^^^
Miscellaneous blocks such as gradient spoiling.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.blocks.make_spoiler_gradient
   

   