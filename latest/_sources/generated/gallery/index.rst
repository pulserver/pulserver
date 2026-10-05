:orphan:

=====================
Gallery source pages
=====================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

======
Course
======

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A scan through pulserver passes four stages: a scanner sequence resolves the protocol the operator edits and designs the sequence, the design is converted into the representation the scanner plays, the scanner plays it, and the raw data are enriched from the sequence and reconstructed. This lesson runs all four in one process, on the virtual scanner, and returns an image.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_01_protocol_to_image_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/01_protocol_to_image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">1. From protocol to image</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A scanner sequence binds a pypulseqpp sequence function to the entries of the scanner protocol. The operator edits the entries; pulserver converts their values into the function&#x27;s arguments, evaluates the protocol under the scanner limits, and returns the protocol the design achieves. The previous lesson used the shipped gre2d; this lesson writes one like it.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_02_sequence_plugin_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/02_sequence_plugin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2. A scanner sequence and its protocol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Pulseq file lists every block of a scan. The scanner plays it as an execution stream of segment instances: each virtual segment, an ordered list of base blocks, is prepared once and played many times with new amplitudes and phases. This lesson converts the design of the previous lessons into that representation, reads what it was reduced to, and walks the stream as the scanner&#x27;s playout does. The model is described in /explanations/scanner-representation.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_03_scanner_representation_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/03_scanner_representation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3. Scanner representation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The raw data a scanner&#x27;s reconstruction client streams carry the samples of each readout and little else: the encoding counters and flags it records describe the interpreter&#x27;s loop, not the sequence. The reconstruction proxy replaces them with what the sequence states, then hands the readouts to a reconstruction plugin grouped into reconstruction units. This lesson acquires a series of the design from the first lesson, enriches it, and reconstructs it with a plugin of its own. The enrichment is described in /explanations/reconstruction.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_04_reconstruction_plugin_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/04_reconstruction_plugin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">4. A reconstruction plugin</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson is optional: it extends the course rather than completing it. The virtual scanner stands in for the scanner and nothing else, so a sequence and a reconstruction plugin can be tested before a scanner is involved. It acquires in two ways: along the played trajectory from an analytic phantom, which leaves relaxation out, or by a Bloch simulation of every block the cache plays. The conversion itself is checked by comparing the gradients a sequence asks for with those its cache plays. The models are described in /explanations/virtual-scanner.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_05_testing_on_the_virtual_scanner_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/05_testing_on_the_virtual_scanner`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">5. Testing on the virtual scanner</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

=====
Tours
=====

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This Tour resolves prescriptions of a two-dimensional gradient echo the way a design call does for the scanner UI, and shows what the resolved protocol depends on: the receiver bandwidth a readout realizes on the sampling rasters, and the shortest echo time the readout admits.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_01_protocol_resolution_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/01_protocol_resolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Protocol resolution for a 2D gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This Tour converts shipped pypulseqpp sequences into the scanner representation and reads what the conversion reduces them to: the repetition of each subsequence, the virtual segments it is cut into, and how those counts scale with the length of the scan.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_02_segmentation_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/02_segmentation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmentation of a sequence for the scanner</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This Tour moves the field of view of two designs away from the isocentre and shows where the phase of the offset is applied: by the scanner, as the frequency and phase offsets of each readout, and by the reconstruction proxy, for the part those offsets cannot carry when the readout gradient varies during sampling.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_03_fov_offset_enrichment_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/03_fov_offset_enrichment`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Field-of-view offset and the proxy phase</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /generated/gallery/01-course/index.rst
   /generated/gallery/02-tours/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: gallery_python.zip </generated/gallery/gallery_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: gallery_jupyter.zip </generated/gallery/gallery_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
