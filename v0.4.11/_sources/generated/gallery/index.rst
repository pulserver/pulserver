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

    <div class="sphx-glr-thumbcontainer" tooltip="In this lesson you run a whole scan without writing any plugin. You use a sequence and a reconstruction that ship with pulserver (``gre2d``, a 2D gradient echo, and pics, a compressed-sensing reconstruction with BART), and the virtual scanner plays the sequence on a phantom. You will do what a scanner does: ask for the protocol, edit it, start the scan and get the image back.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_01_protocol_to_image_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/01_protocol_to_image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">1. Your first scan</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 1 the operator edited the protocol of gre2d, and pulserver answered every edit with the TE, the TR and the scan time. Both the protocol and the answers came from a plugin: a short Python class wrapped around a sequence function. In this lesson you write that plugin yourself, one piece at a time, and after each piece you look at what changed for the operator.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_02_sequence_plugin_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/02_sequence_plugin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2. Your sequence in the scanner UI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When the operator presses Scan, pulserver does three things before the scanner plays a single block: it designs the whole scan with your sequence function, checks it against the scanner&#x27;s limits, and converts it into the form the scanner plays. In lesson 1 all three were one call, console.design(&quot;generate&quot;, ...). In this lesson you run them one at a time and look at what each one produces.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_03_scanner_representation_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/03_scanner_representation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3. What the scanner plays</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scanner sends its raw data back the way Gadgetron expects it: as an ISMRMRD (MRD) stream, a header followed by one acquisition per readout. A scanner knows little about your sequence, so most of what a reconstruction needs is missing from that stream. pulserver fills it in from your sequence before your recon plugin sees a single readout.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_04_reconstruction_plugin_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/04_reconstruction_plugin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">4. What your recon receives</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 4 your plugin received the k-space of a slice and did nothing with it. In this lesson it returns an image: first with a plain Fourier transform, then with a compressed-sensing reconstruction from bartorch, the BART toolbox in Python. Between the two only recon changes; the gadgets, the trigger and the sorting stay the same, as the Gadgetron Online Class swaps reconstruction engines behind one buffer.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_05_recon_plugin_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/05_recon_plugin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">5. Your recon plugin</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="You now have a sequence plugin (lesson 2) and a recon plugin (lesson 5). On a scanner they are two files in two folders, and the operator uses them like any shipped sequence and reconstruction. In this lesson you put them in those files, load them into the virtual scanner as a scanner would, and scan with them. Then you check the images against the physics: if the flip angle, the timing or the reconstruction were wrong, the contrast would show it.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_06_testing_on_the_virtual_scanner_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/06_testing_on_the_virtual_scanner`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6. Test your pair on the virtual scanner</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="In this Tour you resolve prescriptions of a 2D gradient echo the way a design call does for the scanner UI. You see two things the resolved protocol depends on: the receiver bandwidth the readout realizes on the sampling rasters, and the shortest echo time the readout admits.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_01_protocol_resolution_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/01_protocol_resolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Protocol resolution for a 2D gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this Tour you convert shipped pypulseqpp sequences into the scanner representation and read what the conversion reduces them to: the repetition of each subsequence, the virtual segments it is cut into, and how those counts scale with the length of the scan.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_02_segmentation_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/02_segmentation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmentation of a sequence for the scanner</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this Tour you move the field of view of two designs away from the isocentre and see where the phase of the offset is applied. The scanner applies it as the frequency and phase offsets of each readout. The reconstruction proxy applies the part those offsets cannot carry when the readout gradient varies during sampling.">

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
