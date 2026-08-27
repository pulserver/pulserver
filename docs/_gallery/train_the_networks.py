"""
Training the networks the learned figures reconstruct with
==========================================================

Two learned reconstructions appear in the reference pages: a denoiser dropped
into a plug-and-play optimizer, and an unroll whose prior and algorithm
parameters were trained together. Neither is trained when the documentation
builds -- both are trained by the runs on this page, and what they produce is
committed as a model bundle that
:func:`~pulserver.recon.load_model` resolves by name.

That is the deployment path a scanner uses, so the figures exercise it end to
end rather than holding a network in memory that nothing ever had to find.

This page is not executed by the documentation build. Run the commands
yourself to reproduce the bundles it describes.
"""

# %%
# The training set
# ----------------
#
# The fastMRI demo subset DeepInverse distributes, knee and brain -- less the
# one slice the learned figures reconstruct. The subset is four slices and one
# of them is that slice, so a network trained on all four would be scored on
# its own training data and the figures would report nothing. Holding it out
# leaves three slices, which makes what comes out a worked example of the
# path rather than a reconstruction anyone should scan with.
#
# The slices are root-sum-of-square reconstructions, so their imaginary part
# is identically zero. Training on them directly would teach a network that
# the imaginary channel is always zero, which is useless for MRI, so a smooth
# random phase is applied as a :class:`deepinv.transform.Transform` before
# every patch is used.

# %%
# The plug-and-play denoiser
# --------------------------
#
# A DnCNN taking the noise level as a third input channel, trained across the
# range a reconstruction sweeps. A blind denoiser would leave the
# regularization parameter inert, and the run refuses to write a bundle
# whose output does not move with the level it is called at.
#
# The run is DeepInverse's: :class:`deepinv.Trainer` over a
# :class:`deepinv.physics.Denoising` forward operator whose noise level is
# drawn per batch by
# :class:`deepinv.physics.generator.SigmaGenerator`, supervised by
# :class:`deepinv.loss.SupLoss` and reported through
# :class:`deepinv.metric.PSNR`.
#
# .. code-block:: console
#
#    $ python docs/_bench/train_denoiser.py --steps 2500
#    Train epoch 0: TotalLoss=..., PSNR=...
#      sigma 0.03: ... dB -> ... dB
#      the noise level reaches the network
#      wrote docs/_models/fastmri-denoiser/1.0

# %%
# The unroll
# ----------
#
# DeepInverse's proximal gradient descent taken ``unfold=True`` over
# Pulserver's :class:`~pulserver.recon.NormalEquationL2` data fidelity, so
# each step costs one normal-operator apply rather than a transform pair.
# It is trained against the scan the figure reconstructs -- Cartesian,
# four-fold undersampled with a fully sampled centre, through the same
# analytic receive array -- and the run refuses to write a bundle that did
# not improve on the adjoint it starts from.
#
# A bundle records an architecture its manifest can construct, and an
# unfolded optimizer is assembled rather than constructed. What is deployed
# is therefore the prior network, with the algorithm parameters the unroll
# learned recorded in the manifest beside it, so the optimizer reassembles
# from the bundle and nothing else.
#
# .. code-block:: console
#
#    $ python docs/_bench/train_unroll.py --steps 600
#    Train epoch 0: TotalLoss=..., PSNR=...
#      adjoint ... dB -> unrolled ... dB
#      learned beta=..., g_param=..., lambda=..., stepsize=...
#      wrote docs/_models/fastmri-unroll/1.0

# %%
# The slice stack
# ---------------
#
# :class:`~pulserver.recon.ContextAgnosticDenoiser` is worth looking at on
# contiguous slices rather than on independent pictures, so the figure
# denoises a real brain volume. TorchIO fetches IXITiny from a third-party
# host, and every figure in these pages is drawn on every documentation
# build, so the download happens once here:
#
# .. code-block:: python
#
#    import pulserver.recon as recon
#
#    dataset = recon.IXITiny("data/ixi-tiny", download=True)
#
# and the central slices of one subject are committed beside the model
# bundles:
#
# .. code-block:: console
#
#    $ python docs/_bench/make_ixi_stack.py
#    wrote docs/_models/ixi-stack (144 KiB, (16, 83, 55))

# %%
# Deploying what came out
# -----------------------
#
# Each run writes a directory holding the weights, a manifest naming the
# architecture and the arguments to build it with, and a checksum. Nothing
# else is needed to reach it:
#
# .. code-block:: python
#
#    import pulserver.recon as recon
#
#    denoiser = recon.load_model("fastmri-denoiser")
#
# A scanner finds the bundle on its own search path, which is what
# :data:`~pulserver.recon.MODEL_PATH_ENV` sets; the figures pass
# ``paths=[...]`` to read the copy in this repository instead.
