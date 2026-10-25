# pulserver with PyHySCO,
# BrainWeb's normal brain, and the field maps and VOPs of the virtual
# scanner's coils solved in it, on which every exam is scanned. Run without a command, it serves the
# console of the virtual scanner on port 8765, reconstructing each scan in its
# own process, for MaRGE's page:
#
#   docker run -d --restart unless-stopped --name pulserver \
#     -p 127.0.0.1:8765:8765 ghcr.io/pulserver/pulserver
#
# Options after the image's name are added to the console's, as
# docker/entrypoint.sh adds them. Directories mounted at /console/user/plugins
# and /console/user/recon add sequences and reconstructions to those pulserver
# ships; a file there takes the place of the shipped file of the same name.
#
# Any other pulserver command runs in its place, as compose.yaml runs the
# reconstruction proxy and a console that forwards to it.

# The full image carries the compilers; scikit-build-core takes CMake from PyPI.
FROM python:3.12 AS build

COPY . /src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src

FROM python:3.12-slim

# torch's CPU build: bartorch reconstructs the shipped sequences with it, and
# no card is used.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir "$(echo /wheels/pulserver-*.whl)" \
 && rm -rf /wheels

# PyHySCO, with which the EPI reconstruction corrects each image and volume
# for susceptibility distortion against its phase-encode-reversed reference.
# It is GPL-3.0-only: bartorch never ships it, and the image carries it as
# installed from PyPI, where its source is.
RUN pip install --no-cache-dir "bartorch[pyhysco]"

# BrainWeb's normal brain, and the coils' field maps and VOPs that the Fields
# workflow solves in it with mariepy, which that workflow publishes as an image
# of those files alone, so that neither a build nor an exam reaches BrainWeb's
# server; brainweb-dl reads the brain from BRAINWEB_DIR.
ENV BRAINWEB_DIR=/opt/brainweb
COPY --from=ghcr.io/pulserver/brainweb-fields@sha256:b6ca42ee0d21e16fccb77fee165f15fa7b77dc2f6073ade0ff7000c902e3acb6 /brainweb/ /opt/brainweb/
COPY --from=ghcr.io/pulserver/brainweb-fields@sha256:b6ca42ee0d21e16fccb77fee165f15fa7b77dc2f6073ade0ff7000c902e3acb6 /fields/ /console/fields/

# The limits the console's sequences design under.
COPY docker/limits/limits.txt /console/limits.txt
RUN mkdir -p /console/user/plugins /console/user/recon

EXPOSE 8765
COPY docker/entrypoint.sh /console/entrypoint.sh
ENTRYPOINT ["sh", "/console/entrypoint.sh"]
