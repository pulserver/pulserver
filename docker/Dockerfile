# pulserver with its console, BrainWeb and coil extras, BrainWeb's normal
# brain, and the field maps and VOPs of the virtual scanner's coils solved in
# it, on which every exam is scanned. Run without a command, it serves the
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
RUN pip install --no-cache-dir "$(echo /wheels/pulserver-*.whl)[console,brainweb,coils]" \
 && rm -rf /wheels

# BrainWeb's normal brain, downloaded while the image is built, so that no
# exam reaches BrainWeb's server; brainweb-dl reads it from BRAINWEB_DIR.
ENV BRAINWEB_DIR=/opt/brainweb
RUN for attempt in 1 2 3; do \
      python -c "from pulserver.virtual import BrainWeb; BrainWeb().fractions" && break; \
      rm -rf "$BRAINWEB_DIR"; [ "$attempt" = 3 ] && exit 1; sleep 30; \
    done

# The coils' field maps and VOPs, which the Fields workflow solves in the same
# brain with mariepy and publishes as an image of those files alone.
COPY --from=ghcr.io/pulserver/brainweb-fields@sha256:c102b4f2563e1a1a19ca25d9cd3b9b9d7505555041b4795310aee4725a659ddb / /console/fields/

# The limits the console's sequences design under.
COPY docker/limits/limits.txt /console/limits.txt
RUN mkdir -p /console/user/plugins /console/user/recon

EXPOSE 8765
COPY docker/entrypoint.sh /console/entrypoint.sh
ENTRYPOINT ["sh", "/console/entrypoint.sh"]
