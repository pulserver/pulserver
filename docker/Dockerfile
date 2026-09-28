# pulserver with its console, BrainWeb and coil extras, and BrainWeb's normal
# brain, which an exam on the subject brainweb is scanned on. Run without a
# command, it serves the console of the virtual scanner on port 8765,
# reconstructing each scan in its own process, for MaRGE's page:
#
#   docker run -d --restart unless-stopped --name pulserver \
#     -p 127.0.0.1:8765:8765 ghcr.io/pulserver/pulserver
#
# Any other pulserver command runs in its place, as compose.yaml runs the
# reconstruction proxy and a console that forwards to it.

# The full image carries the compilers; scikit-build-core takes CMake from PyPI.
FROM python:3.12 AS build

COPY . /src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src

FROM python:3.12-slim

# torch's CPU build: bartorch samples the head coils' models with it, and no
# card is used.
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

# The console's sequence, its reconstruction, and the limits it designs under.
COPY tests/plugins/gre2d.py /console/plugins/gre2d.py
COPY docker/limits/limits.txt /console/limits.txt
RUN mkdir /console/recon \
 && echo "from pulserver.recon.handlers.simplefft import PLUGIN" > /console/recon/gre2d.py

EXPOSE 8765
ENTRYPOINT ["pulserver"]
CMD ["console", "--plugins=/console/plugins", "--limits=/console/limits.txt", \
     "--store=/console/designs", "--recon-plugins=/console/recon", \
     "--host=0.0.0.0", "--port=8765", "--spacing=2", "--speed=1", \
     "--origin=https://pulserver.github.io", "--origin=http://localhost:8000", \
     "--origin=http://127.0.0.1:8000"]
