# pulserver with its console extra: the reconstruction proxy and the console
# gateway of the virtual scanner.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*

COPY . /src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir "/src[console,brainweb]"

ENTRYPOINT ["pulserver"]
