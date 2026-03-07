ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON="python3.12"

# install python and pulserver dependencies
COPY install_pulserver.sh .
RUN bash install_pulserver.sh
RUN rm install_pulserver.sh

# set user
USER runner

# set entrypoint
ENTRYPOINT ["pulserver"]
