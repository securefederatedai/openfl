# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# docker build --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg HTTP_PROXY=$HTTP_PROXY -t openfl .

ARG BASE_IMAGE=python:3.6-slim
FROM $BASE_IMAGE

LABEL maintainer "Walter riviera <walter.riviera@intel.com>"

ARG http_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG https_proxy
ARG socks_proxy
ARG ftp_proxy
ARG no_proxy

ARG USERNAME
ARG USER_ID
ARG GROUP_ID

ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV socks_proxy=$socks_proxy
ENV ftp_proxy=$ftp_proxy
ENV no_proxy=$no_proxy


# Run this docker as User
RUN addgroup --gid $GROUP_ID $USERNAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

#########################

ENV install_dir=/usr/openfl/

WORKDIR /home/

COPY openfl $install_dir

COPY openfl/openfl-docker/docker_agg.sh /home/.
COPY openfl/openfl-docker/docker_col.sh /home/.

# Set locale details
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install OpenSSH for MPI to communicate between containers 
RUN apt-get update && apt-get install -y --no-install-recommends ssh openssh-client openssh-server \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN set -o pipefail
RUN grep -v StrictHostKeyChecking < /etc/ssh/ssh_config > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# Install the openfl package and its dependency.
RUN pip3 install --upgrade pip
RUN pip install $install_dir


# Setting permissions to run container as user
RUN mkdir -p /home/openfl
RUN chown ${USERNAME} /home/openfl

USER $USERNAME
