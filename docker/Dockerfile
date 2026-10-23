FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && apt-get install -y \
   openssh-client \
   git \
   gcc \
   g++ \
   make \
   python3 \
   python3-pip \
   python3-venv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# the CI pod runs as an unprivileged user, so let it install AutoHF into the venv
RUN chmod -R a+w $VIRTUAL_ENV
