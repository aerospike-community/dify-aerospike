FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm@sha256:9205d67cb7f2a8469be0065605406d3ffefff70dd1a0c26b15bdbaab92e92ec2

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install libgmp-dev libmpfr-dev libmpc-dev
