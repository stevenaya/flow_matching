#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg2 software-properties-common

if ! grep -Rqs "deadsnakes/ppa" /etc/apt/sources.list /etc/apt/sources.list.d; then
  add-apt-repository -y ppa:deadsnakes/ppa
fi

apt-get update
packages=(
  build-essential git git-lfs openssh-client ca-certificates curl wget rsync
  ffmpeg libaio-dev tmux unzip
  python3.12 python3.12-venv python3.12-dev python3-pip python-is-python3
)

if [ "${INSTALL_DROPBEAR:-1}" != "0" ]; then
  packages+=(dropbear openssh-sftp-server)
fi

apt-get install -y --no-install-recommends "${packages[@]}"

python3.12 -V
