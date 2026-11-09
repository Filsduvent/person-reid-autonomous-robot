#!/usr/bin/env bash
# Install an isolated Python matching Constantine, without changing system Python.
set -euo pipefail
bootstrap=/home/addirakoze/.local/share/reid-migration/bootstrap
python_prefix=/home/addirakoze/.local/share/reid-migration/python3112
mkdir -p "$bootstrap"
if test ! -x "$bootstrap/bin/micromamba"; then
    curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest --output "$bootstrap/micromamba.tar.bz2"
    sha256sum "$bootstrap/micromamba.tar.bz2" > "$bootstrap/micromamba.sha256"
    tar -xjf "$bootstrap/micromamba.tar.bz2" -C "$bootstrap" bin/micromamba
fi
if test ! -e "$python_prefix"; then
    MAMBA_ROOT_PREFIX="$bootstrap/mamba" "$bootstrap/bin/micromamba" --no-rc create --yes \
        --prefix "$python_prefix" --override-channels --channel conda-forge python=3.11.2 pip
fi
"$python_prefix/bin/python3.11" -c 'import platform, ssl, sqlite3, bz2, lzma, ctypes; assert platform.python_version() == "3.11.2"; print(platform.python_version(), ssl.OPENSSL_VERSION)'
MAMBA_ROOT_PREFIX="$bootstrap/mamba" "$bootstrap/bin/micromamba" --no-rc list --prefix "$python_prefix" --explicit > "$bootstrap/python3112-explicit.txt"
