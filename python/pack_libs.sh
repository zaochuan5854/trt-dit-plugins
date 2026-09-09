# SPDX-License-Identifier: Apache-2.0
# Copy built shared libs into the Python package dir for wheel bundling.
# Usage: ./python/pack_libs.sh [build-dir]
set -eu
build_dir="${1:-build}"
dest="$(dirname "$0")/trt_dit_plugins/lib"
mkdir -p "$dest"
cp "$build_dir"/libck_kernels.so "$build_dir"/libtrt_dit_plugins.so "$dest"/
ls -la "$dest"
