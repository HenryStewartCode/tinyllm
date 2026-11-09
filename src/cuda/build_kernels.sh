#!/usr/bin/env bash
# Regenerate kernels.ptx from kernels.cu. Only needed after changing the kernels, on a machine
# with the CUDA toolkit. sm_80 is the oldest architecture targeted (Ampere, RTX 30 series and A100),
# because tensor cores learned bfloat16 there. The driver recompiles the PTX for the GPU it finds.
set -euo pipefail
cd "$(dirname "$0")"
nvcc --ptx -arch=sm_80 -O3 --std=c++17 kernels.cu -o kernels.ptx
echo "kernels.ptx: $(wc -c < kernels.ptx) bytes"
