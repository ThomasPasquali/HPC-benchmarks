# OpenBLAS

When you clone the OpenBLAS submodule, make sure to checkout a stable version. For example:

```bash
cd OpenBLAS
git checkout 8795fc7 # Verison 0.3.29
```

## Compile Benchmarks

HCA Cluster:

```bash
# Modules include OpenBLAS
module load gcc/15.1.0 llvm/cross/EPI-0.7-development openBLAS/ubuntu/0.3.29_llvmEPI1.0

# Build benchmarks linking to system compiled library
cd OpenBLAS/benchmark
make HOSTCC=/apps/x86/gcc/15.1.0/bin/gcc TARGET=RISCV64_GENERIC CC=/apps/x86/llvm/cross/EPI-0.7/development/bin/clang
```

## Run Banchmarks

<!-- Setup python dependencies with:

```bash
python -mpip install numpy meson ninja pytest pytest-benchmark
```

Then run with:

```bash

``` -->