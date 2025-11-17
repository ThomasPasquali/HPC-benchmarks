# OpenBLAS

When you clone the OpenBLAS submodule, make sure to checkout a stable version. For example:

```bash
cd OpenBLAS
git checkout 993fad6 # Verison 0.3.30
```

## Compile

```bash
make PREFIX="path/to/intsall" install
```

## Banchmark

Setup python dependencies with:

```bash
python -mpip install numpy meson ninja pytest pytest-benchmark
```

The run with:

```bash

```