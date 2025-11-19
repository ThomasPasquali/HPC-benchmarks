# Popcorn Kernel K-Means

TODOs:
* Check out the code
* Use openblas more
* Write better output
* Implement naive version (no openblas)

## Compilation

First run
```bash
ln -s ../ccutils .
```

### HCA Cluster

```bash
ml cmake/4.0.0 llvm/cross/EPI-development openBLAS/ubuntu/0.3.29_llvmEPI1.0
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DOpenBLAS_DIR=/apps/riscv/ubuntu/openBLAS/0.3.29_llvmEPI1.0/lib/cmake/openblas -B build
cd build/
make popcornkmeans
```

## Example Run

```bash
# From build
export OMP_NUM_THREADS=2
./popcornkmeans -n 1000 -d 3 -k 15 -m 10 --init random -f linear -l 2 --runs 5
```

## Running Experiments

TODO SbatchMan