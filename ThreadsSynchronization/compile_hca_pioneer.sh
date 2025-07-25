#!/bin/bash

set -e

## !! Please run this from a Pioneer board !!

module load llvm/EPI-development

make clean
CXX=clang++ make all
