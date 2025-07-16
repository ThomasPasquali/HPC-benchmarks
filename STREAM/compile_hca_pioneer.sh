#!/bin/bash

## !! Please run this from a Pioneer board !!

module load llvm/EPI-development

cd STREAM

sed -i 's/CC := gcc/CC := clang/' Makefile
make clean
make stream_c.exe