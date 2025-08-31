#!/bin/bash

set -e

# mkdir hpcg-out
ARCH=$1
LAlib=$2
HPL_PATH=${3:-'hpl'}
TOPdir="$(pwd)/${HPL_PATH}"

if [[ -z "$ARCH" || -z "$LAlib" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 <ARCH> <LAlib> [HPL_PATH]"
  exit 1
fi

sed "s|#ARCH#|${ARCH}|g; s|#LAlib#|${LAlib}|g; s|#TOPdir#|${TOPdir}|g" Make.in > Make.out

cxx_compiler=g++
additional_cmake_config=

if [[ ! -d ${HPL_PATH} ]]; then
  curl -O https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
  tar -xzf hpl-2.3.tar.gz
  rm hpl-2.3.tar.gz
  mv hpl-2.3 ${HPL_PATH}
else
  echo "Directory ${HPL_PATH} already exists; assuming source code has been downloaded before"
fi

cd ${HPL_PATH}
cp ../Make.out "./Make.${ARCH}"

if [ -d bin ]; then
  echo "directory \"bin\" already exists - deleting previous HPL build"
  rm -rf bin;
fi

# ./configure
make -j16 "arch=${ARCH}"
cp "bin/${ARCH}/xhpl" ..

[[ ! -z $? ]] || echo "Could'n find the xhpl binary"

echo "Done!"