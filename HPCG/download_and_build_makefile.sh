#!/bin/bash

set -e

mkdir -p hpcg-out

BRANCHNAME=master

ARCH=$1

LAlib=$2
LAname=$3
LAinc=$4

MPlib=$5
MPname=$6
MPinc=$7

HPCG_PATH=${8:-'hpcg-cpu'}
TOPdir="$(pwd)/${HPL_PATH}"

if [[ -z "$ARCH" || -z "$LAlib" || -z "$LAinc" || -z "$MPlib" || -z "$MPname" || -z "$MPinc" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 <ARCH> <LAlib> <LAname> <LAinc> <MPlib> <MPname> <MPinc> [HPCG_PATH]"
  exit 1
fi

sed "s|#ARCH#|${ARCH}|g; s|#LAlib#|${LAlib}|g; s|#LAname#|${LAname}|g; s|#LAinc#|${LAinc}|g; s|#MPlib#|${MPlib}|g; s|#MPname#|${MPname}|g; s|#MPinc#|${MPinc}|g; s|#TOPdir#|${TOPdir}|g" Make.in > Make.out

if [[ ! -d ${HPCG_PATH} ]]; then
  git clone -b $BRANCHNAME --recursive https://github.com/hpcg-benchmark/hpcg.git ${HPCG_PATH}
else
  echo "Directory ${HPCG_PATH} already exists; assuming source code has been downloaded before"
fi

cd ${HPCG_PATH}
cp ../Make.out "./setup/Make.${ARCH}"

# Makefile build
make -j16 "arch=${ARCH}"

cp ./bin/xhpcg ..

echo "Done!"