#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <nodes> <tasks>"
  echo "  nodes = number of nodes"
  echo "  tasks = total MPI tasks (P * Q grid)"
  exit 1
fi

nodes=$1
tasks=$2

# -------------------------
# Fixed parameters
# -------------------------
mem=0.5                     # GB per node
hpl_NB=232                  # block size
hpl_threshold=16.0
hpl_PFACT=2
hpl_NBMIN=4
hpl_NDIV=2
hpl_RFACT=0
hpl_BCAST=1
hpl_DEPTH=0
hpl_SWAP=2
hpl_swapping_threshold=64
hpl_L1=0
hpl_U=0
hpl_equilibration=1
hpl_memory_alignment=8
hpl_PMAP=0

# -------------------------
# Derived parameters
# -------------------------
totalmem=$((mem * nodes))   # GB
# Compute N (rounded to multiple of NB)
N=$(python3 - <<EOF
import math
totalmem = $totalmem
nodes = $nodes
NB = $hpl_NB
# totalmem in GB, convert to bytes
val = (totalmem * (1024**3) * nodes / 8.0)**0.5
N = int(round(val / NB) * NB)
print(N)
EOF
)

# Compute a "nice" process grid (P, Q) from tasks
read P Q <<< $(python3 - <<EOF
from sympy import divisors, divisor_count
tasks = $tasks
divs = divisors(tasks)
mid = divisor_count(tasks)//2
calc_Q = divs[mid]
calc_P = tasks // calc_Q
print(calc_P, calc_Q)
EOF
)

# -------------------------
# Generate HPL.dat
# -------------------------
cat > HPL.dat <<EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$N
1            # of NBs
$hpl_NB
$hpl_PMAP       PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
$P          Ps
$Q          Qs
$hpl_threshold  threshold
1            # of panel fact
$hpl_PFACT      PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
$hpl_NBMIN      NBMINs (>= 1)
1            # of panels in recursion
$hpl_NDIV       NDIVs
1            # of recursive panel fact.
$hpl_RFACT      RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
$hpl_BCAST      BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
$hpl_DEPTH      DEPTHs (>=0)
$hpl_SWAP       SWAP (0=bin-exch,1=long,2=mix)
$hpl_swapping_threshold   swapping threshold
$hpl_L1         L1 in (0=transposed,1=no-transposed) form
$hpl_U          U  in (0=transposed,1=no-transposed) form
$hpl_equilibration    Equilibration (0=no,1=yes)
$hpl_memory_alignment  memory alignment in double (> 0)
EOF

echo "Generated HPL.dat with N=$N, NB=$hpl_NB, P=$P, Q=$Q"
