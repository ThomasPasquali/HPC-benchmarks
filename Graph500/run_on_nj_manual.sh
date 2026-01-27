#!/bin/bash

# Root folder to save results
ROOT_DIR="./manual_results"

# Define variables from your YAML
bins=("graph500_bfs_2KiB" "graph500_bfs_32KiB" "graph500_bfs_256KiB" "graph500_bfs_8MiB")
scales=(20)
edgefactors=(8 16 32)

# Configs
declare -A configs
configs["nanjing-inter"]="8 4 2 1"
configs["nanjing-intra"]="4 2"

# Loop over partitions and nodes
for partition in "${!configs[@]}"; do
    nodes_list=(${configs[$partition]})
    for nodes in "${nodes_list[@]}"; do
       
        # Construct config folder
        config_folder="${ROOT_DIR}/${partition}_${nodes}nodes"
        mkdir -p "$config_folder"

        # Loop over bins, scales, and edgefactors
        for bin in "${bins[@]}"; do
            for scale in "${scales[@]}"; do
                for edgefactor in "${edgefactors[@]}"; do

                    # Construct tag
                    tag="graph500_${partition}"
                    tag_folder="${config_folder}/${tag}"
                    mkdir -p "$tag_folder"

                    # Construct command
                    CMD="mpirun --allow-run-as-root --map-by node -x PATH -x LD_LIBRARY_PATH -np $nodes --hostfile /root/hpc/tpasquali/HPC-benchmarks/NJ/${nodes}hosts_${partition} ./bin/${bin} ${scale} ${edgefactor}"

                    # Construct filenames
                    timestamp=$(date +%Y%m%d_%H%M%S)
                    stdout_file="${tag_folder}/stdout_${bin}_${scale}_${edgefactor}.txt"
                    stderr_file="${tag_folder}/stderr_${bin}_${scale}_${edgefactor}.txt"
                    metadata_file="${tag_folder}/metadata_${bin}_${scale}_${edgefactor}.yaml"

                    # Run command and save outputs
                    echo "Running: $CMD"
                    $CMD >"$stdout_file" 2>"$stderr_file"

                    # Save metadata
                    cat > "$metadata_file" <<EOF
bin: $bin
scale: $scale
edgefactor: $edgefactor
nodes: $nodes
partition: $partition
command: "$CMD"
timestamp: "$timestamp"
EOF

                done
            done
        done
    done
done

echo "All jobs finished."