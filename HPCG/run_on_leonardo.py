import argparse
import sbatchman as sbm
import sys
import pprint
from pathlib import Path
import warnings
import time

sys.path.append(str('machines' / 'Leonardo'))
from nodelists_generator import LeonardoNodelistGenerator, SlurmResources, TopologyConstraint, PARTITION_NAME_MAP

NODES=[1, 2, 4, 8, 16]
PARTITION = 1
MAX_CANDIDATES = 1000

NODELIST_TYPE_EMULATING_NANJING = 'emulating_nanjing'
NODELIST_TYPE_DIFFERENT_DISTANCES = 'different_distances'
NODELIST_TYPE_EMULATING_HAICGU = 'emulating_haicgu'

global_generator = LeonardoNodelistGenerator(verify_with_sinfo=False)

def gen_config_name(nodes: int) -> str:
    return f"{PARTITION_NAME_MAP[str(PARTITION)]}_{nodes}nodes"

def rank_nodelists(candidate_lists, _print=False):
    ranked = global_generator.rank_nodelists_by_availability(
        candidate_lists,
        partition=str(PARTITION),
        consider_pending=True,
        time_weight=0.3
    )
    
    if _print:
        print(f"\n{'Rank':<6} {'Score':<8} {'Free%':<8} {'Running':<10} {'Pending':<10} {'Topology':<15} {'Nodelist'}")
        print("-" * 100)
        
        for i, (nodelist, score, details) in enumerate(ranked[:100], 1):
            slurm_format = global_generator.format_nodelist_for_slurm(nodelist)
            # Truncate long nodelists
            if len(slurm_format) > 50:
                slurm_format = slurm_format[:47] + "..."
            
            print(f"{i:<6} {score:<8.3f} {details['free_percentage']:<8.1f} "
                    f"{details['running_nodes']:<10} {details['pending_nodes']:<10} "
                    f"{details['contiguity']:<15} {slurm_format}")
        
    return ranked

def get_nodelists_emulating_nanjing(NODES):
    config_nodelist_map = {}
    for nodes in NODES:
        config_name = gen_config_name(nodes)

        if nodes > 1:
            print(f'\n\n================== {nodes} NODES ({int(nodes/2)} per switch) ==================')
            generator = LeonardoNodelistGenerator(
                verify_with_sinfo=True,
                sinfo_states=['idle'], #, 'mixed', 'allocated'],  # Only accept these states
                sinfo_partitions=['boost_usr_prod']  # Only check nodes in boost_usr_prod partition
            )
            resources = SlurmResources(num_nodes=int(nodes/2), partition=PARTITION)
            constraint = TopologyConstraint.SAME_SWITCH
            lists = generator.generate_nodelists(resources, constraint, max_nodelists=MAX_CANDIDATES, min_nodelists=10)

            cells = {}
            for l in lists:
                cell = generator.node_to_cell[l[0]]
                if not cells.get(cell): cells[cell] = []
                cells[cell].append(l)

            candidate_lists = []
            
            for _, c in cells.items():
                for l1 in c:
                    for l2 in c:
                        if l1 != l2:
                            if set(l1) & set(l2) != set():
                                print(f'Warning: overlapping lists {l1} -- {l2}')
                            else:
                                candidate_lists.append(list(set(l1) | set(l2)))

            candidate_lists = candidate_lists[:min(MAX_CANDIDATES,len(candidate_lists))]
            # print(candidate_lists)

            ranked = rank_nodelists(candidate_lists)
            
            if len(ranked) > 0:
                config_nodelist_map[config_name] = generator.format_nodelist_for_slurm(ranked[0][0])
            else:
                config_nodelist_map[config_name] = None
        else:
            # Only for one node
            config_nodelist_map[config_name] = None

    return config_nodelist_map

def get_nodelists_emulating_haicgu(NODES):
    config_nodelist_map = {}
    for nodes in NODES:
        config_name = gen_config_name(nodes)

        if nodes % 4 == 0:
            nodes_per_l1 = int(nodes/4)
            print(f'\n\n================== {nodes} NODES ({nodes_per_l1} per L1 switch, 2 different L1, 2 different cells) ==================')
            generator = LeonardoNodelistGenerator(
                verify_with_sinfo=True,
                sinfo_states=['idle'], #, 'mixed', 'allocated'],  # Only accept these states
                sinfo_partitions=['boost_usr_prod']  # Only check nodes in boost_usr_prod partition
            )
            resources = SlurmResources(num_nodes=nodes_per_l1, partition=PARTITION)
            constraint = TopologyConstraint.SAME_SWITCH
            candidate_lists = generator.generate_nodelists(resources, constraint, max_nodelists=MAX_CANDIDATES, min_nodelists=4)

            ranked = rank_nodelists(candidate_lists)

            if len(ranked) > 0:
                config_nodelist_map[config_name] = generator.format_nodelist_for_slurm(ranked[0][0])
            else:
                config_nodelist_map[config_name] = None
        else:
            warnings.warn(f'In "different distances", nodes must be a mupltiple of 4. Ignoring {nodes=}')

    return config_nodelist_map

def get_nodelists_different_distances(NODES):
    config_nodelist_map = {}
    for nodes in NODES:
        config_name = gen_config_name(nodes)

        if nodes % 4 == 0:
            nodes_per_l1 = int(nodes/4)
            print(f'\n\n================== {nodes} NODES ({nodes_per_l1} per L1 switch, 2 different L1, 2 different cells) ==================')
            generator = LeonardoNodelistGenerator(
                verify_with_sinfo=True,
                sinfo_states=['idle'], #, 'mixed', 'allocated'],  # Only accept these states
                sinfo_partitions=['boost_usr_prod']  # Only check nodes in boost_usr_prod partition
            )
            resources = SlurmResources(num_nodes=nodes_per_l1, partition=PARTITION)
            constraint = TopologyConstraint.SAME_SWITCH
            lists = generator.generate_nodelists(resources, constraint, max_nodelists=MAX_CANDIDATES, min_nodelists=4)

            # Step 1 — group pairs by cell
            cells = {}
            for l in lists:  # each l is a pair of 2 nodes
                cell = generator.node_to_cell[l[0]]
                cells.setdefault(cell, []).append(l)
            
            print('--- L1 lists grouped by cell ---')
            pprint.pprint(cells)

            candidate_lists = []

            # Step 2 — choose two different cells
            cell_ids = list(cells.keys())

            for cellA in cell_ids:
                for cellB in cell_ids:
                    if cellA == cellB:
                        continue

                    listsA = cells[cellA]
                    listsB = cells[cellB]

                    # Need at least 2 valid pairs per cell
                    if len(listsA) < 2 or len(listsB) < 2:
                        continue

                    # Step 3 — choose 2 disjoint lists from cell A
                    for a1 in listsA:
                        for a2 in listsA:
                            if set(a1) & set(a2):
                                continue  # overlap → reject
                            # Step 4 — choose 2 disjoint pairs from cell B
                            for b1 in listsB:
                                for b2 in listsB:
                                    if set(b1) & set(b2):
                                        continue
                                    
                                    # Step 5 — ensure group A and group B do not overlap
                                    combined = a1 + a2 + b1 + b2
                                    if len(set(combined)) != nodes:
                                        continue  # overlap across cells

                                    candidate_lists.append(list(combined))

            print('-- Candidate Lists ---')
            print(candidate_lists[:10])

            ranked = rank_nodelists(candidate_lists)

            if len(ranked) > 0:
                config_nodelist_map[config_name] = generator.format_nodelist_for_slurm(ranked[0][0])
            else:
                config_nodelist_map[config_name] = None
        else:
            warnings.warn(f'In "different distances", nodes must be a mupltiple of 4. Ignoring {nodes=}')

    return config_nodelist_map

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--csv',
    #     type=str,
    #     help='Path to CSV file with Leonardo system topology',
    #     required=True,
    # )
    parser.add_argument(
        '--jobs',
        type=str,
        help='Path to jobs YAML file',
        default='',
    )
    parser.add_argument(
        '--skip-configs-gen',
        action='store_true',
        default=False,
        help='Do not generate configurations'
    )
    parser.add_argument(
        '--nodelist-type',
        type=str,
        choices=[NODELIST_TYPE_EMULATING_NANJING, NODELIST_TYPE_DIFFERENT_DISTANCES],
        help='The desired node placement',
        required=True,
    )

    args = parser.parse_args()

    # FIXME just for testing
    nnodes = 4
    NODES = [nnodes]

    config_nodelist_map = None

    if args.nodelist_type == NODELIST_TYPE_EMULATING_NANJING:
        config_nodelist_map = get_nodelists_emulating_nanjing(NODES)

    elif args.nodelist_type == NODELIST_TYPE_DIFFERENT_DISTANCES:
        while (
            config_nodelist_map is None or 
            config_nodelist_map.get(gen_config_name(nnodes)) is None or 
            len(config_nodelist_map.get(gen_config_name(nnodes))) <= 0
        ):
            time.sleep(5)
            print(f'Trying to find a {nnodes}-nodes list...')
            config_nodelist_map = get_nodelists_different_distances([nnodes])

    elif args.nodelist_type == NODELIST_TYPE_EMULATING_HAICGU:
        while (
            config_nodelist_map is None or 
            config_nodelist_map.get(gen_config_name(nnodes)) is None or 
            len(config_nodelist_map.get(gen_config_name(nnodes))) <= 0
        ):
            time.sleep(5)
            print(f'Trying to find a {nnodes}-nodes list for HAICGU...')
            # Use your HAICGU function here
            config_nodelist_map = get_nodelists_emulating_haicgu([nnodes])

    if config_nodelist_map is None:
        raise Exception('should not happen')

    if not args.skip_configs_gen:
        for nodes in NODES:
            config_name = gen_config_name(nodes)
            sbm.create_slurm_config(
                name=config_name,
                cluster_name='leonardo',
                partition=PARTITION_NAME_MAP[str(PARTITION)],
                account='try25_HNS',
                nodes=str(nodes),
                ntasks=str(nodes),
                cpus_per_task=1,
                time="00:10:00",
                gpus=0,
                nodelist=config_nodelist_map[config_name],
                qos="normal",
                modules=[
                    "gcc/12.2.0",
                    "openmpi/4.1.6--gcc--12.2.0-cuda-12.2"
                ],
                # custom_headers=[
                #     "#SBATCH --mail-type=END",
                #     "#SBATCH --mail-user=thomas.pasquali@unitn.it"
                # ],
                overwrite=True,
            )

    pprint.pprint(config_nodelist_map)

    if args.jobs and len(args.jobs) > 0:
        jobs = sbm.launch_jobs_from_file(args.jobs, dry_run=True) # The dry run is not necessary

        for job in jobs:
            if not job.variables: job.variables = {}
            job.variables['nodelist'] = config_nodelist_map[job.config_name]
            sbm.job_submit(job)

        pprint.pprint(jobs)


if __name__ == "__main__":
    main()