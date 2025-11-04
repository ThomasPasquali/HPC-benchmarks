import argparse
import sbatchman as sbm
import sys
import pprint
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'machines' / 'Leonardo'))
from nodelists_generator import LeonardoNodelistGenerator, SlurmResources, TopologyConstraint, PARTITION_NAME_MAP

NODES=[1, 2, 4, 8]
PARTITION = 1
MAX_CANDIDATES = 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with Leonardo system topology',
        required=True,
    )
    parser.add_argument(
        '--jobs',
        type=str,
        help='Path to jobs YAML file',
        required=True,
    )
    parser.add_argument(
        '--skip-configs-gen',
        type=bool,
        help='Do not generate configurations',
        default=False,
    )

    args = parser.parse_args()
    config_nodelist_map = {}

    if not args.skip_configs_gen:
        generator = LeonardoNodelistGenerator(
            csv_path=args.csv,
            verify_with_sinfo=True,
            sinfo_states=['idle', 'mixed', 'allocated'],  # Only accept these states
            sinfo_partitions=['boost_usr_prod']  # Only check nodes in boost_usr_prod partition
        )
        for nodes in NODES:
            config_name = f"{PARTITION_NAME_MAP[str(PARTITION)]}_{nodes}nodes"

            if nodes > 1:
                print(f'\n\n================== {nodes} NODES ({int(nodes/2)} per switch) ==================')
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

                ranked = generator.rank_nodelists_by_availability(
                    candidate_lists,
                    partition=str(PARTITION),
                    consider_pending=True,
                    time_weight=0.3
                )
                
                print(f"\n{'Rank':<6} {'Score':<8} {'Free%':<8} {'Running':<10} {'Pending':<10} {'Topology':<15} {'Nodelist'}")
                print("-" * 100)
                
                for i, (nodelist, score, details) in enumerate(ranked[:100], 1):
                    slurm_format = generator.format_nodelist_for_slurm(nodelist)
                    # Truncate long nodelists
                    if len(slurm_format) > 50:
                        slurm_format = slurm_format[:47] + "..."
                    
                    print(f"{i:<6} {score:<8.3f} {details['free_percentage']:<8.1f} "
                            f"{details['running_nodes']:<10} {details['pending_nodes']:<10} "
                            f"{details['contiguity']:<15} {slurm_format}")
                    
                config_nodelist_map[config_name] = generator.format_nodelist_for_slurm(ranked[0][0])
            else:
                # Only for one node
                config_nodelist_map[config_name] = None

            sbm.create_slurm_config(
                name=config_name,
                cluster_name='leonardo',
                partition=PARTITION_NAME_MAP[str(PARTITION)],
                account='try25_HNS',
                nodes=str(nodes),
                ntasks=str(nodes),
                cpus_per_task=32,
                time="00:10:00",
                gpus=0,
                nodelist=config_nodelist_map[config_name],
                qos="normal",
                env=[
                    "SKIP_VALIDATION=1",
                    "OMP_PROC_BIND=true",
                    "OMP_NUM_THREADS=32",
                ],
                modules=[
                    "gcc/12.2.0",
                    "openmpi/4.1.6--gcc--12.2.0-cuda-12.2"
                ],
                custom_headers=[
                    "#SBATCH --mail-type=END",
                    "#SBATCH --mail-user=thomas.pasquali@unitn.it"
                ],
                overwrite=True,
            )

    jobs = sbm.launch_jobs_from_file(args.jobs, dry_run=True)
    
    pprint.pprint(config_nodelist_map)
    
    for job in jobs:
        if not job.variables: job.variables = {}
        job.variables['nodelist'] = config_nodelist_map[job.config_name]
        pprint.pprint(job)

        sbm.job_submit(job)

  
if __name__ == "__main__":
    main()