"""
Test suite for Leonardo Nodelist Generator
Tests various resource and constraint combinations
"""

import pandas as pd
import sys
from typing import List, Dict
from tabulate import tabulate
import argparse
import re
from typing import Union

# Import from the main library
# Assuming the main file is named 'leonardo_nodelist_generator.py'
try:
    from nodelists_generator import (
        LeonardoNodelistGenerator,
        SlurmResources,
        TopologyConstraint
    )
except ImportError:
    print("Error: Could not import leonardo_nodelist_generator module")
    print("Make sure leonardo_nodelist_generator.py is in the same directory or in PYTHONPATH")
    sys.exit(1)


class LeonardoTestSuite:
    """Test suite for various resource and constraint combinations"""
    
    def __init__(self, system_df: pd.DataFrame, verbose: bool = True):
        """
        Initialize test suite.
        
        Args:
            system_df: System topology dataframe
            verbose: Print detailed output
        """
        self.generator = LeonardoNodelistGenerator(system_df)
        self.verbose = verbose
        self.test_results = []
    
    def run_all_tests(self, include_queue_ranking: bool = True):
        """Run all predefined test cases"""
        print("=" * 80)
        print("Leonardo Nodelist Generator - Test Suite")
        print("=" * 80)
        
        # Test different resource configurations
        self.test_small_job()
        self.test_medium_job()
        self.test_large_job()
        self.test_full_cell()
        self.test_multi_cell()
        
        # Test different constraints
        self.test_all_constraints()
        
        # Test edge cases
        self.test_edge_cases()
        
        # Queue-aware ranking tests
        if include_queue_ranking:
            self.test_queue_ranking()
        
        # Print summary
        self.print_summary()
    
    def test_small_job(self):
        """Test: Small job (4 nodes)"""
        print("\n" + "=" * 80)
        print("TEST: Small Job (4 nodes)")
        print("=" * 80)
        
        resources = SlurmResources(
            num_nodes=4,
            tasks_per_node=4,
            cpus_per_task=8,
            gpus_per_node=4,
            partition=1
        )
        
        self._run_test(
            "Small Job - Same Switch",
            resources,
            TopologyConstraint.SAME_SWITCH
        )
        
        self._run_test(
            "Small Job - Same Rack",
            resources,
            TopologyConstraint.SAME_RACK
        )
    
    def test_medium_job(self):
        """Test: Medium job (32 nodes)"""
        print("\n" + "=" * 80)
        print("TEST: Medium Job (32 nodes)")
        print("=" * 80)
        
        resources = SlurmResources(
            num_nodes=32,
            tasks_per_node=4,
            cpus_per_task=8,
            gpus_per_node=4,
            partition=1
        )
        
        self._run_test(
            "Medium Job - Same Cell",
            resources,
            TopologyConstraint.SAME_CELL
        )
        
        self._run_test(
            "Medium Job - Minimize L2",
            resources,
            TopologyConstraint.MINIMIZE_L2_HOPS
        )
    
    def test_large_job(self):
        """Test: Large job (128 nodes)"""
        print("\n" + "=" * 80)
        print("TEST: Large Job (128 nodes)")
        print("=" * 80)
        
        resources = SlurmResources(
            num_nodes=128,
            tasks_per_node=4,
            cpus_per_task=8,
            gpus_per_node=4
        )
        
        self._run_test(
            "Large Job - Same Cell",
            resources,
            TopologyConstraint.SAME_CELL
        )
        
        self._run_test(
            "Large Job - Different Cells",
            resources,
            TopologyConstraint.DIFFERENT_CELLS
        )
    
    def test_full_cell(self):
        """Test: Full Booster cell (180 nodes)"""
        print("\n" + "=" * 80)
        print("TEST: Full Booster Cell (180 nodes)")
        print("=" * 80)
        
        resources = SlurmResources(
            num_nodes=180,
            partition=1  # Booster
        )
        
        self._run_test(
            "Full Cell - Booster",
            resources,
            TopologyConstraint.SAME_CELL
        )
    
    def test_multi_cell(self):
        """Test: Multi-cell job (512 nodes)"""
        print("\n" + "=" * 80)
        print("TEST: Multi-Cell Job (512 nodes)")
        print("=" * 80)
        
        resources = SlurmResources(
            num_nodes=512,
            partition=1
        )
        
        self._run_test(
            "Multi-Cell - Minimize L2",
            resources,
            TopologyConstraint.MINIMIZE_L2_HOPS
        )
        
        self._run_test(
            "Multi-Cell - No Constraint",
            resources,
            TopologyConstraint.NO_CONSTRAINT
        )
    
    def test_all_constraints(self):
        """Test: All constraint types with same resource request"""
        print("\n" + "=" * 80)
        print("TEST: All Constraint Types (64 nodes)")
        print("=" * 80)
        
        resources = SlurmResources(num_nodes=64, partition=1)
        
        constraints = [
            TopologyConstraint.SAME_CELL,
            TopologyConstraint.SAME_RACK,
            TopologyConstraint.SAME_SWITCH,
            TopologyConstraint.DIFFERENT_CELLS,
            TopologyConstraint.DIFFERENT_RACKS,
            TopologyConstraint.MINIMIZE_L2_HOPS,
            TopologyConstraint.NO_CONSTRAINT
        ]
        
        for constraint in constraints:
            self._run_test(
                f"64 nodes - {constraint.value}",
                resources,
                constraint,
                max_nodelists=5
            )
    
    def test_edge_cases(self):
        """Test: Edge cases"""
        print("\n" + "=" * 80)
        print("TEST: Edge Cases")
        print("=" * 80)
        
        # Single node
        self._run_test(
            "Edge Case - Single Node",
            SlurmResources(num_nodes=1, partition=1),
            TopologyConstraint.NO_CONSTRAINT
        )
        
        # Very large request
        self._run_test(
            "Edge Case - 1000 Nodes",
            SlurmResources(num_nodes=1000),
            TopologyConstraint.NO_CONSTRAINT
        )
        
        # DCGP partition
        self._run_test(
            "Edge Case - DCGP Partition (32 nodes)",
            SlurmResources(num_nodes=32, partition=2),
            TopologyConstraint.SAME_CELL
        )
    
    def test_queue_ranking(self):
        """Test: Queue-aware ranking"""
        print("\n" + "=" * 80)
        print("TEST: Queue-Aware Ranking")
        print("=" * 80)
        
        resources = SlurmResources(num_nodes=32, partition=1)
        
        # Generate candidate nodelists
        nodelists = self.generator.generate_nodelists(
            resources,
            constraint=TopologyConstraint.SAME_CELL,
            max_nodelists=10
        )
        
        if not nodelists:
            print("No nodelists generated, skipping queue ranking test")
            return
        
        print(f"\nGenerated {len(nodelists)} candidate nodelists")
        print("Ranking by availability...")
        
        try:
            ranked = self.generator.rank_nodelists_by_availability(
                nodelists,
                consider_pending=True,
                time_weight=0.3
            )
            
            print(f"\n{'Rank':<6} {'Score':<8} {'Free%':<8} {'Running':<10} {'Pending':<10} {'Topology':<15} {'Nodelist'}")
            print("-" * 100)
            
            for i, (nodelist, score, details) in enumerate(ranked[:10], 1):
                slurm_format = self.generator.format_nodelist_for_slurm(nodelist)
                # Truncate long nodelists
                if len(slurm_format) > 30:
                    slurm_format = slurm_format[:27] + "..."
                
                print(f"{i:<6} {score:<8.3f} {details['free_percentage']:<8.1f} "
                      f"{details['running_nodes']:<10} {details['pending_nodes']:<10} "
                      f"{details['contiguity']:<15} {slurm_format}")
            
            # Test with different time weights
            print("\n--- Testing Different Time Weights ---")
            for weight in [0.1, 0.5, 0.9]:
                ranked_weighted = self.generator.rank_nodelists_by_availability(
                    nodelists,
                    time_weight=weight
                )
                best = ranked_weighted[0]
                print(f"Time weight {weight:.1f}: Best score = {best[1]:.3f}, "
                      f"Free = {best[2]['free_percentage']:.1f}%")
            
            self.test_results.append({
                'test_name': 'Queue Ranking',
                'status': 'PASS',
                'nodelists': len(ranked)
            })
            
        except Exception as e:
            print(f"Queue ranking failed (this is expected if squeue is not available): {e}")
            self.test_results.append({
                'test_name': 'Queue Ranking',
                'status': 'SKIP',
                'nodelists': 0
            })
    
    def _run_test(
        self,
        test_name: str,
        resources: SlurmResources,
        constraint: TopologyConstraint,
        max_nodelists: int = 5
    ):
        """Run a single test case"""
        print(f"\n--- {test_name} ---")
        print(f"Resources: {resources.num_nodes} nodes", end="")
        if resources.partition:
            print(f", partition={resources.partition}", end="")
        print(f"\nConstraint: {constraint.value}")
        
        try:
            nodelists = self.generator.generate_nodelists(
                resources,
                constraint=constraint,
                max_nodelists=max_nodelists
            )
            
            if not nodelists:
                print("❌ No nodelists generated")
                self.test_results.append({
                    'test_name': test_name,
                    'status': 'FAIL',
                    'nodelists': 0,
                    'error': 'No nodelists generated'
                })
                return
            
            print(f"✓ Generated {len(nodelists)} nodelists")
            
            if self.verbose:
                # Analyze and display topology stats
                topology_stats = self._analyze_nodelists(nodelists)
                self._print_topology_table(topology_stats)
                
                # Show first few nodelists
                print(f"\nFirst {min(3, len(nodelists))} nodelists:")
                for i, nodelist in enumerate(nodelists[:3], 1):
                    slurm_format = self.generator.format_nodelist_for_slurm(nodelist)
                    print(f"  {i}. {slurm_format}")
            
            self.test_results.append({
                'test_name': test_name,
                'status': 'PASS',
                'nodelists': len(nodelists)
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results.append({
                'test_name': test_name,
                'status': 'ERROR',
                'nodelists': 0,
                'error': str(e)
            })
    
    def _analyze_nodelists(self, nodelists: List[List[int]]) -> List[Dict]:
        """Analyze topology characteristics of nodelists"""
        stats = []
        for nodelist in nodelists:
            topology = self.generator.analyze_nodelist_topology(nodelist)
            stats.append(topology)
        return stats
    
    def _print_topology_table(self, topology_stats: List[Dict]):
        """Print topology statistics in table format"""
        if not topology_stats:
            return
        
        table_data = []
        for i, stats in enumerate(topology_stats, 1):
            table_data.append([
                i,
                stats['num_nodes'],
                stats['num_cells'],
                stats['num_racks'],
                stats['num_switches'],
                stats['topology_type'],
                'Yes' if stats['uses_l2'] else 'No'
            ])
        
        headers = ['#', 'Nodes', 'Cells', 'Racks', 'Switches', 'Type', 'Uses L2']
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        errors = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        skipped = sum(1 for r in self.test_results if r['status'] == 'SKIP')
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ❌")
        print(f"Errors: {errors} ⚠️")
        print(f"Skipped: {skipped} ⊘")
        
        if failed > 0 or errors > 0:
            print("\nFailed/Error Tests:")
            for result in self.test_results:
                if result['status'] in ['FAIL', 'ERROR']:
                    print(f"  - {result['test_name']}: {result.get('error', 'Unknown error')}")
        
        # Summary table
        table_data = []
        for result in self.test_results:
            status_symbol = {
                'PASS': '✓',
                'FAIL': '❌',
                'ERROR': '⚠️',
                'SKIP': '⊘'
            }.get(result['status'], '?')
            
            table_data.append([
                status_symbol,
                result['test_name'],
                result['nodelists']
            ])
        
        print("\nDetailed Results:")
        print(tabulate(table_data, headers=['Status', 'Test Name', 'Nodelists'], tablefmt='grid'))


def load_leonardo_system_data(csv_path: Union[None,str] = None) -> pd.DataFrame:
    """
    Load Leonardo system data from CSV or create sample data.
    
    Args:
        csv_path: Path to CSV file with system topology
        
    Returns:
        DataFrame with system topology
    """
    if csv_path:
        print(f"Loading system data from {csv_path}...")
        pattern = re.compile(
            r"NODE\s+(\d+)\s+RACK\s+(\d+)\s+CELL\s+(\d+)\s+ROW\s+(\d+)\s+PARTITION\s+(\d+)\s+SWITCH\s+(\d+)"
        )
        
        records = []
        with open(csv_path) as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    records.append(match.groups())
        
        return pd.DataFrame(
            records,
            columns=["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
        ).astype(int)
    else:
        print("Creating sample system data (180 nodes, 1 Booster cell)...")
        # Create sample data representing 1 Booster cell
        nodes_per_rack = 30
        racks_per_cell = 6
        nodes_per_cell = nodes_per_rack * racks_per_cell
        
        data = {
            'NODE': list(range(1, nodes_per_cell + 1)),
            'RACK': [i // nodes_per_rack + 1 for i in range(nodes_per_cell)],
            'CELL': [1] * nodes_per_cell,
            'ROW': [1] * nodes_per_cell,
            'PARTITION': [1] * nodes_per_cell,
            'SWITCH': [10100 + (i // 10) * 2 for i in range(nodes_per_cell)]
        }
        
        return pd.DataFrame(data)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test Leonardo Nodelist Generator with various configurations'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with Leonardo system topology'
    )
    parser.add_argument(
        '--no-queue',
        action='store_true',
        help='Skip queue-aware ranking tests'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Load system data
    try:
        df = load_leonardo_system_data(args.csv)
        print(f"Loaded system data: {len(df)} nodes")
        print(f"Partitions: {sorted(df['PARTITION'].unique())}")
        print(f"Cells: {sorted(df['CELL'].unique())}")
        print()
    except Exception as e:
        print(f"Error loading system data: {e}")
        sys.exit(1)
    
    # Run test suite
    suite = LeonardoTestSuite(df, verbose=not args.quiet)
    suite.run_all_tests(include_queue_ranking=not args.no_queue)
    
    # Exit with appropriate code
    failed = sum(1 for r in suite.test_results if r['status'] in ['FAIL', 'ERROR'])
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()