#!/usr/bin/env python3
"""
SLURM Queue Monitor - Periodically runs squeue and logs only NEW jobs in CSV format
"""
import subprocess
import time
import argparse
import csv
from datetime import datetime
from pathlib import Path


def run_squeue(format_string=None):
    """Execute squeue command and return output."""
    cmd = ["squeue"]
    if format_string:
        cmd.extend(["--format", format_string])
    else:
        # Default format for CSV output
        cmd.extend(["--format", "%.18i|%.9P|%.30j|%.8u|%.2t|%.10M|%.6D|%R"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e}")
        return None
    except FileNotFoundError:
        print("Error: squeue command not found. Make sure SLURM is installed.")
        return None


def parse_squeue_output(output, delimiter="|"):
    """Parse squeue output into list of dictionaries."""
    if not output:
        return []
    
    lines = output.split('\n')
    if len(lines) < 2:
        return []
    
    # First line is header
    headers = [h.strip() for h in lines[0].split(delimiter)]
    
    # Parse data rows
    rows = []
    for line in lines[1:]:
        if line.strip():
            values = [v.strip() for v in line.split(delimiter)]
            if len(values) == len(headers):
                rows.append(dict(zip(headers, values)))
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Monitor SLURM queue and log ONLY NEW jobs in CSV format"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="squeue_log.csv",
        help="Output CSV file path (default: squeue_log.csv)"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        help="squeue format string with | delimiter (e.g., '%%.18i|%%.9P|%%.8j|%%.8u')"
    )
    parser.add_argument(
        "--add-timestamp",
        action="store_true",
        help="Add a timestamp column to each row"
    )
    
    args = parser.parse_args()
    
    output_file = Path(args.output)
    seen_jobids = set()
    file_initialized = False
    
    print(f"Starting SLURM queue monitor (tracking NEW jobs only)...")
    print(f"Interval: {args.interval} seconds")
    print(f"Output file: {output_file}")
    if args.format:
        print(f"Format: {args.format}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            current_output = run_squeue(args.format)
            
            if current_output:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rows = parse_squeue_output(current_output)
                
                # Filter to only NEW jobs
                new_jobs = []
                for row in rows:
                    jobid = row.get('JOBID', row.get('JOBID', list(row.values())[0]))  # First column is usually JOBID
                    if jobid not in seen_jobids:
                        seen_jobids.add(jobid)
                        new_jobs.append(row)
                
                # Write only new jobs to CSV
                if new_jobs:
                    # Add timestamp column if requested
                    if args.add_timestamp:
                        for row in new_jobs:
                            row['Timestamp'] = timestamp
                    
                    # Write to CSV
                    mode = 'w' if not file_initialized else 'a'
                    with open(output_file, mode, newline='') as f:
                        fieldnames = list(new_jobs[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        
                        # Write header only once
                        if not file_initialized:
                            writer.writeheader()
                            file_initialized = True
                        
                        writer.writerows(new_jobs)
                    
                    print(f"[{timestamp}] {len(new_jobs)} new job(s) logged")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped by user")
        print(f"Total unique jobs tracked: {len(seen_jobids)}")


if __name__ == "__main__":
    main()
