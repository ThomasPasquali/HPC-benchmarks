#!/usr/bin/env python3
"""
SLURM Queue Monitor - Periodically runs squeue and logs ALL jobs with snapshot ID
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
        description="Monitor SLURM queue and log ALL jobs at each iteration with snapshot ID"
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
    file_initialized = False
    squeue_id = 0
    
    print(f"Starting SLURM queue monitor (logging ALL jobs at each iteration)...")
    print(f"Interval: {args.interval} seconds")
    print(f"Output file: {output_file}")
    if args.format:
        print(f"Format: {args.format}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            current_output = run_squeue(args.format)
            
            if current_output:
                squeue_id += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rows = parse_squeue_output(current_output)
                
                if rows:
                    # Add squeue_id and timestamp to each row
                    for row in rows:
                        row['squeue_id'] = squeue_id
                        if args.add_timestamp:
                            row['Timestamp'] = timestamp
                    
                    # Reorder so squeue_id is first
                    fieldnames = ['squeue_id']
                    if args.add_timestamp:
                        fieldnames.append('Timestamp')
                    fieldnames.extend([k for k in rows[0].keys() if k not in ['squeue_id', 'Timestamp']])
                    
                    # Write to CSV
                    mode = 'w' if not file_initialized else 'a'
                    with open(output_file, mode, newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        
                        # Write header only once
                        if not file_initialized:
                            writer.writeheader()
                            file_initialized = True
                        
                        writer.writerows(rows)
                    
                    print(f"[{timestamp}] Snapshot {squeue_id}: {len(rows)} job(s) logged")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped by user")
        print(f"Total snapshots taken: {squeue_id}")


if __name__ == "__main__":
    main()
