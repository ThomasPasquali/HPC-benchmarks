# Leonardo Nodelists Generator

```bash
# Basic usage with sample data
python test_nodelists_generator.py

# Use Leonardo actual system topology
python test_nodelists_generator.py --csv leo_map.txt

# Skip queue tests (if squeue not available)
python test_nodelists_generator.py --csv leo_map.txt --no-queue

# Quiet mode (less verbose output)
python test_nodelists_generator.py --csv leo_map.txt --quiet
```