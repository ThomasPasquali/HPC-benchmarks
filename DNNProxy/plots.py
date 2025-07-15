# [tpasquali@cn19 DNNProxy]$ mpirun -np 4 resnet 
# Rank = 2, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141914 s
# Rank = 3, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141914 s
# Rank = 0, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141915 s
# Rank = 1, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141915 s
# Rank = 2, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084589 s
# Rank = 3, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084606 s
# Rank = 0, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084609 s
# Rank = 1, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084592 s
    
# [tpasquali@cn19 DNNProxy]$ mpirun -np 6 bert 24 6
# Rank = 0, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018508 s
# Rank = 3, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018509 s
# Rank = 2, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018509 s
# Rank = 1, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018511 s
# Rank = 4, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018513 s
# Rank = 5, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018513 s

# [tpasquali@cn19 DNNProxy]$ mpirun -np 4 gpt2 
# Rank = 0, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010844 s
# Rank = 1, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010848 s
# Rank = 2, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010854 s
# Rank = 3, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010854 s

# [tpasquali@cn19 DNNProxy]$ mpirun -np 2 bert 24 2
# Rank = 0, world_size = 2, layers = 24, stages = 2, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.007392 s
# Rank = 1, world_size = 2, layers = 24, stages = 2, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.007392 s

import pprint
import sbatchman as sbm
from typing import List
import re

# PATTERN = r'(\w+(?: \w+)*?) = ([^,]+)'

# GPT2_TIME_KEY = "GPT2-large pipeline and data parallelism runtime for each iteration"
# BERT_TIME_KEY = "Bert-large pipeline and data parallelism runtime for each iteration"
# RESNET_RING_TIME_KEY = "ResNet-50 data parallelism (neighbors in ring) runtime for each iteration"
# RESNET_ALLRED_TIME_KEY = "ResNet-50 data parallelism (allreduce) runtime for each iteration"

# KEYS_DICT = {
#   'gpt2': GPT2_TIME_KEY,
#   'bert': BERT_TIME_KEY,
#   'resnet': RESNET_RING_TIME_KEY, # FIXME
# }

def parse_stdout(job: sbm.Job):
  lines = job.get_stdout().strip().split('\n')
  times = []
  for line in lines[1:]:
    # matches = re.findall(PATTERN, line)
    # result = {k.strip(): v.strip() for k, v in matches}
    # times.append(result[KEYS_DICT[job.tag]])
    parts = line.split(', ')
    print(parts)
    times.append(float(parts[-1].split(' = ')[1].split(' ')[0])) # TODO fix resnet
  return times

def filter_jobs(jobs: List[sbm.Job]) -> List[sbm.Job]:
  filtered_jobs = []
  for job in jobs:
    if job.status in ['COMPLETED']:
      filtered_jobs.append(job)
  return filtered_jobs


def main():
  jobs = filter_jobs(sbm.jobs_list(from_active=True, from_archived=True))

  for job in jobs:
    pprint.pprint(job)
    print(parse_stdout(job))

if __name__ == "__main__":
  main()
