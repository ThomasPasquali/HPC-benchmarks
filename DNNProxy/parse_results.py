from pathlib import Path
import re
from statistics import geometric_mean, stdev
import pandas as pd
from typing import Dict, List
import sbatchman as sbm

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_stdout(job: sbm.Job) -> Dict[str, List[float]]:
  stdout = job.get_stdout()
  if stdout is None:
    raise Exception(f'Job stdout empty:\n{job}')
  
  model = None
  if 'GPT2-' in stdout:
    model = 'gpt2'
  elif 'Bert-' in stdout:
    model = 'bert'
  elif 'ResNet-152' in stdout:
    model = 'ResNet-152'
  elif 'DLRM ' in stdout:
    model = 'DLRM'
  elif 'ResNet-50' in stdout:
    model = 'ResNet-50'
      
  print(model)
      
  if model is None:
    raise Exception(f'Could not find the model in output of job: {job}\n{stdout}')
  
  lines = stdout.splitlines()
  times = {}

  if model in ['DLRM', 'ResNet-152']:
    lines = [lines[-1]]
    
  for line in lines:
    parts = line.split(', ') if ', ' in line else [line]
    _, time = parts[-1].split(' = ')
    time = float(time.split(' ')[0])
    
    m = model
    if model in ['ResNet-50']:
      if '(allreduce)' in line:
        m += '-allreduce'
      else:
        m += '-ring'

    if m not in times: times[m] = []
    times[m].append(time)

  return times

def main():
  jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)
  data = []

  for job in jobs:
    # print('='*50)
    # pprint(job)
    # print(job.get_stdout())
    # print('-'*50)
    
    res = parse_stdout(job)
    # print(res)
    for model, times in res.items():
      m = re.match(r'(\w+)_(\d+)nodes', job.config_name)
      # print(model)
      # print(times)
      data.append({
        'cluster': job.cluster_name,
        'partition': m.group(1),
        'nodes': int(m.group(2)),
        'model': model,
        'geomean_time': geometric_mean(times) if len(times) > 1 else times[0],
        'std_time': stdev(times) if len(times) > 1 else times[0],
        'max_time': max(times),
        'min_time': min(times),
      })

  df = pd.DataFrame(data)
  path = OUT_DIR / f'dnnproxies_{sbm.get_cluster_name()}_data.csv'
  df.to_csv(path)
  print(f'Data saved to {path.resolve().absolute()}')

if __name__ == "__main__":
  main()
