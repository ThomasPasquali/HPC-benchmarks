import itertools
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
OUT_DIR = Path('results')

# Default font sizes
FONT_TITLE = 38
FONT_AXES = 28
FONT_TICKS = 20
FONT_LEGEND = 16

# Name maps
BOARD_NAMES_MAP = {
  'brah': 'AMD EPYC 7742',
  'baldo': 'AMD EPYC 7742',
  'pioneer': 'Milk-V Pioneer',
  'bananaf3': 'Banana Pi F3',
  'arriesgado': 'HiFive Unmatched',
}
BOARD_SHORT_NAMES_MAP = {
  'brah': 'AMD',
  'baldo': 'AMD',
  'pioneer': 'Pioneer',
  'bananaf3': 'BananaPi',
  'arriesgado': 'HiFive',
}

# Hard-coded data
CACHE_SIZES = {
  'pioneer':    [(64 * 1024,        'L1d'), (1 * 1024 * 1024,  'L2'),  (64 * 1024 * 1024,  'L3')],
  'bananaf3':   [(32 * 1024,        'L1d'), (500 * 1024     ,  'L2'),  (2* 500 * 1024,     'L2 + TCM')],
  'arriesgado': [(64 * 1024,        'L1d'), (2 * 1024 * 1024,  'L2'),  (0,                 '')        ],
  'brah':       [(4 * 1024 * 1024,  'L1d'), (64 * 1024 * 1024, 'L2'),  (0,                 '')        ],# (512 * 1024 * 1024, 'L3')],
  'baldo':      [(4 * 1024 * 1024,  'L1d'), (64 * 1024 * 1024, 'L2'),  (0,                 '')        ],# (512 * 1024 * 1024, 'L3')],
}

CLUSTER_NAMES_MAP = {
  'nanjing': 'NJ',
  'haicgu': 'HAICGU',
  'leonardo': 'LEO',
}

PARTITION_NAMES_MAP = {
  'ib': 'ib',
  'eth': 'eth',
  'NSLB': 'nslb',
  'plain': 'default',
}

# Plots style
MARKERS_LIST = ["o", "s", "^", "d", "x", "P", "*", "v", ">"]
COLORS_LIST = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
COLORS_CYCLE = itertools.cycle(COLORS_LIST)
LINESTYLES_LIST = ["-", ":", "-.", "--"]
LINESTYLES_CYCLE = itertools.cycle(LINESTYLES_LIST)

SET_FIG_TITLE = False