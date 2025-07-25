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

# Hard-coded data
CACHE_SIZES = {
  'pioneer':    [(64 * 1024, 'L1 cache'), (1 * 1024 * 1024, 'L2 cache'),  (64 * 1024 * 1024, 'L3 cache')],
  'bananaf3':   [(32 * 1024, 'L1 cache'), (512 * 1024     , 'L2 cache'),  (512 * 1024,       'TCM')     ],
  'arriesgado': [(64 * 1024, 'L1 cache'), (2 * 1024 * 1024, 'L2 cache'),  (0,                '')        ],
}

# Plots style
MARKERS_LIST = ["o", "s", "^", "d", "x", "P", "*", "v", ">"]
COLORS_CYCLE = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
LINESTYLES_CYCLE = itertools.cycle(["-", "--", "-.", ":"])