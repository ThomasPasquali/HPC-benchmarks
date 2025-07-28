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
  'pioneer':    [(64 * 1000,        'L1d'), (1 * 1000 * 1000,  'L2'),  (64 * 1000 * 1000,  'L3')],
  'bananaf3':   [(32 * 1000,        'L1d'), (500 * 1000     ,  'L2'),  (2* 500 * 1000,     'L2 + TCM')],
  'arriesgado': [(64 * 1000,        'L1d'), (2 * 1000 * 1000,  'L2'),  (0,                 '')        ],
  'brah':       [(4 * 1000 * 1000,  'L1d'), (64 * 1000 * 1000, 'L2'),  (512 * 1000 * 1000, 'L3')],
}

# Plots style
MARKERS_LIST = ["o", "s", "^", "d", "x", "P", "*", "v", ">"]
COLORS_CYCLE = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
LINESTYLES_CYCLE = itertools.cycle(["-", "--", "-.", ":"])