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
  'nanjing-inter': 'NJ',
  'nanjing-intra': 'NJ',
  'haicgu': 'HAICGU',
  'leonardo': 'LEO',
  'local': 'local',
}

PARTITION_NAMES_MAP = {
  # HAICGU
  'ib': 'ib',
  'eth': 'eth',
  
  # Leonardo
  'boost_usr_prod': 'booster',
  'emulating_nanjing': 'nj',
  'emulating_haicgu': 'haicgu',
  
  # Nanjing
  'nanjing-inter': 'inter',
  'nanjing-intra': 'intra',
  
  # Debug
  'test': 'test',
  'other': 'other',
  
  # Other
  'NSLB': 'nslb',
  'plain': 'default',
}

