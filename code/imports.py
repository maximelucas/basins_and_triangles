from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm

import xgi

sb.set_theme(style="ticks", context="notebook")

results_dir = "results/"

Path(results_dir).mkdir(parents=True, exist_ok=True)