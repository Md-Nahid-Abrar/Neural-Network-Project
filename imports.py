import os, random, math, zipfile, urllib.request, warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from collections import Counter, defaultdict
from pathlib import Path
from IPython.display import Audio, display, HTML
import subprocess, shutil

# Reproducibility
SEED = 42
random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Imports complete.  Device: {DEVICE}")