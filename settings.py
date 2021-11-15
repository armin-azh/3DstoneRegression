from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from pathlib import Path
import torch

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
