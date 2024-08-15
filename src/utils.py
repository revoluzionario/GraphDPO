import os
import shutil
import logging
import random
import sys
import re
import datetime
import time
from copy import deepcopy
import itertools
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.backends.cudnn
from torch.nn.init import xavier_normal_
import numpy as np
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable
from advertorch.attacks import L2PGDAttack
from torch.cuda.amp import autocast, GradScaler

def get_datetime():
    """ Get current time """
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def set_seeds(seed):
    """ Set random seeds """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    pass