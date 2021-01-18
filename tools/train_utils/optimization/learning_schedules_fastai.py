import math
from functools import partial

import numpy as np
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper