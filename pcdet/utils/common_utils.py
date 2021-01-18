import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_dist_slurm(tcp_port, local_rank, backend = 'nccl'):
    