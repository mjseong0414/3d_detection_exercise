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

def create_logger(log_file = None, rank = 0, log_level = logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR') # logger level을 logging.INFO로 지정. 이제 logger 객체는 INFO 이상의 메시지를 출력할 수 있음. (레벨은 총 5게, INFO는 2단계)
    formatter = logging.Formatter('%(asctime)s %(levelname)5s %(message)s') # 앞으로 모든 log에 time, level, message를 추가해서 출력
    console = logging.StreamHandler() # console에 log를 남기고 싶을 때
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file) # log를 계속 쌓고 싶을 때
        file_handler.setLevel(log_file if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
