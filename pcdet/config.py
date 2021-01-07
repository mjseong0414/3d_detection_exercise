from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
    # cfg는 dataset.yaml
    for key, val in cfg.items(): # cfg가 dict 자료형임. items()로 key와 value를 따로 호출
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))

def cfg_from_yaml_file(cfg_file, config):
    # cfg_file = 모델.yaml파일 경로
    # config = {'ROOT_DIR': PosixPath('/home/minjae/OpenPCDet'), 'LOCAL_RANK': 0} 아직 이거만 쓰여있음
    with open(cfg_file, 'r') as f: 
    # cfg_file을 read 모드로 열고 new_config변수에 저장한 뒤 기존 config에 내용 추가
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader) # yaml.load => yaml을 파일을 파싱해서 파이썬 객체로 읽어오기.
        except:
            new_config = yaml.load(f)
        
        merge_new_config(config=config, new_config=new_config)
    
    return config

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config)) # .update => dict 자료형에서 같은 key이면 value를 업데이트하고 key가 없으면 추가하는 함수
    
    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    
    return config

# 밑의 세 줄은 import 할 때 실행됨.
cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0