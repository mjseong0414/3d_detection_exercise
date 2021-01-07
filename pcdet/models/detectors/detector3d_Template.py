import os

import torch
import torch.nn as nn

class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__():
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero(1))

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step():
        self.global_step += 1
    
    # build_networks for문이 돌면서 모듈 함수들 실행
    # => config 파일에 안 적었으면 mudule은 None이 return됨
    def build_networks(self):
        model_info_dict = {
            'module_list' : [],
            'num_rawpoint_features' : self.dataset.point_feature_encoder.num_point_features,
            'num_point_features' :  self.dataset.point_feature_encoder.num_point_features,
            'grid_size' : self.dataset.grid_size,
            'point_cloud_range' : self.dataset.point_cloud_range,
            'voxel_size' : self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            # getattr(self, 'build_vfe)(model_info_dict=model_info_dict) => build_vfe 함수에 model_info_dict를 넣은 뒤 값을 return받는다.
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list'] # 사용할 모듈 리스트들 저장
    
    def build_vfe(self, model_info_dict):
        # get('VFE', None) => VFE key를 찾는데 없으면 None을 반환
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME]
