from functools import partial

import numpy as np

from ...utils import box_utils, common_utils

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config = cur_cfg)
            self.data_processor_queue.append(cur_processor)
        
    
    def mask_points_and_boxes_outside_range(self, data_dict = None, config = None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config = config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range) # mask point 얻기
        data_dict['points'] = data_dict['points'][mask] # mask에 해당하는 points만 저장
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_points_and_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners = config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict
    
    def shuffle_points(self, data_dict = None, config = None):
        # points를 random으로 섞은 후 다시 data_dict['points']에 저장 이거 왜 쓰지
        if data_dict is None:
            return partial(self.shuffle_points, config=config)
        
        if config.SHUFFLE_ENABLED(self.mode):
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict
    
    def transform_poits_to_voxels(self, data_dict = None, config = None, voxel_generator = None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator
            
            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_poits_to_voxels, voxel_generator=voxel_generator)
        
        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output
        
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]
        
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict