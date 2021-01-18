import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils
from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset

__all__ = {
    'DatasetTemplate' = DatasetTemplate,
    'KittiDataset' = KittiDataset,
}

class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas = None, rank = None, shuffle = True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator() # psedo random number로 알고리즘의 state를 관리하는 generator object 생성
            g.manual_seed(self.epoch) # generating random number setting
            indices = torch.randperm(len(self.dataset), generator=g).tolist() # len(self.dataset) - 1 까지의 수를 g를 통해 랜덤하게 나열.
        else:
            indices = torch.arange(len(self.dataset)).tolist() # ex) torch.arange(5) => tensor([ 0,  1,  2,  3,  4])

        indices += indices[:(self.total_size - len(indices))]
        
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path = None, workers = 4,
                    logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):
    
    # 바로 KittiDataset 클래수 실행됨.
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg = dataset_cfg,
        class_names = class_names,
        root_path = root_path,
        training = training,
        logger = logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch') # dataset에 'merge_all_iters_to_one_epoch' 이 있나 확인. 있으면 True
        dataset.merge_all_iters_to_one_epoch(merge = True, epoch = total_epochs)
    
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, workd_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, workd_size, rank, shuffle=False)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory = True, num_workers = workers,
        shuffle = (sampler is None) and training, collate_fn = dataset.collate_batch,
        drop_last = False, sampler = sampler, timeout = 0
    )

    return dataset, dataloader, sampler