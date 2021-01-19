# 3d_detection

## 진행 상황 (kitti, pvrcnn 구현 중)

- dataloader
  - 완료
- model (pvrcnn에 해당하는 것만 구현 중)
  - MeanVFE (완료)
  - VoxelBackBone8x (완료)
  - HeightCompression (완료)
  - BaseBEVBackbone (완료)
  - AnchorHeadSingle
  - AxisAlignedTargetAssigner
  - VoxelSetAbstraction
  - PointHeadSimple
  - PVRCNNHead
- optimizer
  - 미완료
- train_utils (train_model)
  - 완료

- evaluation
  - 미완료







- kitti_dataset.yaml 

  - 4번라인 point_cloud_range 의미
  - 47번라인 point_feature_encoding 의미

  - 55번라인 mask_points_and_boxes_outside_range 의미
    - masking된 points와 masking된 gt_boxe들만 사용하려고 걸러주는거?
  - 58번라인 shuffle_points
    - point들을 랜덤으로 섞은 후 다시 data_dict['points']에 저장
    - 왜 해주는거

