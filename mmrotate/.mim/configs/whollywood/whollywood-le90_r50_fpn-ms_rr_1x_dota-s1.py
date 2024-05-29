_base_ = ['point2rbox-hdr-hbox-dota.py']

model = dict(
    bbox_head=dict(
        use_query_mode=True, 
        query_mode_output_path='data/split_ms_dota/trainval/pseudo_labels',))

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RBox2Point', dummy=48),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.RandomShift', prob=0.5, max_shift_px=16),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict())

data_root = 'data/split_ms_dota/'

train_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
