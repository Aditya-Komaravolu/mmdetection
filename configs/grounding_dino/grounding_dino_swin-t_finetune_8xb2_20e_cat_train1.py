_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

# data_root = '/home/aditya/snaglist_sem_aug20/'
data_root = '/home/aditya/snaglist_dataset_aug12/'



# class_name = ("cement_slurry", "chipping", "honeycomb", "incomplete_deshuttering",)
class_name = ("cement_slurry", "honeycomb",)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(5, 85, 19), (6, 64, 148), (89, 120, 115), (178, 29, 226)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    classwise=True
)
test_evaluator = val_evaluator

max_iters = 150000

log_processor = dict(by_epoch=False)

default_hooks = dict(
    checkpoint=dict(save_best='auto',by_epoch=False, interval=1000),
    logger=dict(type='LoggerHook', interval=100)
    )
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,  
    val_interval=1000) 
# val_cfg = dict(type='ValLoop') 

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=max_iters),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[50000, 100000, 125000],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)


# evaluation = dict(interval=50, metric=['bbox'], save_best='bbox_mAP')


# evaluation = dict(interval=1000, metric=['mAP', 'mAP_50'], save_best='mAP_50')

# vis_backends = [
#     # dict(type='LocalVisBackend'),
#     dict(type='TensorboardVisBackend')
# ]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# log_config = dict(
#     interval=100,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='TensorboardLoggerHook')
#     ])