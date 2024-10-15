_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py',
]

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'  # noqa
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth'
model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=1024,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
)
