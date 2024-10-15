CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat_train.py \
    1 \
    --work-dir /home/aditya/grounding_dino_training_aug21 \
    --resume