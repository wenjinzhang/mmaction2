python tools/train.py configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
    --cfg-options randomness.seed=0


# train on two GPU
nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
    2 --cfg-options randomness.seed=0 > log/train_vit-base-p16_videomae.txt 2>&1 &

# trauma dataset
nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb.py \
    2 --cfg-options randomness.seed=0 > log/train_vit-base-p16_trauma_videomae.txt 2>&1 &

# global feature
nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global.py \
    2 --cfg-options randomness.seed=0 > log/train_vit-base-p16_trauma_videomae_global.txt 2>&1 &


# global feature
nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global_NONONE.py 2 \
    --resume work_dirs/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global_NONONE/best_mAP_overall_epoch_16.pth --cfg-options randomness.seed=0 > log/train_vit-base-p16_trauma_videomae_global_NONONE_epoch35.txt 2>&1 &



nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global_NONONE_subset.py 2 \
    --resume work_dirs/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global_NONONE_subset/best_mAP_overall_epoch_41.pth --cfg-options randomness.seed=0 > log/train_vit-base-p16_trauma_videomae_global_NONONE_subset_continue.txt 2>&1 &


nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global_NONONE_subset_roiout8.py 2 \
    --cfg-options randomness.seed=0 > log/train_vit-base-p16_trauma_videomae_global_NONONE_subset_roiout8.txt 2>&1 &

# VIT large
nohup bash tools/dist_train.sh configs/app/video_mae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_global_subset.py 2 \
    --cfg-options randomness.seed=0 > log/train_vit-large-p16_trauma_videomae_global_subset.txt 2>&1 &


nohup bash tools/dist_train.sh configs/app/video_mae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_subset.py 2 \
    --resume work_dirs/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_subset/best_mAP_overall_epoch_97.pth --cfg-options randomness.seed=0 > log/train_vit-large-p16_trauma_videomae_subset_cont98.txt 2>&1 &


# box feature only
nohup bash tools/dist_train.sh configs/app/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_NONONE_subset.py 2 \
     --cfg-options randomness.seed=0 > log/train_vit-base-p16_trauma_videomae_NONONE_subset.txt 2>&1 &


# check model
