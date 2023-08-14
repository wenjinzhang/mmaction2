bash tools/dist_train.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py 2 --seed 0


python tools/train.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py --seed 0 --deterministic
python tools/train.py configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb.py --seed 0

nohup python tools/train.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py --seed 0 > log/train.txt 2>&1 &

# train on single GPU
nohup python tools/train.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py \
    --seed 0 > log/train_r50_8xb8-8x8x1-20e_ava21.txt 2>&1 &

# train on two GPU
nohup bash tools/dist_train.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py \
    2 --seed 0 > log/train_r50_8xb8-8x8x1-20e_ava21.txt 2>&1 &

nohup bash tools/dist_train.sh configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb_subset.py \
    2 --seed 0 > log/train_r50_8xb8-8x8x1-20e_trauma_subset.txt 2>&1 &

# no global feature

nohup bash tools/dist_train.sh configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb_subset_V2.py \
    2 --seed 0 > log/train_r50_8xb8-8x8x1-20e_trauma_subset_V2.txt 2>&1 &

# grobal_feature map
nohup bash tools/dist_train.sh configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb_subset_global.py \
    2 --seed 0 > log/train_r50_8xb8-8x8x1-20e_trauma_subset_global.txt 2>&1 &


nohup bash tools/dist_train.sh configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb_subset_global_V2.py \
    2 --seed 0 > log/train_r50_8xb8-8x8x1-20e_trauma_subset_globalV2.txt 2>&1 &

nohup bash tools/dist_train.sh configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb_subset_global_V2_NONONE.py 2 \
    --resume  --seed 0 > log/train_r50_8xb8-8x8x1-20e_trauma_subset_globalV2_NONONE_continue.txt 2>&1 &

nohup bash tools/dist_train.sh configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb_subset_V2_NONONE.py 2 \
    --seed 0 > log/train_r50_8xb8-8x8x1-20e_trauma_subset_V2_NONONE.txt 2>&1 &


# consider global feature map
nohup bash tools/dist_train.sh configs/app/slowfast/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_trauma-rgb_subset.py \
    2 --seed 0 > log/train_acrn_r50_8xb8-8x8x1-20e_trauma_subset.txt 2>&1 &

# test the SlowFast model on AVA2.1 and dump the result to a pkl file.***
python tools/test.py configs/app/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_trauma-rgb.py \
    checkpoints/epoch_20.pth --dump log/result.pkl


# Example: test the SlowFast model on AVA2.1 and dump the result to a pkl file.
python tools/test.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py \
    checkpoints/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth --dump log/result.pkl

nohup bash tools/dist_test.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py \
    checkpoints/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth 2 --dump log/result.pkl  > log/test_r50_8xb8-8x8x1-20e_ava21.txt 2>&1 &
    


data/ava/videos/xeGWXqSvC-8.webm

ffmpeg -ss 900 -t 901 -i ../../../data/ava/videos/xeGWXqSvC-8.webm -r 30 -strict experimental ../../../data/ava/videos_15min/xeGWXqSvC-8.webm
ffmpeg -i ../../../data/ava/videos_15min/xeGWXqSvC-8.webm -r 30 -q:v 1 ../../../data/ava/rawframes/xeGWXqSvC-8/img_%05d.jpg