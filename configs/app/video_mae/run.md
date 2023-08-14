# test script
```
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

1. test vit-large box only

```
python tools/test.py configs/app/video_mae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_subset.py \
work_dirs/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_subset/best_mAP_overall_epoch_97.pth \
--dump result.pkl \
--show \
--show-dir "./visualization_result"
```