# DLHelper

## Introduction
- A self-customized functions for deep learning
- `dataset`: splittable by parsing ratio
- `main`: supervised training
- `visualize`
    - `t_sne`
    - `grad_cam`

## Example

### Finetune Pretrained weights
```shell=
python3 -m dlhelper.main \
        --dataset food11 --base-dir /path/to/base \
        --pretrained /path/to/your/pretrained/weights \
        --checkpoint /path/to/where/ckpt/is/saved \
        --batch-size 256 \
        --lr 3e-4 \
        --num-workers 4 \
        --finetune
```

### T-SNE
```shell=
python3 -m dlhelper.visualize.tsne \
    --dataset dataset_name --base-dir /path/to/base \
    --pretrained checkpoint_dir \
    --result ./checkpoints/t_sne.png
```

### Grad-CAM
```shell=
python3 -m dlhelper.visualize.grad_cam \
    --dataset dataset_name --base-dir /path/to/base \
    --pretrained checkpoint_dir \
    --result ./checkpoints/grad_cam.png
```
