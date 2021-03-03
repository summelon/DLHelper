# DLHelper

## Introduction
- A self-customized functions for deep learning
- `dataset`: splittable by parsing ratio
- `main`: supervised training
- `visualize`
    - `t_sne`
    - `grad_cam`

## Example
### T-SNE
```python=
cd dlhelper
python3 -m visualize.tsne \
    --dataset dataset_name \
    --pretrained checkpoint_dir \
    --result ../checkpoints/t_sne.png
```

### Grad-CAM
```python=
cd dlhelper
python3 -m visualize.grad_cam \
    --dataset dataset_name \
    --pretrained checkpoint_dir \
    --result ../checkpoints/grad_cam.png
```

