# Unet with Diffusion IELs on White Blood Cell Dataset

## Data preparation
1. Download the dataset from https://github.com/zxaoyou/segmentation_WBC 

2. Put the downloaded "Dataset 1" under the directory "./data"

## Training 
1. Unet without IELs on noisy labels (20 epochs)
```bash
python train.py --add_noise
```
Segmentation maps on validation set will be saved in "./predictions"
2. Unet with IELs on noisy labels (20 epochs)
```bash
python train.py --add_noise --use_iels
```
Segmentation maps on validation set will be saved in "./predictions"