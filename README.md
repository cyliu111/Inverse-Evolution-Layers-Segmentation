# Inverse Evolution Layers for Image Segmentation

This project explores the application of Inverse Evolution Layers (IELs) to enhance the regularization of neural network outputs in image segmentation tasks. It includes implementations of both heat-diffusion IELs and curve-motion based IELs.

## Data Preparation

Before running the code, ensure you have downloaded the datasets required for the experiments. Modify the data paths in `train.py` to match your local dataset locations.

The datasets are organized into four folders corresponding to different datasets:

- White Blood Cell (WBC)
- 2018 Data Science Bowl (2018 DSB)
- Digital Retinal Images for Vessel Extraction (DRIVE)
- Retinal Fundus Glaucoma Challenge (REFUGE)

## How to Run the Code

To train the original UNet model on noisy labels:
```bash
python train.py --add_noise

To train the original UNet model using IELs on noisy labels:
```bash
python train.py --add_noise --use_iels
