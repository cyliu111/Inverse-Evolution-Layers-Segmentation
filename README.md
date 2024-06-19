# Inverse Evolution Layers for Image Segmentation

This project explores the application of Inverse Evolution Layers (IELs) to enhance the regularization of neural network outputs in image segmentation tasks. It includes implementations of both heat-diffusion IELs and curve-motion based IELs.

## Data Preparation
The datasets involved in our experiments are:
- White Blood Cell (WBC)
- 2018 Data Science Bowl (2018 DSB)
- Digital Retinal Images for Vessel Extraction (DRIVE)
- Retinal Fundus Glaucoma Challenge (REFUGE)

Before running the code, ensure you have downloaded the datasets required for the experiments. The codes for the four datasets are organized into four different folders. Modify the data paths in `train.py` to match your local dataset locations.

## How to Run the Code
Make sure all dependencies are installed and correctly configured before running the commands.
Modify other hyperparameters and configurations directly in `train.py` according to your experimental setup.
Navigate to Project Directory where `train.py` is located. For example:
```
cd /path/to/your/project/unet_iels_wbc
```

To train the original UNet model on noisy labels:
```
python train.py --add_noise
```

To train the original UNet model using IELs on noisy labels:
```
python train.py --add_noise --use_iels
```
