# How To Guide

Data Clinic is a browser-based framework to train and test deep-learning based approaches for
latent space exploration.

## Data Format
Currently, Data Clinic supports data access through [Tiled](https://blueskyproject.io/tiled/) or filesystem. If accessing files through filesystem, the data is expected as shown below:

```
data_directory
│
│   image001.jpeg
│   image002.jpeg
│   ...

```

The supported image formats are: TIFF, TIF, JPG, JPEG, and PNG.

To train a model from a Dash application, please follow the following steps:

1. Choose your dataset. Click on "Open File Manager", and choose your dataset.
2. Modify the [training parameters](./concepts.md) as needed.
3. Click Train/Run.
4. The training job has been successfully submitted! You can check the progress of this job in dropdown `Trained Jobs`, where you can select the corresponding job to display the training stats and/or logs.

## Inference
To run inference, please follow the following steps:

1. Choose your dataset. Click on "Open File Manager", and choose your dataset.
2. Modify the [inference parameters](./concepts.md) as needed.
4. Choose a trained model from the dropdown `Trained Jobs`.
4. Click Inference/Run.
5. The inference job has been successfully submitted! You can check the progress of this job in dropdown `Inference Jobs`, where you can select the corresponding job to display the results and/or logs.
