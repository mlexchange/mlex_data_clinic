# How To Guide

Data Clinic is a browser-based framework to train and test deep-learning based approaches for 
latent space exploration.

## Data Format
Currently, Data Clinic supports directory based data definition, similar to the following
example:

```
data_directory
│
│   image001.jpeg
│   image002.jpeg
│   ...

```

The supported image formats are: TIFF, TIF, JPG, JPEG, and PNG.

## Training
To train a model, please follow the following steps:

1. Choose your dataset.
   1. As a standalone application: Click on "Open File Manager", and upload your dataset 
   as a ZIP file OR choose your `data_directory` from the table (click on "Browse",
   choose a row from the table directory, and click "Import").
   2. From [Label Maker](https://github.com/mlexchange/mlex_dash_labelmaker_demo): The 
   dataset you uploaded in Label Maker should be visible in Data Clinic by default at start-up.
   When selecting a different dataset in Label Maker after start-up, you can refresh the 
   dataset in Data Clinic by clicking "Refresh Images".
2. Choose "Model Training" in Actions.
3. Modify the [training parameters](./concepts.md) as needed.
4. Click Execute.
5. Choose the computing resources that should be used for this task. Please note that 
these should not exceed the constraints defined in the [computing API](https://github.com/mlexchange/mlex_computing_api).
Recommended values: CPU - 4 and GPU - 0. Click "Submit".
6. The training job has been successfully submitted! You can check the progress of this
job in the "List of Jobs", where you can select the corresponding row to display the loss
plot in real-time. Additionally, you can check the logs and parameters of each job by 
clicking on it's corresponding cells.

## Testing
To test a mode, please follow the following steps:

1. Choose your dataset.
   1. As a standalone application: Click on "Open File Manager", and upload your dataset 
   as a ZIP file OR choose your `data_directory` from the table (click on "Browse",
   choose a row from the table directory, and click "Import").
   2. From [Label Maker](https://github.com/mlexchange/mlex_dash_labelmaker_demo): The 
   dataset you uploaded in Label Maker should be visible in MLCoach by default at start-up.
   When selecting a different dataset in Label Maker after start-up, you can refresh the 
   dataset in MLCoach by clicking "Refresh Images".
2. Choose "Test Prediction using Model" in Actions.
3. Modify the [testing parameters](./concepts.md) as needed.
4. Choose a trained model from the "List of Jobs".
5. Click Execute.
6. Choose the computing resources that should be used for this task. Please note that 
these should not exceed the constraints defined in the [computing API](https://github.com/mlexchange/mlex_computing_api).
Recommended values: CPU - 4 and GPU - 0. Click "Submit".
7. The testing job has been successfully submitted! You can check the progress of this
job in the "List of Jobs", where you can select the corresponding row to display the 
reconstructed images in real-time. Additionally, you can check the logs and parameters 
of each job by clicking on it's corresponding cells.
