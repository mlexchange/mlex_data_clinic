# MLExchange Data Clinic

## Description
This app provides a training/testing platform for latent space exploration with
unsupervised deep-learning approaches.

## Running as a Standalone Application (Using Docker)

The **Prefect server, Tiled server and the application** are all defined within a **single Docker Compose file**. Each service runs in its own Docker container, simplifying the setup process while maintaining modularity.

However, the **Prefect worker** must be run separately on your local machine (refer to step 5).

## Steps to Run the Application

### 1 Clone the Repository

```sh
git clone https://github.com/mlexchange/mlex_data_clinic.git
cd mlex_data_clinic
```

### 2 Configure Environment Variables

Create a `.env` file using `.env.example` as a reference:

```sh
cp .env.example .env
```

Then **update the** `.env` file with the correct values.

**Important Note:** Due to the current tiled configuration, ensure that the `WRITE_DIR` is a subdirectory of the `READ_DIR` if the same tiled server is used for both reading data and writing results.

#### MLFlow Configuration in .env

When setting `MLFLOW_TRACKING_URI` in the `.env` file:

- If you run the [MLFlow server](https://github.com/xiaoyachong/mlex_mlflow) locally, you can set it to:
  ```
  MLFLOW_TRACKING_URI="http://mlflow-server:5000"
  ```
  This works because the MLFlow server also runs in the `mle_net` Docker network.

- If you run MLFlow server on vaughan and use SSH port forwarding:
  ```
  ssh -S forward -L 5000:localhost:5000 <your-username>@vaughan.als.lbl.gov
  ```
  Then you can set it to:
  ```
  MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
  ```

You also need to set  `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in the `.env` file and modify the admin_username and admin_password in `basic_auth.ini` as well.

Create a `basic_auth.ini` file using `basic_auth.ini.example` as a reference:

```sh
cp basic_auth.ini.example basic_auth.ini
```


### 3 Build and Start the Application

#### 3.1 Algorithm Registry Setup in MLFlow

Before starting the application, you need to register your algorithms in MLflow. This is a one-time setup process:

1. Start only the MLflow services:
   ```sh
   docker compose up -d mlflow mlflow_db
   ```

2. Wait a few seconds for MLflow to initialize, then register the algorithms:
   ```sh
   cd scripts
   python save_mlflow_algorithms.py
   ```
   
   This script will:
   - Connect to the MLflow server
   - List any existing algorithms
   - Register all algorithms from the JSON file specified by the `ALGORITHM_JSON_PATH` environment variable
   - Show the registration status for each algorithm

   > **Note:** By default, `ALGORITHM_JSON_PATH` points to `./all_models.json`, which is the combination of the models defined in Data Clinic and Latent Space Explorer. You can customize this by setting the environment variable in your `.env` file.

#### 3.2 Start the Full Application

After successfully registering the algorithms, you can start the complete application:

```sh
docker compose up -d
```

This command will:
- Start all services defined in your docker-compose.yml
- Run the containers in the background (detached mode)
- Use the algorithms registered in MLflow

### 4 Verify Running Containers

```sh
docker ps
```

### 5 Start a Prefect Worker

Open another terminal and start a Prefect worker. Refer to [mlex_prefect_worker](https://github.com/mlexchange/mlex_prefect_worker) for detailed instructions on setting up and running the worker.


### 6 Access the Application

Once the container is running, open your browser and visit:
* **Dash app:** http://localhost:8072/

### 7 Stopping the Application

To stop and remove the running containers, use:

```sh
docker compose down
```

This will **shut down all services** but **retain data** if volumes are used.


## Model Description
**pytorch_autoencoder:** User-defined autoencoders implemented in [PyTorch](https://pytorch.org).

Further information can be found in [mlex_pytorch_autoencoders](https://github.com/mlexchange/mlex_pytorch_autoencoders/tree/main).

## Developer Setup
If you are developing this library, there are a few things to note.

1. Install development dependencies:

```
pip install .
pip install ".[dev]"
```

2. Install pre-commit
This step will setup the pre-commit package. After this, commits will get run against flake8, black, isort.

```
pre-commit install
```

3. (Optional) If you want to check what pre-commit would do before commiting, you can run:

```
pre-commit run --all-files
```

4. To run test cases:

```
python -m pytest
```

# Copyright
MLExchange Copyright (c) 2024, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
