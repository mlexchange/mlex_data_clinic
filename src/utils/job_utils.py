import copy
import json
import os
from urllib.parse import urljoin

# I/O parameters for job execution
READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", None)
WRITE_DIR_MOUNT = os.getenv("WRITE_DIR_MOUNT", None)
WRITE_DIR = os.getenv("WRITE_DIR", "")
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

# Flow parameters
PARTITIONS_CPU = json.loads(os.getenv("PARTITIONS_CPU", "[]"))
RESERVATIONS_CPU = json.loads(os.getenv("RESERVATIONS_CPU", "[]"))
MAX_TIME_CPU = os.getenv("MAX_TIME_CPU", "1:00:00")
PARTITIONS_GPU = json.loads(os.getenv("PARTITIONS_CPU", "[]"))
RESERVATIONS_GPU = json.loads(os.getenv("RESERVATIONS_CPU", "[]"))
MAX_TIME_GPU = os.getenv("MAX_TIME_CPU", "1:00:00")
SUBMISSION_SSH_KEY = os.getenv("SUBMISSION_SSH_KEY", "")
FORWARD_PORTS = json.loads(os.getenv("FORWARD_PORTS", "[]"))
CONTAINER_NETWORK = os.getenv("CONTAINER_NETWORK", "")


def parse_tiled_url(url, user, project_name, tiled_base_path="/api/v1/metadata"):
    """
    Given any URL (e.g. http://localhost:8000/results),
    return the same scheme/netloc but with path='/api/v1/metadata'.
    """
    if tiled_base_path not in url:
        url = urljoin(url, os.path.join(tiled_base_path, user, project_name))
    else:
        url = urljoin(url, f"/{user}/{project_name}")
    return url


def parse_train_job_params(
    data_project,
    model_parameters,
    user,
    project_name,
    flow_type,
    latent_space_params,
    dim_reduction_params,
):
    """
    Parse training job parameters
    """
    # TODO: Use model_name to define the conda_env/algorithm to be executed
    data_uris = [dataset.uri for dataset in data_project.datasets]

    results_dir = f"{WRITE_DIR}/{user}"

    io_parameters = {
        "uid_retrieve": "",
        "data_uris": data_uris,
        "data_tiled_api_key": data_project.api_key,
        "data_type": data_project.data_type,
        "root_uri": data_project.root_uri,
        "models_dir": f"{results_dir}/models",
        "results_tiled_uri": parse_tiled_url(RESULTS_TILED_URI, user, project_name),
        "results_tiled_api_key": RESULTS_TILED_API_KEY,
        "results_dir": f"{results_dir}",
    }

    ls_python_file_name_train = latent_space_params["python_file_name"]["train"]
    ls_python_file_name_inference = latent_space_params["python_file_name"]["inference"]
    dm_python_file_name = dim_reduction_params["python_file_name"]

    if flow_type == "podman" or "docker":
        job_params = {
            "flow_type": flow_type,
            "params_list": [
                {
                    "image_name": latent_space_params["image_name"],
                    "image_tag": latent_space_params["image_tag"],
                    "command": f"python {ls_python_file_name_train}",
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/tiled_storage",
                    ],
                    "network": CONTAINER_NETWORK,
                },
                {
                    "image_name": latent_space_params["image_name"],
                    "image_tag": latent_space_params["image_tag"],
                    "command": f"python {ls_python_file_name_inference}",
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/tiled_storage",
                    ],
                    "network": CONTAINER_NETWORK,
                },
                {
                    "image_name": dim_reduction_params["image_name"],
                    "image_tag": dim_reduction_params["image_tag"],
                    "command": f"python {dm_python_file_name}",
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": {
                            "n_components": 2,
                            "min_dist": 0.1,
                            "n_neighbors": 5,
                        },
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/tiled_storage",
                    ],
                    "network": CONTAINER_NETWORK,
                },
            ],
        }

    elif flow_type == "conda":
        job_params = {
            "flow_type": "conda",
            "params_list": [
                {
                    "conda_env_name": latent_space_params["conda_env"],
                    "python_file_name": ls_python_file_name_train,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
                {
                    "conda_env_name": latent_space_params["conda_env"],
                    "python_file_name": ls_python_file_name_inference,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
                {
                    "conda_env_name": dim_reduction_params["conda_env"],
                    "python_file_name": dm_python_file_name,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": {
                            "n_components": 2,
                            "min_dist": 0.1,
                            "n_neighbors": 5,
                        },
                    },
                },
            ],
        }

    else:
        job_params = {
            "flow_type": "slurm",
            "params_list": [
                {
                    "job_name": "latent_space_explorer",
                    "num_nodes": 1,
                    "partitions": PARTITIONS_CPU,
                    "reservations": RESERVATIONS_CPU,
                    "max_time": MAX_TIME_CPU,
                    "conda_env_name": latent_space_params["conda_env"],
                    "python_file_name": ls_python_file_name_train,
                    "submission_ssh_key": SUBMISSION_SSH_KEY,
                    "forward_ports": FORWARD_PORTS,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
                {
                    "job_name": "latent_space_explorer",
                    "num_nodes": 1,
                    "partitions": PARTITIONS_CPU,
                    "reservations": RESERVATIONS_CPU,
                    "max_time": MAX_TIME_CPU,
                    "conda_env_name": latent_space_params["conda_env"],
                    "python_file_name": ls_python_file_name_inference,
                    "submission_ssh_key": SUBMISSION_SSH_KEY,
                    "forward_ports": FORWARD_PORTS,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
                {
                    "job_name": "latent_space_explorer",
                    "num_nodes": 1,
                    "partitions": PARTITIONS_CPU,
                    "reservations": RESERVATIONS_CPU,
                    "max_time": MAX_TIME_CPU,
                    "conda_env_name": dim_reduction_params["conda_env"],
                    "python_file_name": dm_python_file_name,
                    "submission_ssh_key": SUBMISSION_SSH_KEY,
                    "forward_ports": FORWARD_PORTS,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": {
                            "n_components": 2,
                            "min_dist": 0.1,
                            "n_neighbors": 5,
                        },
                    },
                },
            ],
        }

    return job_params


def parse_inference_job_params(
    data_project,
    model_parameters,
    user,
    project_name,
    flow_type,
    latent_space_params,
    dim_reduction_params,
):
    """
    Parse inference job parameters
    """
    # TODO: Use model_name to define the conda_env/algorithm to be executed
    data_uris = [dataset.uri for dataset in data_project.datasets]

    results_dir = f"{WRITE_DIR}/{user}"

    io_parameters = {
        "uid_retrieve": "",
        "data_uris": data_uris,
        "data_tiled_api_key": data_project.api_key,
        "data_type": data_project.data_type,
        "root_uri": data_project.root_uri,
        "models_dir": f"{results_dir}/models",
        "results_tiled_uri": parse_tiled_url(RESULTS_TILED_URI, user, project_name),
        "results_tiled_api_key": RESULTS_TILED_API_KEY,
        "results_dir": f"{results_dir}",
    }

    ls_python_file_name_inference = latent_space_params["python_file_name"]["inference"]
    dm_python_file_name = dim_reduction_params["python_file_name"]

    if flow_type == "podman" or "docker":
        job_params = {
            "flow_type": flow_type,
            "params_list": [
                {
                    "image_name": latent_space_params["image_name"],
                    "image_tag": latent_space_params["image_tag"],
                    "command": f"python {ls_python_file_name_inference}",
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,  # Default parameters
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/tiled_storage",
                    ],
                    "network": CONTAINER_NETWORK,
                },
                {
                    "image_name": dim_reduction_params["image_name"],
                    "image_tag": dim_reduction_params["image_tag"],
                    "command": f"python {dm_python_file_name}",
                    "params": {
                        "io_parameters": copy.copy(
                            io_parameters
                        ),  # Ensures uid_retrieve is empty
                        "model_parameters": {
                            "n_components": 2,
                            "min_dist": 0.1,
                            "n_neighbors": 5,
                        },
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/tiled_storage",
                    ],
                    "network": CONTAINER_NETWORK,
                },
            ],
        }
    elif flow_type == "conda":
        job_params = {
            "flow_type": "conda",
            "params_list": [
                {
                    "conda_env_name": latent_space_params["conda_env"],
                    "python_file_name": ls_python_file_name_inference,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
                {
                    "conda_env_name": dim_reduction_params["conda_env"],
                    "python_file_name": dm_python_file_name,
                    "params": {
                        "io_parameters": copy.copy(
                            io_parameters
                        ),  # Ensures uid_retrieve is empty
                        "model_parameters": {
                            "n_components": 2,
                            "min_dist": 0.1,
                            "n_neighbors": 5,
                        },
                    },
                },
            ],
        }

    else:
        job_params = {
            "flow_type": "slurm",
            "params_list": [
                {
                    "job_name": "latent_space_explorer",
                    "num_nodes": 1,
                    "partitions": PARTITIONS_CPU,
                    "reservations": RESERVATIONS_CPU,
                    "max_time": MAX_TIME_CPU,
                    "conda_env_name": latent_space_params["conda_env"],
                    "python_file_name": ls_python_file_name_inference,
                    "submission_ssh_key": SUBMISSION_SSH_KEY,
                    "forward_ports": FORWARD_PORTS,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
                {
                    "job_name": "latent_space_explorer",
                    "num_nodes": 1,
                    "partitions": PARTITIONS_CPU,
                    "reservations": RESERVATIONS_CPU,
                    "max_time": MAX_TIME_CPU,
                    "conda_env_name": dim_reduction_params["conda_env"],
                    "python_file_name": dm_python_file_name,
                    "submission_ssh_key": SUBMISSION_SSH_KEY,
                    "forward_ports": FORWARD_PORTS,
                    "params": {
                        "io_parameters": copy.copy(
                            io_parameters
                        ),  # Ensures uid_retrieve is empty
                        "model_parameters": {
                            "n_components": 2,
                            "min_dist": 0.1,
                            "n_neighbors": 5,
                        },
                    },
                },
            ],
        }

    return job_params


def parse_model_params(model_parameters_html, log, percentiles, mask):
    """
    Extracts parameters from the children component of a ParameterItems component,
    if there are any errors in the input, it will return an error status
    """
    errors = False
    input_params = {}
    for param in model_parameters_html["props"]["children"]:
        # param["props"]["children"][0] is the label
        # param["props"]["children"][1] is the input
        parameter_container = param["props"]["children"][1]
        # The achtual parameter item is the first and only child of the parameter container
        parameter_item = parameter_container["props"]["children"]["props"]
        key = parameter_item["id"]["param_key"]
        if "value" in parameter_item:
            value = parameter_item["value"]
        elif "checked" in parameter_item:
            value = parameter_item["checked"]
        if "error" in parameter_item:
            if parameter_item["error"] is not False:
                errors = True
        input_params[key] = value

    # Manually add data transformation parameters
    input_params["log"] = log
    input_params["percentiles"] = percentiles
    input_params["mask"] = mask if mask != "None" else None
    return input_params, errors
