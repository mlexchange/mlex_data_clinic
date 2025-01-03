import os
from urllib.parse import urljoin

RESULTS_DIR = os.getenv("RESULTS_DIR", "")
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")
CONDA_ENV_NAME = os.getenv("CONDA_ENV_NAME", "mlex_pytorch_autoencoders")
TRAIN_SCRIPT_PATH = os.getenv("TRAIN_SCRIPT_PATH", "scr/train_model.py")
TUNE_SCRIPT_PATH = os.getenv("TUNE_SCRIPT_PATH", "scr/tune_model.py")
INFERENCE_SCRIPT_PATH = os.getenv("INFERENCE_SCRIPT_PATH", "scr/predict_model.py")

TRAIN_PARAMS_EXAMPLE = {
    "flow_type": "conda",
    "params_list": [
        {
            "conda_env_name": f"{CONDA_ENV_NAME}",
            "python_file_name": f"{TRAIN_SCRIPT_PATH}",
            "params": {
                "io_parameters": {"uid_save": "uid0001", "uid_retrieve": "uid0001"}
            },
        },
        {
            "conda_env_name": f"{CONDA_ENV_NAME}",
            "python_file_name": f"{INFERENCE_SCRIPT_PATH}",
            "params": {
                "io_parameters": {"uid_save": "uid0001", "uid_retrieve": "uid0001"}
            },
        },
    ],
}

TUNE_PARAMS_EXAMPLE = {
    "flow_type": "conda",
    "params_list": [
        {
            "conda_env_name": f"{CONDA_ENV_NAME}",
            "python_file_name": f"{TUNE_SCRIPT_PATH}",
            "params": {
                "io_parameters": {"uid_save": "uid0001", "uid_retrieve": "uid0001"}
            },
        },
        {
            "conda_env_name": f"{CONDA_ENV_NAME}",
            "python_file_name": f"{INFERENCE_SCRIPT_PATH}",
            "params": {
                "io_parameters": {"uid_save": "uid0001", "uid_retrieve": "uid0001"}
            },
        },
    ],
}

INFERENCE_PARAMS_EXAMPLE = {
    "flow_type": "conda",
    "params_list": [
        {
            "conda_env_name": f"{CONDA_ENV_NAME}",
            "python_file_name": f"{INFERENCE_SCRIPT_PATH}",
            "params": {
                "io_parameters": {"uid_save": "uid0001", "uid_retrieve": "uid0001"}
            },
        },
    ],
}


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
    data_project, model_parameters, model_name, user, project_name
):
    """
    Parse training job parameters
    """
    # TODO: Use model_name to define the conda_env/algorithm to be executed
    data_uris = [dataset.uri for dataset in data_project.datasets]
    io_parameters = {
        "uid_retrieve": "",
        "data_uris": data_uris,
        "data_tiled_api_key": data_project.api_key,
        "data_type": data_project.data_type,
        "root_uri": data_project.root_uri,
        "model_dir": f"{RESULTS_DIR}/models",
        "results_tiled_uri": parse_tiled_url(RESULTS_TILED_URI, user, project_name),
        "results_tiled_api_key": RESULTS_TILED_API_KEY,
        "results_dir": f"{RESULTS_DIR}",
    }

    TRAIN_PARAMS_EXAMPLE["params_list"][0]["params"]["io_parameters"] = io_parameters
    TRAIN_PARAMS_EXAMPLE["params_list"][1]["params"]["io_parameters"] = io_parameters

    TRAIN_PARAMS_EXAMPLE["params_list"][0]["params"][
        "model_parameters"
    ] = model_parameters
    TRAIN_PARAMS_EXAMPLE["params_list"][1]["params"][
        "model_parameters"
    ] = model_parameters

    return TRAIN_PARAMS_EXAMPLE, project_name


def parse_model_params(model_parameters_html, log, percentiles):
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
    return input_params, errors
