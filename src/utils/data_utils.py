import pathlib
import uuid

import pandas as pd

from src.app_layout import DATA_DIR, TILED_KEY


def prepare_directories(user_id, data_project, train=True, correct_path=False):
    """
    Prepare data directories that host experiment results and data movements processes for tiled
    If data is served through tiled, a local copy will be made for ML training and inference
    processes in file system located at data/mlexchange_store/user_id/tiledprojectid_localprojectid
    Args:
        user_id:        User ID
        data_project:   List of data sets in the application
        train:          Flag to indicate if the data is used for training or inference
        correct_path:   Flag to indicate if the path should be corrected to be used in the docker
    Returns:
        experiment_id:  ML experiment ID
        out_path:       Path were experiment results will be stored
        info_file:      Filename of a parquet file that contains the list of data sets within the
                        current project
    """
    experiment_id = str(uuid.uuid4())
    out_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{user_id}/{experiment_id}")
    out_path.mkdir(parents=True, exist_ok=True)
    data_type = data_project.data_type
    if data_type == "tiled" and train:
        # Download tiled data to local
        uri_list = data_project.tiled_to_local_project(
            DATA_DIR, correct_path=correct_path
        )
        data_info = pd.DataFrame({"uri": uri_list})
    elif data_type == "tiled":
        # Save sub uris
        root_uri = data_project.root_uri
        data_info = pd.DataFrame({"root_uri": [root_uri]})
        sub_uris_df = pd.DataFrame(
            {"sub_uris": [dataset.uri for dataset in data_project.datasets]}
        )
        data_info = pd.concat([data_info, sub_uris_df], axis=1)
        data_info["api_key"] = [TILED_KEY] * len(data_info)
    else:
        # Save filenames
        uri_list = []
        for dataset in data_project.datasets:
            uri_list.extend(
                [dataset.uri + "/" + filename for filename in dataset.filenames]
            )
        if correct_path:
            root_uri = "/app/work/data" + data_project.root_uri.split(DATA_DIR, 1)[-1]
        else:
            root_uri = data_project.root_uri
        data_info = pd.DataFrame({"uri": [root_uri + "/" + uri for uri in uri_list]})
    data_info["type"] = data_type
    data_info.to_parquet(f"{out_path}/data_info.parquet", engine="pyarrow")
    return experiment_id, out_path, f"{out_path}/data_info.parquet"


def get_input_params(children):
    """
    Gets the model parameters and its corresponding values
    """
    input_params = {}
    if bool(children):
        try:
            for child in children["props"]["children"]:
                key = child["props"]["children"][1]["props"]["id"]["param_key"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        except Exception:
            for child in children:
                key = child["props"]["children"][1]["props"]["id"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
    return input_params
