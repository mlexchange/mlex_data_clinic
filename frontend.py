import os
import pathlib
import pickle
import shutil
import tempfile
from uuid import uuid4

from dash import Input, Output, State, dcc, html
from dotenv import load_dotenv
from file_manager.data_project import DataProject

from src.app_layout import (
    DATA_DIR,
    TILED_KEY,
    USER,
    app,
    long_callback_manager,
    mlex_components,
    models,
)
from src.callbacks.display import (  # noqa: F401
    close_warning_modal,
    open_warning_modal,
    refresh_bottleneck,
    refresh_image,
    refresh_reconstruction,
    update_slider_boundaries_new_dataset,
    update_slider_boundaries_prediction,
)
from src.callbacks.download import disable_download, toggle_storage_modal  # noqa: F401
from src.callbacks.execute import close_resources_popup, execute  # noqa: F401
from src.utils.data_utils import get_input_params, prepare_directories

load_dotenv(".env")

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = os.getenv("APP_PORT", "8072")
DIR_MOUNT = os.getenv("DIR_MOUNT", DATA_DIR)

server = app.server


@app.callback(
    Output("model-parameters", "children"),
    Input("model-list", "value"),
)
def update_model_parameters(model_name):
    model = models[model_name]
    if model["gui_parameters"]:
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        return item_list
    else:
        return html.Div("Model has no parameters")


@app.long_callback(
    Output("download-out", "data"),
    Input("download-button", "n_clicks"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    manager=long_callback_manager,
    prevent_intial_call=True,
)
def save_results(download, job_data, row):
    """
    This callback saves the experimental results as a ZIP file
    Args:
        download:   Download button
        job_data:   Table of jobs
        row:        Selected job/row
    Returns:
        ZIP file with results
    """
    if download and row:
        experiment_id = job_data[row[0]]["experiment_id"]
        experiment_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{experiment_id}")
        with tempfile.TemporaryDirectory():
            tmp_dir = tempfile.gettempdir()
            archive_path = os.path.join(tmp_dir, "results")
            shutil.make_archive(archive_path, "zip", experiment_path)
        return dcc.send_file(f"{archive_path}.zip")
    else:
        return None


@app.long_callback(
    Output("job-alert-confirm", "is_open"),
    Input("submit", "n_clicks"),
    State("app-parameters", "children"),
    State("num-cpus", "value"),
    State("num-gpus", "value"),
    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("model-name", "value"),
    State("model-selection", "value"),
    State("log-transform", "value"),
    State("min-max-percentile", "value"),
    running=[(Output("job-alert", "is_open"), "True", "False")],
    manager=long_callback_manager,
    prevent_initial_call=True,
)
def submit_ml_job(
    submit,
    children,
    num_cpus,
    num_gpus,
    action_selection,
    job_data,
    row,
    data_project_dict,
    model_name,
    model_id,
    log,
    percentiles,
):
    """
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        execute:            Execute button
        submit:             Submit button
        children:           Model parameters
        num_cpus:           Number of CPUs assigned to job
        num_gpus:           Number of GPUs assigned to job
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        data_project_dict:  Data project information
        model_name:         Model name/description assigned by the user
        model_id:           UID of model in content registry
        log:                Log toggle
        percentiles:        Min-Max Percentile values
    Returns:
        open the alert indicating that the job was submitted
    """
    data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)
    # Update retrieval of model_uri, train_cmd, prediction_cmd, tune_cmd
    train_cmd, prediction_cmd, tune_cmd = 2, 3, 4
    experiment_id, orig_out_path, data_info = prepare_directories(
        USER,
        data_project,
        train=(action_selection != "prediction_model"),
        correct_path=(DATA_DIR == DIR_MOUNT),
    )  # noqa: F841
    input_params = get_input_params(children)
    input_params["log"] = log
    input_params["percentiles"] = percentiles
    kwargs = {}  # noqa: F841

    # Find the relative data directory in docker container
    if DIR_MOUNT == DATA_DIR:
        relative_data_dir = "/app/work/data"
        out_path = "/app/work/data" + str(orig_out_path).split(DATA_DIR, 1)[-1]
        data_info = "/app/work/data" + str(data_info).split(DATA_DIR, 1)[-1]
    else:
        relative_data_dir = DATA_DIR

    if action_selection == "train_model":
        command = f"{train_cmd} -d {data_info} -o {out_path}"  # noqa: F841

    elif action_selection == "tune_model":
        training_exp_id = job_data[row[0]]["experiment_id"]
        model_path = pathlib.Path(
            f"{relative_data_dir}/mlex_store/{USER}/{training_exp_id}"
        )
        kwargs = {"train_params": job_data[row[0]]["parameters"]}  # noqa: F841
        train_params = {}

        # Get target size from training process
        input_params["target_width"] = train_params["target_width"]
        input_params["target_height"] = train_params["target_height"]

        # Define command to run
        command = (
            f"{tune_cmd} -d {data_info} -m {model_path} -o {out_path}"  # noqa: F841
        )

    else:
        training_exp_id = job_data[row[0]]["experiment_id"]
        model_path = pathlib.Path(
            f"{relative_data_dir}/mlex_store/{USER}/{training_exp_id}"
        )
        if job_data[row[0]]["job_type"] == "train_model":
            train_params = job_data[row[0]]["parameters"]
        else:
            train_params = job_data[row[0]]["parameters"].split("Training Parameters:")[
                -1
            ]

        kwargs = {"train_params": train_params}  # noqa: F841
        train_params = {}

        # Get target size from training process
        input_params["target_width"] = train_params["target_width"]
        input_params["target_height"] = train_params["target_height"]

        # Define command to run
        command = f"{prediction_cmd} -d {data_info} -m {model_path} -o {out_path}"  # noqa: F841

        # Save data project dict
        data_project_dict = data_project.to_dict()

        with open(f"{orig_out_path}/.file_manager_vars.pkl", "wb") as file:
            pickle.dump(
                data_project_dict,
                file,
            )

    job = 1  # Update job submission
    job.submit(USER, num_cpus, num_gpus)
    return True


if __name__ == "__main__":
    app.run_server(debug=True, host=APP_HOST, port=APP_PORT)
