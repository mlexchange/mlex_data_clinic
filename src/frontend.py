import json
import pathlib
import shutil
from uuid import uuid4

from dash import Input, Output, State, dcc
from file_manager.data_project import DataProject

from app_layout import DATA_DIR, USER, app, long_callback_manager
from callbacks.display import refresh_image, toggle_warning_modal  # noqa: F401
from callbacks.download import disable_download, toggle_storage_modal  # noqa: F401
from callbacks.execute import execute  # noqa: F401
from callbacks.table import delete_row, update_table  # noqa: F401
from dash_component_editor import JSONParameterEditor
from utils.data_utils import get_input_params, prepare_directories
from utils.job_utils import MlexJob, str_to_dict
from utils.model_utils import get_gui_components, get_model_content


@app.callback(
    Output("app-parameters", "children"),
    Input("model-selection", "value"),
    Input("action", "value"),
    prevent_intial_call=True,
)
def load_parameters(model_selection, action_selection):
    """
    This callback dynamically populates the parameters and contents of the website according to the
    selected action & model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in Data Clinic)
    Returns:
        app-parameters:     Parameters according to the selected model & action
    """
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(
        _id={"type": str(uuid4())},
        json_blob=parameters,
    )
    gui_item.init_callbacks(app)
    return gui_item


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
        experiment_path = pathlib.Path(
            "data/mlexchange_store/{}/{}".format(USER, experiment_id)
        )
        shutil.make_archive("/app/tmp/results", "zip", experiment_path)
        return dcc.send_file("/app/tmp/results.zip")
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
    State({"base_id": "file-manager", "name": "docker-file-paths"}, "data"),
    State("model-name", "value"),
    State({"base_id": "file-manager", "name": "project-id"}, "data"),
    State("model-selection", "value"),
    State({"base_id": "file-manager", "name": "log-toggle"}, "on"),
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
    file_paths,
    model_name,
    project_id,
    model_id,
    log,
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
        file_paths:         Selected data files
        model_name:         Model name/description assigned by the user
        project_id:         Data project id
        model_id:           UID of model in content registry
        log:                Log toggle
    Returns:
        open the alert indicating that the job was submitted
    """
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    data_project.project_id = project_id
    model_uri, [train_cmd, prediction_cmd, tune_cmd] = get_model_content(model_id)
    experiment_id, out_path, data_info = prepare_directories(
        USER, data_project, train=(action_selection != "prediction_model")
    )
    input_params = get_input_params(children)
    input_params["log"] = log
    kwargs = {}

    if action_selection == "train_model":
        command = f"{train_cmd} -d {data_info} -o {out_path}"

    elif action_selection == "tune_model":
        training_exp_id = job_data[row[0]]["experiment_id"]
        model_path = pathlib.Path(
            "data/mlexchange_store/{}/{}".format(USER, training_exp_id)
        )
        kwargs = {"train_params": job_data[row[0]]["parameters"]}
        train_params = str_to_dict(job_data[row[0]]["parameters"])

        # Get target size from training process
        input_params["target_width"] = train_params["target_width"]
        input_params["target_height"] = train_params["target_height"]

        # Define command to run
        command = f"{tune_cmd} -d {data_info} -m {model_path} -o {out_path}"

    else:
        training_exp_id = job_data[row[0]]["experiment_id"]
        model_path = pathlib.Path(
            "data/mlexchange_store/{}/{}".format(USER, training_exp_id)
        )
        if job_data[row[0]]["job_type"] == "train_model":
            train_params = job_data[row[0]]["parameters"]
        else:
            train_params = job_data[row[0]]["parameters"].split("Training Parameters:")[
                -1
            ]

        kwargs = {"train_params": train_params}
        train_params = str_to_dict(train_params)

        # Get target size from training process
        input_params["target_width"] = train_params["target_width"]
        input_params["target_height"] = train_params["target_height"]

        # Define command to run
        command = f"{prediction_cmd} -d {data_info} -m {model_path} -o {out_path}"

    job = MlexJob(
        service_type="backend",
        description=model_name,
        working_directory="{}".format(DATA_DIR),
        job_kwargs={
            "uri": model_uri,
            "type": "docker",
            "cmd": f"{command} -p '{json.dumps(input_params)}'",
            "container_kwargs": {"shm_size": "2gb"},
            "kwargs": {
                "job_type": action_selection,
                "experiment_id": experiment_id,
                "dataset": project_id,
                "params": input_params,
                **kwargs,
            },
        },
    )
    job.submit(USER, num_cpus, num_gpus)
    return True


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
