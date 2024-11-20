import os
import traceback
import uuid
from datetime import datetime

import pytz
from dash import MATCH, Input, Output, State, callback
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import schedule_prefect_flow

from src.utils.job_utils import parse_model_params, parse_train_job_params
from src.utils.plot_utils import generate_notification

MODE = os.getenv("MODE", "")
TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
FLOW_NAME = os.getenv("FLOW_NAME", "")
PREFECT_TAGS = os.getenv("PREFECT_TAGS", ["data-clinic"])


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "notifications_container",
            "aio_id": MATCH,
        },
        "children",
        allow_duplicate=True,
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train_button",
            "aio_id": MATCH,
        },
        "n_clicks",
    ),
    State("model-parameters", "children"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("model-list", "value"),
    State("log-transform", "value"),
    State("min-max-percentile", "value"),
    State(
        {"component": "DbcJobManagerAIO", "subcomponent": "job_name", "aio_id": MATCH},
        "value",
    ),
    prevent_initial_call=True,
)
def run_train(
    n_clicks,
    model_parameter_container,
    data_project_dict,
    model_name,
    log,
    percentiles,
    job_name,
):
    """
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        n_clicks:                   Number of clicks
        model_parameter_container:  App parameters
        data_project_dict:          Data project dictionary
        model_name:                 Selected model name
        log:                        Log transform
        percentiles:                Min-max percentiles
        job_name:                   Job name
    Returns:
        open the alert indicating that the job was submitted
    """
    model_parameters, parameter_errors = parse_model_params(
        model_parameter_container, log, percentiles
    )
    # Check if the model parameters are valid
    if parameter_errors:
        notification = generate_notification(
            "Model Parameters",
            "red",
            "fluent-mdl2:machine-learning",
            "Model parameters are not valid!",
        )
        return notification

    data_project = DataProject.from_dict(data_project_dict)
    train_params, project_name = parse_train_job_params(
        data_project, model_parameters, model_name
    )

    if MODE == "dev":
        job_uid = str(uuid.uuid4())
        job_message = (
            f"Dev Mode: Job has been succesfully submitted with uid: {job_uid}"
        )
        notification_color = "primary"
    else:
        try:
            # Schedule job
            current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
                "%Y/%m/%d %H:%M:%S"
            )
            job_uid = schedule_prefect_flow(
                FLOW_NAME,
                parameters=train_params,
                flow_run_name=f"{job_name} {current_time}",
                tags=PREFECT_TAGS + ["train", project_name],
            )
            job_message = f"Job has been succesfully submitted with uid: {job_uid}"
            notification_color = "indigo"
        except Exception as e:
            # Print the traceback to the console
            traceback.print_exc()
            job_uid = None
            job_message = f"Job presented error: {e}"
            notification_color = "danger"

    notification = generate_notification(
        "Job Submission", notification_color, "formkit:submit", job_message
    )

    return notification
