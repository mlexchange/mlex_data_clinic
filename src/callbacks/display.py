import os
import pathlib
import pickle

import numpy as np
from dash import ALL, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject
from PIL import Image

from src.app_layout import DATA_DIR, USER
from src.utils.job_utils import str_to_dict
from src.utils.plot_utils import get_bottleneck, plot_figure


@callback(
    Output("current-target-size", "data"),
    Output("ls_graph", "src"),
    Input(
        {
            "type": ALL,
            "param_key": "latent_dim",
            "name": "latent_dim",
            "layer": "input",
        },
        "value",
    ),
    Input({"type": ALL, "param_key": "target_width", "name": "target_width"}, "value"),
    Input(
        {"type": ALL, "param_key": "target_height", "name": "target_height"}, "value"
    ),
    Input("jobs-table", "selected_rows"),
    Input("jobs-table", "data"),
    State("action", "value"),
    prevent_initial_call=True,
)
def refresh_bottleneck(
    ls_var,
    target_width,
    target_height,
    row,
    data_table,
    action_selection,
):
    """
    This callback refreshes the bottleneck plot according to the selected job type
    Args:
        ls_var:             Latent space value
        target_width:       Target width
        target_height:      Target height
        row:                Selected row (job)
        data_table:         Lists of jobs
        action_selection:   Action selected
    Returns:
        current-target-size:    Target size
        ls_graph:               Bottleneck plot
    """
    # Get selected job type
    if row and len(row) > 0:
        selected_job_type = data_table[row[0]]["job_type"]
    else:
        selected_job_type = None

    # If selected job type is train_model or tune_model
    if selected_job_type:
        if selected_job_type == "train_model":
            train_params = data_table[row[0]]["parameters"]
        else:
            train_params = data_table[row[0]]["parameters"].split(
                "Training Parameters:"
            )[-1]

        train_params = str_to_dict(train_params)
        ls_var = int(train_params["latent_dim"])
        target_width = int(train_params["target_width"])
        target_height = int(train_params["target_height"])

    else:
        if action_selection != "train_model":
            return [32, 32], get_bottleneck(1, 1, 1, False)

        target_width = target_width[0] if len(target_width) > 0 else 32
        target_height = target_height[0] if len(target_height) > 0 else 32
        ls_var = ls_var[0] if len(ls_var) > 0 else 32

    # Generate bottleneck plot
    ls_plot = get_bottleneck(ls_var, target_width, target_height)
    return [target_width, target_height], ls_plot


@callback(
    Output("img-slider", "max", allow_duplicate=True),
    Output("img-slider", "value", allow_duplicate=True),
    Input("jobs-table", "selected_rows"),
    Input("jobs-table", "data"),
    State("img-slider", "value"),
    prevent_initial_call=True,
)
def update_slider_boundaries_prediction(
    row,
    data_table,
    slider_ind,
):
    """
    This callback updates the slider boundaries according to the selected job type
    Args:
        row:                Selected row (job)
        data_table:         Lists of jobs
        slider_ind:         Slider index
    Returns:
        img-slider:         Maximum value of the slider
        img-slider:         Slider index
    """
    # Get selected job type
    if row and len(row) > 0:
        selected_job_type = data_table[row[0]]["job_type"]
    else:
        selected_job_type = None

    # If selected job type is train_model or tune_model
    if selected_job_type == "prediction_model":
        job_id = data_table[row[0]]["experiment_id"]
        data_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{job_id}")

        with open(f"{data_path}/.file_manager_vars.pkl", "rb") as file:
            data_project_dict = pickle.load(file)
        data_project = DataProject.from_dict(data_project_dict)

        # Check if slider index is out of bounds
        if (
            len(data_project.datasets) > 0
            and slider_ind > data_project.datasets[-1].cumulative_data_count - 1
        ):
            slider_ind = 0

        return data_project.datasets[-1].cumulative_data_count - 1, slider_ind

    else:
        raise PreventUpdate


@callback(
    Output("img-slider", "max"),
    Output("img-slider", "value"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    Input("jobs-table", "selected_rows"),
    State("img-slider", "value"),
    prevent_initial_call=True,
)
def update_slider_boundaries_new_dataset(
    data_project_dict,
    row,
    slider_ind,
):
    """
    This callback updates the slider boundaries according to the selected job type
    Args:
        data_project_dict:  Data project dictionary
        row:                Selected row (job)
        slider_ind:         Slider index
    Returns:
        img-slider:         Maximum value of the slider
        img-slider:         Slider index
    """
    data_project = DataProject.from_dict(data_project_dict)
    if len(data_project.datasets) > 0:
        max_ind = data_project.datasets[-1].cumulative_data_count - 1
    else:
        max_ind = 0

    slider_ind = min(slider_ind, max_ind)
    return max_ind, slider_ind


@callback(
    Output("orig_img_store", "data"),
    Output("data-size-out", "children"),
    Input("img-slider", "value"),
    Input("current-target-size", "data"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("jobs-table", "selected_rows"),
    State("jobs-table", "data"),
)
def refresh_image(
    img_ind,
    target_size,
    data_project_dict,
    row,
    data_table,
):
    """
    This callback refreshes the original image according to the selected job type
    Args:
        img_ind:            Image index
        target_size:        Target size
        data_project_dict:  Data project dictionary
        row:                Selected row (job)
        data_table:         Lists of jobs
    Returns:
        orig_img:           Original image
        data_size:          Data size
    """
    # Get selected job type
    if row and len(row) > 0:
        selected_job_type = data_table[row[0]]["job_type"]
    else:
        selected_job_type = None

    if selected_job_type == "prediction_model":
        job_id = data_table[row[0]]["experiment_id"]
        data_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{job_id}")

        with open(f"{data_path}/.file_manager_vars.pkl", "rb") as file:
            data_project_dict = pickle.load(file)

    data_project = DataProject.from_dict(data_project_dict)
    if (
        len(data_project.datasets) > 0
        and data_project.datasets[-1].cumulative_data_count > 0
    ):
        origimg, _ = data_project.read_datasets(
            indices=[img_ind], export="pillow", resize=False
        )
        origimg = origimg[0]
    else:
        origimg = Image.fromarray((np.zeros((32, 32)).astype(np.uint8)))
    (width, height) = origimg.size
    origimg = plot_figure(origimg.resize((target_size[0], target_size[1])))
    data_size = f"Original Image: ({width}x{height}). Resized Image: ({target_size[0]}x{target_size[1]})."
    return origimg, data_size


@callback(
    Output("rec_img", "src"),
    Input("img-slider", "value"),
    Input("jobs-table", "selected_rows"),
    Input("jobs-table", "data"),
    # Input({"base_id": "file-manager", "name": "log-toggle"}, "on"),
    Input("current-target-size", "data"),
)
def refresh_reconstruction(
    img_ind,
    row,
    data_table,
    # log,
    target_size,
):
    """
    This callback refreshes the reconstructed image according to the selected job type
    Args:
        img_ind:            Image index
        row:                Selected row (job)
        data_table:         Lists of jobs
        log:                Log toggle
        target_size:        Target size
    Returns:
        rec_img:            Reconstructed image
    """
    # Get selected job type
    if row and len(row) > 0:
        selected_job_type = data_table[row[0]]["job_type"]
    else:
        selected_job_type = None

    if selected_job_type == "prediction_model":
        job_id = data_table[row[0]]["experiment_id"]
        reconstructed_path = f"{DATA_DIR}/mlex_store/{USER}/{job_id}"
        if os.path.exists(f"{reconstructed_path}/reconstructed_{img_ind}.jpg"):
            reconst_img = Image.open(
                f"{reconstructed_path}/reconstructed_{img_ind}.jpg"
            )
        else:
            reconst_img = Image.fromarray(
                (np.zeros((target_size[1], target_size[0])).astype(np.uint8))
            )

    else:
        reconst_img = Image.fromarray(
            (np.zeros((target_size[1], target_size[0])).astype(np.uint8))
        )

    return plot_figure(reconst_img)


@callback(
    Output("warning-modal", "is_open"),
    Output("warning-msg", "children"),
    Input("warning-cause", "data"),
    State("warning-modal", "is_open"),
    prevent_initial_call=True,
)
def open_warning_modal(warning_cause, is_open):
    """
    This callback toggles a warning/error message
    Args:
        warning_cause:      Cause that triggered the warning
        is_open:            Close/open state of the warning
    Returns:
        is_open:            Close/open state of the warning
         warning_msg:       Warning message
    """
    if warning_cause == "wrong_dataset":
        return not is_open, "The dataset you have selected is not supported."
    elif warning_cause == "no_row_selected":
        return not is_open, "Please select a trained model from the List of Jobs"
    elif warning_cause == "no_dataset":
        return not is_open, "Please upload the dataset before submitting the job."
    elif warning_cause == "data_project_not_ready":
        return (
            not is_open,
            "The data project is still being created. Please try again in a couple minutes.",
        )
    else:
        return False, ""


@callback(
    Output("warning-modal", "is_open", allow_duplicate=True),
    Input("ok-button", "n_clicks"),
    prevent_initial_call=True,
)
def close_warning_modal(ok_n_clicks):
    """
    This callback closes warning/error message
    Args:
        ok_n_clicks:        Close the warning
    Returns:
        is_open:            Close/open state of the warning
    """
    return False
