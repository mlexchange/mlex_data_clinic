import numpy as np
from dash import ALL, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import get_children_flow_run_ids
from PIL import Image

from src.app_layout import TILED_KEY, USER
from src.utils.data_utils import hash_list_of_strings, tiled_results
from src.utils.plot_utils import get_bottleneck, plot_figure


@callback(
    Output("current-target-size", "data"),
    Output("ls-graph", "src"),
    Input(
        {
            "type": ALL,
            "param_key": "latent_dim",
            "name": "latent_dim",
            "layer": "input",
        },
        "value",
    ),
    Input(
        {
            "type": ALL,
            "param_key": "target_width",
            "name": "target_width",
            "layer": "input",
        },
        "value",
    ),
    Input(
        {
            "type": ALL,
            "param_key": "target_height",
            "name": "target_height",
            "layer": "input",
        },
        "value",
    ),
    prevent_initial_call=True,
)
def refresh_bottleneck(
    ls_var,
    target_width,
    target_height,
):
    """
    This callback refreshes the bottleneck plot according to the selected job type
    Args:
        ls_var:             Latent space value
        target_width:       Target width
        target_height:      Target height
    Returns:
        current-target-size:    Target size
        ls_graph:               Bottleneck plot
    """
    target_width = target_width[0] if len(target_width) > 0 else 32
    target_height = target_height[0] if len(target_height) > 0 else 32
    ls_var = ls_var[0] if len(ls_var) > 0 else 32

    # Generate bottleneck plot
    ls_plot = get_bottleneck(ls_var, target_width, target_height)
    return [target_width, target_height], ls_plot


@callback(
    Output("img-slider", "max"),
    Output("img-slider", "value"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("img-slider", "value"),
    prevent_initial_call=True,
)
def update_slider_boundaries_new_dataset(
    data_project_dict,
    slider_ind,
):
    """
    This callback updates the slider boundaries according to the selected job type
    Args:
        data_project_dict:  Data project dictionary
        slider_ind:         Slider index
    Returns:
        img-slider:         Maximum value of the slider
        img-slider:         Slider index
    """
    if data_project_dict != {}:
        data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)
        if len(data_project.datasets) > 0:
            max_ind = data_project.datasets[-1].cumulative_data_count - 1
        else:
            max_ind = 0

        slider_ind = min(slider_ind, max_ind)
        return max_ind, slider_ind
    else:
        raise PreventUpdate


@callback(
    Output("orig-img", "src"),
    Output("data-size-out", "children"),
    Input("img-slider", "value"),
    Input("current-target-size", "data"),
    Input("log-transform", "value"),
    Input("min-max-percentile", "value"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
)
def refresh_image(
    img_ind,
    target_size,
    log_transform,
    percentiles,
    data_project_dict,
):
    """
    This callback refreshes the original image according to the selected job type
    Args:
        img_ind:            Image index
        target_size:        Target size
        log_transform:      Log transform
        percentiles:        Percentiles
        data_project_dict:  Data project dictionary
    Returns:
        orig_img:           Original image
        data_size:          Data size
    """
    if data_project_dict != {}:
        data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)
        if (
            len(data_project.datasets) > 0
            and data_project.datasets[-1].cumulative_data_count > 0
        ):
            if percentiles is None:
                percentiles = [0, 100]
            origimg, _ = data_project.read_datasets(
                indices=[img_ind],
                export="pillow",
                resize=False,
                log=log_transform,
                percentiles=percentiles,
            )
            origimg = origimg[0]
        else:
            origimg = Image.fromarray((np.zeros((32, 32)).astype(np.uint8)))
        (width, height) = origimg.size
        origimg = plot_figure(origimg.resize((target_size[0], target_size[1])))
        data_size = f"Original Image: ({width}x{height}). Resized Image: ({target_size[0]}x{target_size[1]})."
        return origimg, data_size
    else:
        raise PreventUpdate


@callback(
    Output("project-name", "data"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def update_project_name(data_project_dict):
    data_project = DataProject.from_dict(data_project_dict)
    data_uris = [dataset.uri for dataset in data_project.datasets]
    project_name = hash_list_of_strings(data_uris)
    return project_name


@callback(
    Output("rec-img", "src", allow_duplicate=True),
    Input("show-reconstructions", "value"),
    Input("img-slider", "value"),
    Input("current-target-size", "data"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": "data-clinic-jobs",
        },
        "value",
    ),
    State("project-name", "data"),
    prevent_initial_call=True,
)
def refresh_reconstruction(show_recons, img_ind, target_size, job_id, project_name):
    """
    This callback refreshes the reconstructed image according to the selected job
    Args:
        show_recons:        Show reconstructed image
        img_ind:            Image index
        target_size:        Target size
        job_id:             Selected job
        project_name:       Data project name
    Returns:
        rec_img:            Reconstructed image
    """
    if show_recons:
        child_job_id = get_children_flow_run_ids(job_id)[1]
        expected_result_uri = f"{USER}/{project_name}/{child_job_id}/reconstructions"
        recons_array = tiled_results.get_data_by_trimmed_uri(
            expected_result_uri, slice=img_ind
        )
    else:
        recons_array = np.zeros((target_size[1], target_size[0])).astype(np.uint8)
    return plot_figure(Image.fromarray(recons_array))


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
