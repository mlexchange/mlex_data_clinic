from dash import Input, Output, State, callback


@callback(
    Output("resources-setup", "is_open"),
    Output("warning-cause", "data"),
    Input("execute", "n_clicks"),
    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State({"base_id": "file-manager", "name": "total-num-data-points"}, "data"),
    prevent_initial_call=True,
)
def execute(execute, action_selection, job_data, row, num_imgs):
    """
    This callback validates the ml model and opens the resources modal
    Args:
        execute:            Execute button
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        num_imgs:           Number of images
    Returns:
        open/close the resources setup modal, and submits the training/prediction job accordingly
        warning_cause:      Activates a warning pop-up window if needed
    """
    if num_imgs == 0:
        return False, "no_dataset"
    elif action_selection != "train_model" and not row:
        return False, "no_row_selected"
    elif row:
        if (
            action_selection != "train_model"
            and job_data[row[0]]["job_type"] == "prediction_model"
        ):
            return False, "no_row_selected"
    return True, ""


@callback(
    Output("resources-setup", "is_open", allow_duplicate=True),
    Output("warning-cause", "data", allow_duplicate=True),
    Input("submit", "n_clicks"),
    prevent_initial_call=True,
)
def close_resources_popup(submit):
    return False, ""
