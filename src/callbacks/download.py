from dash import Input, Output, State, callback


@callback(
    Output("download-button", "disabled"),
    Input("jobs-table", "selected_rows"),
    State("jobs-table", "data"),
    prevent_initial_call=True,
)
def disable_download(row, job_table):
    """
    This callback enables or disables the download button
    """
    disabled_button = True
    if row is not None and len(row) > 0 and job_table[row[0]]["status"] == "complete":
        disabled_button = False
    return disabled_button


@callback(
    Output("storage-modal", "is_open"),
    Input("download-button", "n_clicks"),
    Input("close-storage-modal", "n_clicks"),
    State("storage-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_storage_modal(download, close_modal, is_open):
    """
    This callback toggles the storage message in modal
    """
    return not (is_open)
