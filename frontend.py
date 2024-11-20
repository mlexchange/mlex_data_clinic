import os
import pathlib
import shutil
import tempfile
from uuid import uuid4

from dash import Input, Output, State, dcc, html
from dotenv import load_dotenv

from src.app_layout import (
    DATA_DIR,
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
from src.callbacks.execute import run_train  # noqa: F401

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


if __name__ == "__main__":
    app.run_server(debug=True, host=APP_HOST, port=APP_PORT)
