import os
from uuid import uuid4

from dash import MATCH, Input, Output, html
from dotenv import load_dotenv

from src.app_layout import DATA_DIR, app, latent_space_models, mlex_components
from src.callbacks.display import (  # noqa: F401
    refresh_bottleneck,
    refresh_image,
    refresh_reconstruction,
    toggle_sidebar,
    update_slider_boundaries_new_dataset,
)
from src.callbacks.execute import run_train  # noqa: F401
from src.callbacks.infrastructure_check import (  # noqa: F401
    check_infra_state,
    update_infra_state,
)

load_dotenv(".env")

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = os.getenv("APP_PORT", "8072")
DIR_MOUNT = os.getenv("DIR_MOUNT", DATA_DIR)

server = app.server


@app.callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": MATCH,
        },
        "children",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": MATCH,
        },
        "value",
    ),
)
def update_model_parameters(model_name):
    model = latent_space_models[model_name]
    if model["gui_parameters"]:
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        return item_list
    else:
        return html.Div("Model has no parameters")


if __name__ == "__main__":
    app.run_server(debug=True, host=APP_HOST, port=APP_PORT)
