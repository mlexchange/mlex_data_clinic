import dash_bootstrap_components as dbc
from dash import dcc, html
from mlex_utils.dash_utils.components_bootstrap.component_utils import (
    DbcControlItem as ControlItem,
)

from src.utils.mask_utils import get_mask_options


def sidebar(file_explorer, job_manager, models):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:      Dash file explorer
        job_manager:        Job manager object
        models:             Currently available ML algorithms
    """
    model_list = [{"label": model, "value": model} for model in models.modelname_list]
    sidebar = [
        dbc.Accordion(
            id="sidebar",
            children=[
                dbc.AccordionItem(
                    title="Data selection",
                    children=file_explorer,
                ),
                dbc.AccordionItem(
                    title="Data transformation",
                    children=[
                        ControlItem(
                            "",
                            "empty-title",
                            dbc.Switch(
                                id="log-transform",
                                value=False,
                                label="Log Transform",
                            ),
                        ),
                        html.P(),
                        ControlItem(
                            "Min-Max Percentile",
                            "min-max-percentile-title",
                            dcc.RangeSlider(
                                id="min-max-percentile",
                                min=0,
                                max=100,
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True,
                                },
                            ),
                        ),
                        html.P(),
                        ControlItem(
                            "Mask Selection",
                            "mask-dropdown-title",
                            dbc.Select(
                                id="mask-dropdown",
                                options=get_mask_options(),
                            ),
                        ),
                    ],
                ),
                dbc.AccordionItem(
                    [
                        ControlItem(
                            "Algorithm",
                            "select-algorithm",
                            dbc.Select(
                                id="model-list",
                                options=model_list,
                                value=(
                                    model_list[0]["value"]
                                    if model_list[0]["value"]
                                    else None
                                ),
                            ),
                        ),
                        html.Div(id="model-parameters"),
                        html.P(),
                        job_manager,
                    ],
                    title="Model Configuration",
                ),
            ],
            style={"overflow-y": "scroll", "height": "90vh"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Warning"),
                dbc.ModalBody(id="warning-msg"),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "OK",
                            id="ok-button",
                            color="danger",
                            outline=False,
                            className="ms-auto",
                            n_clicks=0,
                        ),
                    ]
                ),
            ],
            id="warning-modal",
            is_open=False,
        ),
        dcc.Store(id="warning-cause", data=""),
    ]
    return sidebar
