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
        models:             Currently available ML algorithms in content registry
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
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label("Log Transform"), width=4, align="start"
                                ),
                                dbc.Col(
                                    dbc.Switch(
                                        id="log-transform",
                                        value=False,
                                        label_style={"display": "none"},
                                        style={"height": "20px"},
                                    ),
                                    align="start",
                                ),
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label("Min-Max Percentile"),
                                    width=4,
                                ),
                                dbc.Col(
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
                            ],
                            style={"margin-bottom": "10px"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label("Mask Selection"),
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="mask-dropdown",
                                        options=get_mask_options(),
                                    ),
                                ),
                            ]
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
