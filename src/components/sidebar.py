import dash_bootstrap_components as dbc
from dash import dcc

from src.utils.mask_utils import get_mask_options


def sidebar(file_explorer, models):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:      Dash file explorer
        models:             Currently available ML algorithms in content registry
    """
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
                    title="Model configuration",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label(
                                        "Action",
                                        style={
                                            "height": "100%",
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                    width=2,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="action",
                                        options=[
                                            {"label": "Train", "value": "train_model"},
                                            {"label": "Tune", "value": "tune_model"},
                                            {
                                                "label": "Prediction",
                                                "value": "prediction_model",
                                            },
                                        ],
                                        value="train_model",
                                    ),
                                    width=10,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label(
                                        "Model",
                                        style={
                                            "height": "100%",
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                    width=2,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="model-selection",
                                        options=models,
                                        value=models[0]["value"],
                                    ),
                                    width=10,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    id="app-parameters",
                                    style={
                                        "overflowY": "scroll",
                                        "height": "58vh",  # Adjust as needed
                                    },
                                ),
                            ]
                        ),
                        dbc.Button(
                            "Execute",
                            id="execute",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "margin-left": "0px",
                                "margin-top": "10px",
                            },
                        ),
                    ],
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
