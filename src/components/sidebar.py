import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc


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
                        dbc.Label("Log-transform"),
                        daq.BooleanSwitch(
                            id="log-transform",
                            on=False,
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
