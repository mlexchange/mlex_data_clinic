import dash_bootstrap_components as dbc
from dash import dcc, html

from src.utils.plot_utils import plot_empty_scatter


def main_display(loss_plot):
    """
    Creates the dash components within the main display in the app
    Args:
        loss_plot:          Loss plot of trainin process
    """
    main_display = html.Div(
        id="main-display",
        style={"padding": "0px 10px 0px 510px"},
        children=[
            dbc.Card(
                id="inter_graph",
                style={"width": "100%"},
                children=[
                    dbc.CardHeader("Graphical Representation", className="card-title"),
                    dbc.CardBody(
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dcc.Loading(
                                            id="loading-original",
                                            parent_className="transparent-loader-wrapper",
                                            children=[
                                                html.Img(
                                                    id="orig-img",
                                                    title="Input Image",
                                                    style={
                                                        "width": "15vw",
                                                        "height": "200px",
                                                        "padding": "0px",
                                                        "display": "inline-block",
                                                    },
                                                )
                                            ],
                                            type="circle",
                                        ),
                                        html.Img(
                                            id="ls-graph",
                                            title="",
                                            style={
                                                "width": "30vw",
                                                "height": "200px",
                                                "padding": "0px",
                                                "display": "inline-block",
                                            },
                                        ),
                                        dcc.Loading(
                                            id="loading-recons",
                                            parent_className="transparent-loader-wrapper",
                                            children=html.Img(
                                                id="rec-img",
                                                title="Reconstructed Image",
                                                style={
                                                    "width": "15vw",
                                                    "height": "200px",
                                                    "padding": "0px",
                                                    "display": "inline-block",
                                                },
                                            ),
                                            type="circle",
                                            style={
                                                "width": "15vw",
                                                "padding-right": "0px",
                                            },
                                        ),
                                    ],
                                    align="center",
                                    justify="center",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Row(
                                                html.P("Input Image"),
                                                align="center",
                                                justify="center",
                                            ),
                                            width=3,
                                        ),
                                        dbc.Col(
                                            dbc.Row(
                                                html.P("Latent Space"),
                                                align="center",
                                                justify="center",
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Row(
                                                html.P("Reconstructed Image"),
                                                align="center",
                                                justify="center",
                                            ),
                                            width=3,
                                        ),
                                    ],
                                    align="center",
                                    justify="center",
                                ),
                                dbc.Row(
                                    dcc.Slider(
                                        id="img-slider",
                                        min=0,
                                        step=1,
                                        marks=None,
                                        value=0,
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    )
                                ),
                            ]
                        ),
                        style={
                            "margin-bottom": "0rem",
                            "align-items": "center",
                            "justify-content": "center",
                        },
                    ),
                    dbc.CardFooter(id="data-size-out"),
                ],
            ),
            dbc.Card(
                id="latent-space-card",
                style={"width": "100%"},
                children=[
                    dbc.CardHeader(
                        "Latent Space Visualization", className="card-title"
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            id="latent-space-viz",
                            figure=plot_empty_scatter(),
                            style={"width": "98%", "height": "30vh"},
                        )
                    ),
                ],
            ),
            html.Div(loss_plot),
            dcc.Interval(id="interval", interval=5 * 1000, n_intervals=0),
        ],
    )
    return main_display
