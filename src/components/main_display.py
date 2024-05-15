import dash_bootstrap_components as dbc
from dash import dcc, html


def main_display(loss_plot, job_table):
    """
    Creates the dash components within the main display in the app
    Args:
        loss_plot:          Loss plot of trainin process
        job_table:          Job table
    """
    main_display = html.Div(
        [
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
                                        html.Img(
                                            id="orig_img",
                                            title="Input Image",
                                            style={
                                                "width": "15vw",
                                                "height": "200px",
                                                "padding": "0px",
                                                "display": "inline-block",
                                            },
                                        ),
                                        html.Img(
                                            id="ls_graph",
                                            title="",
                                            style={
                                                "width": "30vw",
                                                "height": "200px",
                                                "padding": "0px",
                                                "display": "inline-block",
                                            },
                                        ),
                                        html.Img(
                                            id="rec_img",
                                            title="Reconstructed Image",
                                            style={
                                                "width": "15vw",
                                                "height": "200px",
                                                "padding": "0px",
                                                "display": "inline-block",
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
            html.Div(loss_plot),
            job_table,
            dcc.Interval(id="interval", interval=5 * 1000, n_intervals=0),
        ]
    )
    return main_display
