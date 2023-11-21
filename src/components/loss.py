from dash import dcc
import dash_bootstrap_components as dbc


def loss_plot():
    loss_plot = dbc.Collapse(
                    id = 'show-plot',
                    children = dbc.Card(
                                    id="plot-card",
                                    children=[
                                        dbc.CardHeader("Loss Plot"),
                                        dbc.CardBody([
                                            dcc.Graph(
                                                id='loss-plot',
                                                style={
                                                    'width':'100%', 
                                                    'height': '20rem'
                                                    }
                                                )
                                            ]
                                        )
                                    ]
                                )
                    )
    return loss_plot