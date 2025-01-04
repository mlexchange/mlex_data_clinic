import dash_bootstrap_components as dbc


def training_stats_plot():
    training_stats_plot = dbc.Collapse(
        id="show-plot",
        children=dbc.Card(
            id="plot-card",
            children=[
                dbc.CardHeader("Training Stats"),
                dbc.CardBody(
                    id="stats-card-body",
                ),
            ],
        ),
    )
    return training_stats_plot
