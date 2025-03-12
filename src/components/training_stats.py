import dash_bootstrap_components as dbc


def training_stats_plot():
    training_stats_plot = dbc.Modal(
        id="show-plot",
        children=[
            dbc.ModalHeader("Training Stats"),
            dbc.ModalBody(
                id="stats-card-body",
                style={"height": "30%", "overflow": "auto"},
            ),
        ],
        size="xl",
    )
    return training_stats_plot
