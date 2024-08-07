import dash_bootstrap_components as dbc
from dash import html


def resources_setup(num_processors, num_gpus):
    """
    Window for computing resources setup before job execution
    Args:
        num_processors:     Maximum number of processors at host
        num_gpus:           Maximum number of gpus at host
    """
    resources_setup = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader("Choose number of computing resources:"),
                    dbc.ModalBody(
                        children=[
                            dbc.Row(
                                [
                                    dbc.Label(
                                        f"Number of CPUs (Maximum available: {num_processors})"
                                    ),
                                    dbc.Input(id="num-cpus", type="int", value=2),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Label(
                                        f"Number of GPUs (Maximum available: {num_gpus})"
                                    ),
                                    dbc.Input(id="num-gpus", type="int", value=0),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Label("Model Name"),
                                    dbc.Input(id="model-name", type="str", value=""),
                                ]
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Submit Job", id="submit", className="ms-auto", n_clicks=0
                        )
                    ),
                ],
                id="resources-setup",
                centered=True,
                is_open=False,
            ),
        ]
    )
    return resources_setup
