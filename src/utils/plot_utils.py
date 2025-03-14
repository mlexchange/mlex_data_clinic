import base64

import dash_bootstrap_components as dbc
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from dash_iconify import DashIconify


def plot_figure(image):
    """
    Plots images in frontend
    Args:
        image:  Image to plot
    Returns:
        plot in base64 format
    """
    try:
        h, w = image.size
    except Exception:
        h, w, c = image.size
    fig = px.imshow(image, height=200, width=200 * w / h)
    fig.update_xaxes(
        showgrid=False, showticklabels=False, zeroline=False, fixedrange=True
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, zeroline=False, fixedrange=True
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    try:
        fig.update_traces(dict(showscale=False, coloraxis=None))
    except Exception as e:
        print(f"plot error {e}")
    png = plotly.io.to_image(fig, format="jpg")
    png_base64 = base64.b64encode(png).decode("ascii")
    return "data:image/jpg;base64,{}".format(png_base64)


def get_bottleneck(ls_var, width, height, annotations=True):
    """
    Plots the latent space representation
    Args:
        ls_var:         latent space value
        width:          data width
        height:         data height
        annotations:    Bool
    Returns:
        plot with graphical representation of the latent space in base64 format
    """
    # ratio between flatten input data and selected latent space size
    ratio = 400 / (width * height)
    annotation1 = str(width) + "x" + str(height)  # target data size
    annotation2 = str(ls_var) + "x1"  # target latent space
    # if the latent space is larger than the data dimension (flatten), the bottleneck is shown in red
    if ls_var > width * height:
        color = "rgba(238, 69, 80, 1)"
    else:
        color = "rgba(168, 216, 234, 1)"
    # adjusting the latent space with respect to the images size in frontend
    ls_var = ls_var * ratio
    x = [-200, -200, 0, 200, 200, 0, -200]
    y = [-200, 200, ls_var / 2, 200, -200, -ls_var / 2, -200]
    fig = go.Figure(
        go.Scatter(x=x, y=y, fill="toself", fillcolor=color, line_color=color)
    )
    fig.add_shape(
        type="rect",
        x0=-1,
        y0=ls_var / 2,
        x1=1,
        y1=-ls_var / 2,
        fillcolor="RoyalBlue",
        line_color="RoyalBlue",
    )
    fig.update_traces(marker_size=1, hoverinfo="skip")
    if annotations:
        fig.add_annotation(
            x=-187, y=-25, text=annotation1, textangle=270, font={"size": 28}
        )
        fig.add_annotation(
            x=199, y=-25, text=annotation1, textangle=270, font={"size": 28}
        )
        fig.add_annotation(
            x=-10,
            y=0,
            text=annotation2,
            textangle=270,
            font={"size": 28},
            showarrow=False,
        )
    fig.update_xaxes(
        range=[-200, 200],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        range=[-200, 200],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    png = plotly.io.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")
    return "data:image/png;base64,{}".format(png_base64)


def generate_notification(title, color, icon, message=""):
    iconify_icon = DashIconify(
        icon=icon,
        width=24,
        height=24,
        style={"verticalAlign": "middle"},
    )
    return [
        dbc.Toast(
            id="auto-toast",
            children=[
                html.Div(
                    [
                        iconify_icon,
                        html.Span(title, style={"margin-left": "10px"}),
                    ],
                    className="d-flex align-items-center",
                ),
                html.P(message, className="mb-0"),
            ],
            duration=4000,
            is_open=True,
            color=color,
            style={
                "position": "fixed",
                "top": 66,
                "right": 10,
                "width": 350,
                "zIndex": 9999,
            },
        )
    ]


def plot_empty_scatter():
    return go.Figure(
        go.Scattergl(mode="markers"),
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(
                l=20,
                r=20,
                b=20,
                t=20,
                pad=0,
            ),
        ),
    )


def generate_scatter_data(latent_vectors):
    """
    Generate latent vectors plot
    """
    vals_names = {}
    vals = [-1 for i in range(latent_vectors.shape[0])]
    vals_names = {a: a for a in np.unique(vals).astype(int)}

    scatter_data = generate_scattergl_plot(
        latent_vectors[:, 0], latent_vectors[:, 1], vals, vals_names
    )

    fig = go.Figure(scatter_data)
    fig.update_layout(
        margin=go.layout.Margin(l=20, r=20, b=20, t=20, pad=0),
        legend=dict(tracegroupgap=20),
    )
    return fig


def generate_scattergl_plot(
    x_coords,
    y_coords,
    labels,
    label_to_string_map,
    show_legend=False,
    custom_indices=None,
):
    """
    Generates a multi-trace Scattergl plot with one trace per label,
    preserving the exact i-th ordering across all data.

    Each trace is the same length as x_coords/y_coords, but for points
    not belonging to that trace's label, we insert None. This ensures:
      - i-th point in the figure is i-th data row (helpful for selectedData).
      - Each label gets its own legend entry.
    """

    if custom_indices is None:
        custom_indices = list(range(len(x_coords)))

    # Gather unique labels in order of first appearance
    unique_labels = []
    for lbl in labels:
        if lbl not in unique_labels:
            unique_labels.append(lbl)

    traces = []
    for label in unique_labels:
        # Initialize the entire length with None
        trace_x = [None] * len(x_coords)
        trace_y = [None] * len(y_coords)
        trace_custom = [None] * len(x_coords)

        # Fill in data only where labels match
        for i, lbl in enumerate(labels):
            if lbl == label:
                trace_x[i] = x_coords[i]
                trace_y[i] = y_coords[i]
                trace_custom[i] = custom_indices[i]

        # Convert custom_indices to a 2D array if needed by Plotly
        trace_custom = np.array(trace_custom).reshape(-1, 1)

        traces.append(
            go.Scattergl(
                x=trace_x,
                y=trace_y,
                mode="markers",
                name=str(label_to_string_map[label]),
                customdata=trace_custom,
            )
        )

    fig = go.Figure(data=traces)
    if not show_legend:
        fig.update_layout(showlegend=False)
    return fig
