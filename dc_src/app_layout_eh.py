from functools import total_ordering
from locale import normalize
import numbers
from tkinter.ttk import Style
from dash import Dash
from dash import html
from dash import dcc
from dash import dash_table
from dash.dash_table.Format import Group
from dash.dependencies import Input,Output

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

# import packages for latent space interactive graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import numpy as np
from io import BytesIO
import base64

#import dash_auth
##### HELPER UTILS
#import helper_utils
##### TEMPLATE MODULES
import templates
import numpy as np

### GLOBAL VARIABLES
data_path = "/Users/computing/MLExchange/testdata/mixed_small_32x32.npz"
DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
# hardcoded model database as dict
MODEL_DATABASE = {"The Model": "path-to-model"}

# making reference dataset
data_npy = np.load(data_path)
img = data_npy['x_train'][0]
res_img = img.reshape((32, 32))

# define function to save figure as a URI
def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

# define ls_var_img for trapezoid thing
ls_var = 10
ls_var_img = (ls_var * 100) // len(res_img)

# create reference plots for latent space interactive graph
ls_graph_list = []

for ls_var_idx in range(1, 101, 1):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(xmin = -100, xmax = 100)
    ax.set_ylim(ymin = -100, ymax = 100)
    ax.axis('off')

    # Trapez_01
    x = [-100,-100,0,0]
    y = [-100,100,ls_var,-ls_var]
    ax.add_patch(patches.Polygon(xy=list(zip(x,y)), fill=True))

    # Trapez_02
    x = [0,0,100,100]
    y = [-ls_var,ls_var,100,-100]
    ax.add_patch(patches.Polygon(xy=list(zip(x,y)), fill=True))

    ls_graph_list.append(fig)
    plt.close(fig)

# plotting original image
origimg = plt.figure()
plt.imshow(res_img)
plt.axis('off')
#origimg.update_layout(sizing ="fill")
fig1_origimg = fig_to_uri(origimg)

# plotting trapezoidal representation of latent space
fig2_lsgraph = fig_to_uri(ls_graph_list[ls_var])

# plotting reconstructed image
recimg = plt.figure()
plt.imshow(res_img)
plt.axis('off')
#recimg.update_layout(sizing ="fill")
fig3_recimg = fig_to_uri(recimg)
#fig3_recimg = px.imshow(res_img)
#fig3_recimg.update_layout(height=200, width=200, coloraxis_showscale=False)
#fig3_recimg.update_xaxes(showticklabels=False)
#fig3_recimg.update_yaxes(showticklabels=False)

#####################



#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
#auth = dash_auth.BasicAuth(
#        app,
#        VALID_USERNAME_PASSWORD_PAIRS,
#        )

server = app.server
app.title = "MLExchange | Data Clinic"

### BEGIN DASH CODE ###
header = templates.header()

# sidebar for categories of data-handling
column_01 = html.Div(
    [dbc.Card(
        id="data_preproc",
        style={"width" : "100%"},
        children=[
            dbc.CardHeader("Data Pre-processing"),
            dbc.CardBody(
                [
                    dcc.Input(
                        id="latent_space_size",
                        type="number"
                    ),
                    html.Div([
                        'Datatype:',
                        dcc.RadioItems(
                            ['Float32 (0 - 255)', 'Integer (0 - 1)'],
                            'linear',
                            id='scalebar-scale'
                        )
                    ])
                ]
            )
        ]
    ),
    dbc.Card(
        id="data_ml",
        style={"width" : "100%"},
        children=[
            dbc.CardHeader("Exploring Data with Machine Learning"),
            dbc.CardBody(
                [
                ]
            )
        ]
    )]
)

# main section with interactive graph (card 1) and job table (card 2)
column_02 = html.Div([
    dbc.Card(
        id="inter_graph",
        style={"width" : "100%"},
        children=[
            dbc.CardHeader(html.H4("Graphical Representation", className="card-title")),
            dbc.CardBody(
                    html.Div([
                        html.Img(id='orig_img', src=fig1_origimg, title="Original Image",
                        style={'width':'15vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'}),
                        html.Img(id='ls_graph', src=fig2_lsgraph,
                        style={'width':'30vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'}),
                        html.Img(id='rec_img', src=fig3_recimg, title="Reconstructed Image",
                        style={'width':'15vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'})
                    ])
                    #dcc.Graph(id='rec_img', figure=fig3_recimg,
                    #    style={'width':'100%', 'display': 'inline-block'},
                    #    #style={"margin-left":"1px", "margin-right":"50px"},
                    #    config={"displayModeBar": False,
                    #    "responsive":False, "showAxisDragHandles": False, "showAxisRangeEntryBoxes": False}
                    #    )
                ),
        dbc.CardFooter(
            "Latent Space Dimension: " + str(ls_var)
        )
    ]),
    dbc.Card(
        id="job_table",
        style={"width" : "100%"},
        children=[
            dbc.CardHeader("List of Jobs"),
            dbc.CardBody(
                [dash_table.DataTable(
                id='jobs_table',
                columns=[
                    {'name': 'Job ID', 'id': 'job_id'},
                    {'name': 'Type', 'id': 'job_type'},
                    {'name': 'Status', 'id': 'status'},
                    {'name': 'Dataset', 'id': 'dataset'},
                    {'name': 'Image length', 'id': 'image_length'},
                    {'name': 'Model', 'id': 'model_name'},
                    {'name': 'Parameters', 'id': 'parameters'},
                    {'name': 'Experiment ID', 'id': 'experiment_id'},
                    {'name': 'Logs', 'id': 'job_logs'}
                ],
                data = [],
                hidden_columns = ['job_id', 'image_length', 'experiment_id', 'job_logs'],
                row_selectable='single',
                style_cell={'padding': '1rem', 'textAlign': 'left'}, #, 'maxWidth': '7rem', 'whiteSpace': 'normal'},
                fixed_rows={'headers': True},
                css=[{"selector": ".show-hide", "rule": "display: none"}],
                style_data_conditional=[
                    {'if': {'column_id': 'status', 'filter_query': '{status} = completed'},
                     'backgroundColor': 'green',
                     'color': 'white'},
                    {'if': {'column_id': 'status', 'filter_query': '{status} = failed'},
                     'backgroundColor': 'red',
                     'color': 'white'}
                ],
                style_table={'height':'18rem', 'overflowY': 'auto'}
                )]
            )
        ]
    )]
)

##### DEFINE LAYOUT ####

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row(
                    [dbc.Col(column_01, width=4), dbc.Col(column_02, width=8)]
                )
            ],
            fluid=True
        )
    ]
)

##### CALLBACKS #### (COMPLETELY NOT CORRECT)
@app.callback(
  Output('graph_ls', 'figure'),
    [Input('latent-space-size', 'value')])

# THIS ONE IS CORRECT
def update_graph(selected_ls):
    fig2_lsgraph = fig_to_uri(ls_graph_list[selected_ls])

    return fig2_lsgraph

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')