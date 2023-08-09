import os
import logging
import pathlib

import dash
from dash import html, dcc, dash_table
import dash_uploader as du
import dash_bootstrap_components as dbc

from file_manager.file_manager import FileManager
from helper_utils import get_host, get_counter, model_list_GET_call
import templates

### GLOBAL VARIABLES AND DATA LOADING
DATA_DIR = str(os.environ['DATA_DIR'])
MODEL_DATABASE = {"The Model": "path-to-model"} # hardcoded model database as dict
USER = 'admin'
DOCKER_DATA = pathlib.Path.home() / 'data'
LOCAL_DATA = str(os.environ['DATA_DIR'])
DOCKER_HOME = str(DOCKER_DATA) + '/'
LOCAL_HOME = str(LOCAL_DATA)
UPLOAD_FOLDER_ROOT = DOCKER_DATA / 'upload'
SUPPORTED_FORMATS = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
HOST_NICKNAME = str(os.environ['HOST_NICKNAME'])
num_processors, num_gpus = get_host(HOST_NICKNAME)
MODELS = model_list_GET_call()

#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Data Clinic"
app._favicon = 'mlex.ico'
dash_file_explorer = FileManager(DOCKER_DATA, UPLOAD_FOLDER_ROOT, open_explorer=False)
dash_file_explorer.init_callbacks(app)
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)

### BEGIN DASH CODE ###
header = templates.header()

SIDEBAR = [
    dbc.Card(
        id="sidebar",
        children=[
            dbc.CardHeader("Exploring Data with Machine Learning"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Label('Action'),
                    dcc.Dropdown(
                        id='action',
                        options=[
                            {'label': 'Model Training', 
                             'value': 'train_model'},
                            {'label': 'Test Prediction using Model', 
                             'value': 'prediction_model'},
                        ],
                        value='train_model')
                ]),
                dbc.Row([
                    dbc.Label('Model'),
                    dcc.Dropdown(
                        id='model-selection',
                        options=MODELS,
                        value=MODELS[0]['value'])
                ]),
                dbc.Row([
                    dbc.Label('Data'),
                    dash_file_explorer.file_explorer,
                ]),
                dbc.Button('Execute',
                           id='execute',
                           n_clicks=0,
                           className='m-1',
                           style={'width': '100%', 
                                  'justify-content': 'center'})
            ])
        ]
    ),
    dbc.Card(
        children=[
            dbc.CardHeader("Parameters"),
            dbc.CardBody([html.Div(id='app-parameters')])
        ]
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("Warning"),
            dbc.ModalBody(id="warning-msg"),
            dbc.ModalFooter([
                dbc.Button(
                    "OK", 
                    id="ok-button", 
                    color='danger', 
                    outline=False,
                    className="ms-auto", 
                    n_clicks=0
                ),
            ]),
        ],
        id="warning-modal",
        is_open=False,
    ),
    dcc.Store(id='warning-cause', 
              data=''),
    dcc.Store(id='counters', 
              data=get_counter(USER))
]


LOSS_PLOT = dbc.Collapse(id = 'show-plot',
                         children = dbc.Card(id="plot-card",
                                             children=[dbc.CardHeader("Loss Plot"),
                                                       dbc.CardBody([
                                                           dcc.Graph(id='loss-plot',
                                                                     style={'width':'100%', 
                                                                            'height': '20rem'})])
                                                       ])),


# Job Status Display
JOB_STATUS = dbc.Card(
    children=[
            dbc.CardHeader("List of Jobs"),
            dbc.CardBody(
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button("Deselect Row", 
                                           id="deselect-row",
                                           style={'width': '100%', 'margin-bottom': '1rem'})
                            ),
                            dbc.Col(
                                dbc.Button("Stop Job", 
                                           id="stop-row", 
                                           color='warning',
                                           style={'width': '100%'})
                            ),
                            dbc.Col(
                                dbc.Button("Delete Job", 
                                           id="delete-row", 
                                           color='danger',
                                           style={'width': '100%'})
                            ),
                        ]
                    ),
                    dash_table.DataTable(
                        id='jobs-table',
                        columns=[
                            {'name': 'Job ID', 'id': 'job_id'},
                            {'name': 'Type', 'id': 'job_type'},
                            {'name': 'Name', 'id': 'name'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Parameters', 'id': 'parameters'},
                            {'name': 'Experiment ID', 'id': 'experiment_id'},
                            {'name': 'Dataset', 'id': 'dataset'},
                            {'name': 'Logs', 'id': 'job_logs'}
                        ],
                        data=[],
                        hidden_columns=['job_id', 'experiment_id', 'dataset'],
                        row_selectable='single',
                        style_cell={'padding': '1rem',
                                    'textAlign': 'left',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'maxWidth': 0},
                        fixed_rows={'headers': True},
                        css=[{"selector": ".show-hide", "rule": "display: none"}],
                        page_size=8,
                        style_data_conditional=[
                            {'if': {'column_id': 'status', 'filter_query': '{status} = complete'},
                             'backgroundColor': 'green',
                             'color': 'white'},
                            {'if': {'column_id': 'status', 'filter_query': '{status} = failed'},
                             'backgroundColor': 'red',
                             'color': 'white'},
                        ],
                        style_table={'height': '30rem', 
                                     'overflowY': 'auto'}
                    )
                ],
            ),
        dbc.Modal(
            [
                dbc.ModalHeader("Warning"),
                dbc.ModalBody('Models cannot be recovered after deletion.  \
                                Do you still want to proceed?"'),
                dbc.ModalFooter([
                    dbc.Button(
                        "OK", 
                        id="confirm-delete-row", 
                        color='danger', 
                        outline=False,
                        className="ms-auto", 
                        n_clicks=0
                    ),
                ]),
            ],
            id="delete-modal",
            is_open=False,
        ),
        dbc.Modal([
            dbc.ModalHeader("Job Logs"),
            dbc.ModalBody(id='log-display'),
            dbc.ModalFooter(dbc.Button("Close", 
                                       id="modal-close", 
                                       className="ml-auto")),
            ],
            id='log-modal',
            size='xl')
    ]
)


# main section with interactive graph (card 1) and job table (card 2)
MAIN_COLUMN = html.Div([
    dbc.Card(
        id="inter_graph",
        style={"width" : "100%"},
        children=[
            dbc.CardHeader("Graphical Representation", className="card-title"),
            dbc.CardBody(
                dbc.Col(
                    [dbc.Row([
                        html.Img(id='orig_img', 
                                 title="Input Image",
                                 style={'width':'15vw', 
                                        'height': '200px', 
                                        'padding':'0px', 
                                        'display': 'inline-block'}),
                        html.Img(id='ls_graph', title='',
                                 style={'width':'30vw', 
                                        'height': '200px', 
                                        'padding':'0px', 
                                        'display': 'inline-block'}),
                        html.Img(id='rec_img', title="Reconstructed Image",
                                 style={'width':'15vw', 
                                        'height': '200px', 
                                        'padding':'0px', 
                                        'display': 'inline-block'})
                        ], align="center", justify='center'),
                    dbc.Row([
                        dbc.Col(dbc.Row(html.P('Input Image'), 
                                        align="center", 
                                        justify='center'), 
                                width=3),
                        dbc.Col(dbc.Row(html.P('Latent Space'), 
                                        align="center", 
                                        justify='center')
                                ),
                        dbc.Col(dbc.Row(html.P('Reconstructed Image'),
                                        align="center", 
                                        justify='center'), 
                                width=3),
                        ], align="center", justify='center'),
                    dbc.Row([
                        dbc.Col(
                            dbc.Label('Image: ', id='current-image-label'),
                        ),
                        dbc.Col(
                            dcc.Input(id='img-slider',
                                      min=0,
                                      type='number',
                                      value=0)
                        )
                        ]),
                    ]), 
                    style={'margin-bottom': '0rem', 
                               'align-items': 'center', 
                               'justify-content': 'center'}
            ),
        dbc.CardFooter(id='data-size-out')
    ]),
    html.Div(LOSS_PLOT),
    JOB_STATUS,
    dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0),
])


RESOURCES_SETUP = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("Choose number of computing resources:"),
                dbc.ModalBody(
                    children=[
                        dbc.Row([
                            dbc.Label(f'Number of CPUs (Maximum available: {num_processors})'),
                            dbc.Input(id='num-cpus',
                                      type="int",
                                      value=2)]),
                        dbc.Row([
                            dbc.Label(f'Number of GPUs (Maximum available: {num_gpus})'),
                            dbc.Input(id='num-gpus',
                                      type="int",
                                      value=0)]),
                        dbc.Row([
                            dbc.Label('Model Name'),
                            dbc.Input(id='model-name',
                                      type="str",
                                      value="")])
                    ]),
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


##### DEFINE LAYOUT ####
app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row(
                    [dbc.Col(SIDEBAR, width=4),
                     dbc.Col(MAIN_COLUMN, width=8),
                     html.Div(id='dummy-output')]
                ),
                RESOURCES_SETUP
            ],
            fluid=True
        )
    ]
)
