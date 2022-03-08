import os
import dash
from dash import Dash, html, dcc, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pathlib
import PIL.Image as Image
import plotly.graph_objects as go
import uuid

from helper_utils import SimpleJob
from helper_utils import plot_figure, get_bottleneck, get_job, generate_loss_plot
import templates


### GLOBAL VARIABLES
DATA_DIR = str(os.environ['DATA_DIR'])
DATA_PATH = "data/mixed_small_32x32.npz"
DATA = np.load(DATA_PATH)   # making reference dataset
MODEL_DATABASE = {"The Model": "path-to-model"} # hardcoded model database as dict
USER = 'admin'


#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = "MLExchange | Data Clinic"

### BEGIN DASH CODE ###
header = templates.header()


# Training Parameters
TRAINING_PARAMS = [
    dbc.FormGroup([
        dbc.Label('Target Width'),
        dbc.Input(id='target_width', debounce=True, type="int", value=32),
    ]),
    dbc.FormGroup([
        dbc.Label('Target Height'),
        dbc.Input(id='target_height', type="int", value=32),
    ]),
    dbc.FormGroup([
        dbc.Label('Shuffle Training Data'),
        dbc.RadioItems(
            id='shuffle',
            options=[
               {'label': 'True', 'value': True},
               {'label': 'False', 'value': False},
            ],
            value = True
        )
    ]),
    dbc.FormGroup([
        dbc.Label('Batch Size'),
        dcc.Slider(id='batch_size',
                  min=16,
                  max=128,
                  value=32,
                  step=16,
                  tooltip={'always_visible': True,
                           'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Validation Percentage'),
        dcc.Slider(id='val_pct',
                  min=0,
                  max=100,
                  value=20,
                  step=5,
                  tooltip={'always_visible': True,
                           'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Latent Dimension'),
        dcc.Slider(id='latent_dim',
                  min=0,
                  max=1000,
                  value=32,
                  step=1,
                  tooltip={'always_visible': True,
                           'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Base Channel Size'),
        dcc.Slider(id='base_channel_size',
                  min=0,
                  max=1000,
                  value=32,
                  step=1,
                  tooltip={'always_visible': True,
                           'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Number of epochs'),
        dcc.Slider(id='num_epochs',
                   min=1,
                   max=1000,
                   value=3,
                   tooltip={'always_visible': True,
                            'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Optimizer'),
        dcc.Dropdown(
            id='optimizer',
            options=[
                {'label': 'Adadelta', 'value': 'Adadelta'},
                {'label': 'Adagrad', 'value': 'Adagrad'},
                {'label': 'Adam', 'value': 'Adam'},
                {'label': 'AdamW', 'value': 'AdamW'},
                {'label': 'SparseAdam', 'value': 'SparseAdam'},
                {'label': 'Adamax', 'value': 'Adamax'},
                {'label': 'ASGD', 'value': 'ASGD'},
                {'label': 'LBFGS', 'value': 'LBFGS'},
                {'label': 'RMSprop', 'value': 'RMSprop'},
                {'label': 'Rprop', 'value': 'Rprop'},
                {'label': 'SGD', 'value': 'SGD'}
            ],
            value = 'Adam'
        )
    ]),
    dbc.FormGroup([
        dbc.Label('Criterion'),
        dcc.Dropdown(
            id='criterion',
            options=[
                {'label': 'L1Loss', 'value': 'L1Loss'},
                {'label': 'MSELoss', 'value': 'MSELoss'},
                {'label': 'CrossEntropyLoss', 'value': 'CrossEntropyLoss'},
                {'label': 'CTCLoss', 'value': 'CTCLoss'},
                {'label': 'NLLLoss', 'value': 'NLLLoss'},
                {'label': 'PoissonNLLLoss', 'value': 'PoissonNLLLoss'},
                {'label': 'GaussianNLLLoss', 'value': 'GaussianNLLLoss'},
                {'label': 'KLDivLoss', 'value': 'KLDivLoss'},
                {'label': 'BCELoss', 'value': 'BCELoss'},
                {'label': 'BCEWithLogitsLoss', 'value': 'BCEWithLogitsLoss'},
                {'label': 'MarginRankingLoss', 'value': 'MarginRankingLoss'},
                {'label': 'HingeEnbeddingLoss', 'value': 'HingeEnbeddingLoss'},
                {'label': 'MultiLabelMarginLoss', 'value': 'MultiLabelMarginLoss'},
                {'label': 'HuberLoss', 'value': 'HuberLoss'},
                {'label': 'SmoothL1Loss', 'value': 'SmoothL1Loss'},
                {'label': 'SoftMarginLoss', 'value': 'SoftMarginLoss'},
                {'label': 'MutiLabelSoftMarginLoss', 'value': 'MutiLabelSoftMarginLoss'},
                {'label': 'CosineEmbeddingLoss', 'value': 'CosineEmbeddingLoss'},
                {'label': 'MultiMarginLoss', 'value': 'MultiMarginLoss'},
                {'label': 'TripletMarginLoss', 'value': 'TripletMarginLoss'},
                {'label': 'TripletMarginWithDistanceLoss', 'value': 'TripletMarginWithDistanceLoss'}
            ],
            value = 'MSELoss'
        )
    ]),
    dbc.FormGroup([
        dbc.Label('Learning Rate'),
        dbc.Input(id='learning_rate', type="float", value=0.001)
    ]),
    dbc.FormGroup([
        dbc.Label('Seed'),
        dbc.Input(id='seed', type="int", value=0)])
]


TESTING_PARAMS = [
    dbc.FormGroup([
        dbc.Label('Target Width'),
        dbc.Input(id='target_width', debounce=True, type="int", value=32),
    ]),
    dbc.FormGroup([
        dbc.Label('Target Height'),
        dbc.Input(id='target_height', type="int", value=32),
    ]),
    dbc.FormGroup([
        dbc.Label('Batch Size'),
        dcc.Slider(id='batch_size',
                  min=16,
                  max=128,
                  value=32,
                  step=16,
                  tooltip={'always_visible': True,
                           'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Seed'),
        dbc.Input(id='seed', type="int", value=0)
    ])
]


SIDEBAR = [
    dbc.Card(
        id="sidebar",
        children=[
            dbc.CardHeader("Exploring Data with Machine Learning"),
            dbc.CardBody([
                dbc.FormGroup([
                    dbc.Label('Action'),
                    dcc.Dropdown(
                        id='action',
                        options=[
                            {'label': 'Model Training', 'value': 'train_model'},
                            # {'label': 'Latent Space Exploration', 'value': 'evaluate_model'},
                            {'label': 'Test Prediction using Model', 'value': 'prediction_model'},
                        ],
                        value='train_model')
                ]),
                dbc.FormGroup([
                    dbc.Label('Model'),
                    dcc.Dropdown(
                        id='model-selection',
                        options=[
                            {'label': 'PyTorch Autoencoder', 'value': 'pytorch-auto'}],
                        value='pytorch-auto')
                ])
            ])
        ]
    ),
    dbc.Card(
        children=[
            dbc.CardHeader("Parameters"),
            dbc.CardBody([html.Div(id='app-parameters'),
                          dbc.Button('Execute',
                                     id='execute',
                                     n_clicks=0,
                                     className='m-1',
                                     style={'width': '100%', 'justify-content': 'center'})
                          ])
        ]
    )
]


LOSS_PLOT = dbc.Collapse(id = 'show-plot',
                         children = dbc.Card(id="plot-card",
                                             children=[dbc.CardHeader("Loss Plot"),
                                                       dbc.CardBody([
                                                           dcc.Graph(id='loss-plot',
                                                                     style={'width':'100%', 'height': '20rem'})])
                                                       ])),


# Job Status Display
JOB_STATUS = dbc.Card(
    children=[
            dbc.CardHeader("List of Jobs"),
            dbc.CardBody(
                children=[
                    dash_table.DataTable(
                        id='jobs-table',
                        columns=[
                            {'name': 'Job ID', 'id': 'job_id'},
                            {'name': 'Type', 'id': 'job_type'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Parameters', 'id': 'parameters'},
                            {'name': 'Experiment ID', 'id': 'experiment_id'},
                            {'name': 'Logs', 'id': 'job_logs'}
                        ],
                        data=[],
                        hidden_columns=['job_id', 'experiment_id'],
                        row_selectable='single',
                        style_cell={'padding': '1rem',
                                    'textAlign': 'left',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'maxWidth': 0},
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
                        style_table={'height': '30rem', 'overflowY': 'auto', 'overflowX': 'scroll'}
                    )
                ],
            ),
        dbc.Modal([
            dbc.ModalHeader("Job Logs"),
            dbc.ModalBody(id='log-display'),
            dbc.ModalFooter(dbc.Button("Close", id="modal-close", className="ml-auto")),
            ],
            id='log-modal',
            size='xl')
    ]
)


# main section with interactive graph (card 1) and job table (card 2)
column_02 = html.Div([
    dbc.Card(
        id="inter_graph",
        style={"width" : "100%"},
        children=[
            dbc.CardHeader("Graphical Representation", className="card-title"),
            dbc.CardBody(
                dbc.Col(
                    [dbc.Row([
                        html.Img(id='orig_img', title="Original Image",
                                 style={'width':'15vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'}),
                        html.Img(id='ls_graph', title='',
                                 style={'width':'30vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'}),
                        html.Img(id='rec_img', title="Reconstructed Image",
                                 style={'width':'15vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'})
                        ], align="center", justify='center'),
                    dbc.Row([
                        dbc.Col(dbc.Row(html.P('Original Image'), align="center", justify='center'), width=3),
                        dbc.Col(dbc.Row(html.P('Latent Space'), align="center", justify='center')),
                        dbc.Col(dbc.Row(html.P('Reconstructed Image') ,align="center", justify='center'), width=3),
                        ], align="center", justify='center'),
                    dbc.Label('Image: '),
                    dcc.Slider(id='img-slider',
                               min=0,
                               value=0,
                               tooltip={'always_visible': True, 'placement': 'bottom'})]
                ), style={'margin-bottom': '0rem', 'align-items': 'center', 'justify-content': 'center'}
            ),
        dbc.CardFooter(id='latent-size-out')
    ]),
    dbc.Row(LOSS_PLOT),
    dbc.Row(JOB_STATUS),
    dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0)
])

##### DEFINE LAYOUT ####
app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row(
                    [dbc.Col(SIDEBAR, width=4),
                     dbc.Col(column_02, width=8),
                     html.Div(id='dummy-output')]
                )
            ],
            fluid=True
        )
    ]
)

##### CALLBACKS ####
@app.callback(
    Output('ls_graph', 'src'),
    Output('latent-size-out', 'children'),
    Input('latent_dim', 'value'),
)
def update_latent_space_graph(ls_var):
    '''
    This callback updates the latent space graph
    Args:
        ls_var:         Latent space value
    Returns:
        bottleneck_graph
    '''
    ratio = 400/(DATA['x_train'].shape[1]*DATA['x_train'].shape[2])       # ratio between flatten input data and selected latent space size
    return get_bottleneck(ls_var*ratio), 'Latent Space Dimension: '+str(ls_var)


@app.callback(
    Output('app-parameters', 'children'),
    Input('model-selection', 'value'),
    Input('action', 'value'),
    prevent_intial_call=True)
def load_parameters_and_content(model_selection, action_selection):
    '''
    This callback dynamically populates the parameters and contents of the website according to the selected action &
    model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in MLCoach)
    Returns:
        app-parameters:     Parameters according to the selected model & action
    '''
    parameters = []
    if model_selection == 'pytorch-auto' and action_selection == 'train_model':
        parameters = TRAINING_PARAMS.copy()
    if model_selection == 'pytorch-auto' and action_selection == 'prediction_model':
        parameters = TESTING_PARAMS.copy()
    return parameters


@app.callback(
    [Output('orig_img', 'src'),
     Output('rec_img', 'src'),
     Output('img-slider', 'max')],
    Input('img-slider', 'value'),
    Input('jobs-table', 'selected_rows'),
    State('action', 'value'),
    State('jobs-table', 'data'),
)
def refresh_image(img_ind, row, action_selection, data_table):
    '''
    This callback updates the image in the display
    Args:
        img_ind:            Index of image according to the slider value
        action_selection:   Action selection (train vs test set)
    Returns:
        img-output:         Output figure
        img-reconst-output: Reconstructed output (if prediction is selected, ow. input figure)
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'jobs-table.selected_rows' in changed_id and data_table[row[0]]['job_type'] == 'prediction_model':
        try:
            origimg = Image.fromarray((np.squeeze(DATA['x_test'][img_ind]*255)).astype(np.uint8))
            origimg = plot_figure(origimg)
            recimg = origimg
            slider_max = DATA['x_test'].shape[0] - 1
            return plot_figure(origimg), plot_figure(recimg), slider_max
        except Exception as e:
            print(e)
    if action_selection in ['train_model', 'transfer_learning']:
        origimg = Image.fromarray((np.squeeze(DATA['x_train'][img_ind]*255)).astype(np.uint8))
        origimg = plot_figure(origimg)
        recimg = origimg
        slider_max = DATA['x_train'].shape[0] - 1
    return origimg, recimg,  slider_max


@app.callback(
    Output('jobs-table', 'data'),
    Output('loss-plot', 'figure'),
    Output('show-plot', 'is_open'),
    Output('log-modal', 'is_open'),
    Output('log-display', 'children'),
    Output('jobs-table', 'active_cell'),
    Input('interval', 'n_intervals'),
    Input('jobs-table', 'selected_rows'),
    Input('jobs-table', 'active_cell'),
    Input('modal-close', 'n_clicks'),
    prevent_initial_call=True
)
def update_table(n, row, active_cell, close_clicks):
    '''
    This callback updates the job table, loss plot, and results according to the job status in the compute service.
    Args:
        n:              Time intervals that triggers this callback
        row:            Selected row (job)
    Returns:
        jobs-table:     Updates the job table
        show-plot:      Shows/hides the loss plot
        loss-plot:      Updates the loss plot according to the job status (logs)
        results:        Testing results (probability)
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'modal-close.n_clicks' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update, False, dash.no_update, None
    job_list = get_job(USER, 'data_clinic')
    data_table = []
    if job_list is not None:
        for job in job_list:
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  job_type=job['job_type'],
                                  status=job['status'],
                                  parameters=str(job['container_kwargs']['parameters']),
                                  experiment_id=job['container_kwargs']['experiment_id'],
                                  job_logs=job['container_logs'])
                              )
    is_open = dash.no_update
    log_display = dash.no_update
    if active_cell:
        row_log = active_cell["row"]
        col_log = active_cell["column_id"]
        if col_log == 'job_logs':       # show job logs
            is_open = True
            log_display = dcc.Textarea(value=data_table[row_log]["job_logs"],
                                       style={'width': '100%', 'height': '30rem', 'font-family':'monospace'})
        if col_log == 'parameters':     # show job parameters
            is_open = True
            log_display = dcc.Textarea(value=str(job['container_kwargs']['parameters']),
                                       style={'width': '100%', 'height': '30rem', 'font-family': 'monospace'})
    style_fig = {'display': 'none'}
    fig = go.Figure(go.Scatter(x=[], y=[]))
    show_plot = False
    if row:
        log = data_table[row[0]]["job_logs"]
        if log:
            if data_table[row[0]]['job_type'] == 'train_model':
                start = log.find('epoch')
                if start > -1 and len(log) > start + 5:
                    fig = generate_loss_plot(log, start)
                    show_plot = True
                    style_fig = {'width': '100%', 'display': 'block'}
    return data_table, fig, show_plot, is_open, log_display, None


@app.callback(
    Output('dummy-output', 'children'),
    Input('execute', 'n_clicks'),
    [State('app-parameters', 'children'),
     State('action', 'value'),
     State('jobs-table', 'data'),
     State('jobs-table', 'selected_rows')],
    prevent_intial_call=True)
def execute(clicks, children, action_selection, job_data, row):
    '''
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        clicks:             Execute button triggers this callback
        children:           Model parameters
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
    Returns:
        None
    '''
    if clicks > 0:
        contents = []
        experiment_id = str(uuid.uuid4())
        out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        out_path.mkdir(parents=True, exist_ok=True)
        input_params = {}
        if bool(children):
            for child in children:
                key = child["props"]["children"][1]["props"]["id"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        json_dict = input_params
        if action_selection == 'train_model':
            command = "python3 src/train_model.py"
            directories = [DATA_PATH, str(out_path)]
        else:
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
        if action_selection == 'prediction_model':
            command = "python3 src/predict_model.py"
            directories = [DATA_PATH, str(in_path) , str(out_path)]
        job = SimpleJob(user=USER,
                        job_type=action_selection,
                        description='',
                        deploy_location='local',
                        gpu=True,
                        data_uri='{}'.format(DATA_DIR),
                        container_uri='mlexchange/unsupervised-classifier',
                        container_cmd=command,
                        container_kwargs={'parameters': json_dict,
                                          'directories': directories,
                                          'experiment_id': experiment_id}
                        )
        job.launch_job()
        return contents
    return []


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8052)
