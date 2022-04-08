import os
import json
import dash
from dash import Dash, html, dcc, dcc, dash_table, MATCH, ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pathlib
import PIL.Image as Image
import plotly.graph_objects as go
import uuid

from helper_utils import SimpleJob
from helper_utils import plot_figure, get_bottleneck, get_job, generate_loss_plot, load_from_dir, str_to_dict, \
                         model_list_GET_call, get_gui_components
from kwarg_editor import JSONParameterEditor
import templates


### GLOBAL VARIABLES AND DATA LOADING
DATA_DIR = str(os.environ['DATA_DIR'])
DATA_PATH = "data/mixed_small_32x32.npz"
# DATA_PATH = 'data'
if os.path.splitext(DATA_PATH)[-1] == '.npz':
    DATA = np.load(DATA_PATH)   # making reference dataset
else:
    DATA = load_from_dir(DATA_PATH)
MODEL_DATABASE = {"The Model": "path-to-model"} # hardcoded model database as dict
USER = 'admin'
MODELS = model_list_GET_call()


#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)#, suppress_callback_exceptions=True)
server = app.server
app.title = "MLExchange | Data Clinic"

### BEGIN DASH CODE ###
header = templates.header()

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
                        options=MODELS,
                        value=MODELS[0]['value'])
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
                            {'if': {'column_id': 'status', 'filter_query': '{status} = complete'},
                             'backgroundColor': 'green',
                             'color': 'white'},
                            {'if': {'column_id': 'status', 'filter_query': '{status} = failed'},
                             'backgroundColor': 'red',
                             'color': 'white'},
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
                        html.Img(id='orig_img', title="Input Image",
                                 style={'width':'15vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'}),
                        html.Img(id='ls_graph', title='',
                                 style={'width':'30vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'}),
                        html.Img(id='rec_img', title="Reconstructed Image",
                                 style={'width':'15vw', 'height': '200px', 'padding':'0px', 'display': 'inline-block'})
                        ], align="center", justify='center'),
                    dbc.Row([
                        dbc.Col(dbc.Row(html.P('Input Image'), align="center", justify='center'), width=3),
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
        dbc.CardFooter(id='data-size-out')
    ]),
    html.Div(LOSS_PLOT),
    JOB_STATUS,
    dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0)
])


RESOURCES_SETUP = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("Resources Setup"),
                dbc.ModalBody(
                    children=[
                        dbc.FormGroup([
                                dbc.Label('Number of CPUs'),
                                dbc.Input(id='num-cpus',
                                          type="int",
                                          value=1)]),
                        dbc.FormGroup([
                                dbc.Label('Number of GPUs'),
                                dbc.Input(id='num-gpus',
                                          type="int",
                                          value=0)])
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
                     dbc.Col(column_02, width=8),
                     html.Div(id='dummy-output')]
                ),
                RESOURCES_SETUP
            ],
            fluid=True
        )
    ]
)

##### CALLBACKS ####
@app.callback(
    Output('app-parameters', 'children'),
    Input('model-selection', 'value'),
    Input('action', 'value')
)
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
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(_id={'type': 'parameter_editor'},        # pattern match _id (base id), name
                                   json_blob=parameters,
                                   )
    gui_item.init_callbacks(app)
    return gui_item


@app.callback(
    [Output('orig_img', 'src'),
     Output('rec_img', 'src'),
     Output('ls_graph', 'src'),
     Output('img-slider', 'max'),
     Output('img-slider', 'value'),
     Output('data-size-out', 'children')],
    Input({'type': ALL, 'param_key': 'latent_dim', 'name': 'latent_dim', 'layer': 'input'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_width', 'name': 'target_width'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_height', 'name': 'target_height'}, 'value'),
    Input('img-slider', 'value'),
    Input('jobs-table', 'selected_rows'),
    Input('action', 'value'),
    State('jobs-table', 'data'),
    prevent_initial_call=True
)
def refresh_image(ls_var, target_width, target_height, img_ind, row, action_selection, data_table):
    '''
    This callback updates the images in the display
    Args:
        ls_var:             Latent space value
        target_width:       Target data width (if resizing)
        target_height:      Target data height (if resizing)
        img_ind:            Index of image according to the slider value
        row:                Selected job (model)
        action_selection:   Action selection (train vs test set)
        data_table:         Data in table of jobs
    Returns:
        img-output:         Output figure
        img-reconst-output: Reconstructed output (if prediction is selected, ow. blank image)
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # if 'jobs-table.selected_rows' in changed_id or 'img-slider.value' in changed_id:
    #     if row:
    #
    if 'data_name' not in locals():
        if action_selection in ['train_model', 'transfer_learning']:
            data_name = 'x_train'
            if len(ls_var)>0:
                ls_var = int(ls_var[0])
                target_width = int(target_width[0])
                target_height = int(target_height[0])
                ls_plot = get_bottleneck(ls_var, target_width, target_height)
            else:
                ls_plot = dash.no_update
        else:
            data_name = 'x_test'
            if row:
                if data_table[row[0]]['job_type'] == 'train_model':
                    train_params = str_to_dict(data_table[row[0]]['parameters'])
                    ls_var = int(train_params['latent_dim'])
                    target_width = int(train_params['target_width'])
                    target_height = int(train_params['target_height'])
                    ls_plot = get_bottleneck(ls_var, target_width, target_height)
                else:
                    data_name = 'x_test'
                    job_id = data_table[row[0]]['experiment_id']
                    reconstructed_path = 'data/mlexchange_store/{}/{}/reconstructed_images.npy'.format(USER, job_id)
                    try:
                        reconstructed_data = np.load(reconstructed_path)
                        slider_max = reconstructed_data.shape[0]
                        img_ind = min(slider_max, img_ind)
                        reconst_img = Image.fromarray(
                            (np.squeeze(reconstructed_data[img_ind] * 255)).astype(np.uint8))
                    except Exception:
                        print('Reconstructed images are not ready')
                    indx = data_table[row[0]]['parameters'].find('Training Parameters:')
                    train_params = str_to_dict(data_table[row[0]]['parameters'][indx + 21:])
                    ls_var = int(train_params['latent_dim'])
                    target_width = int(train_params['target_width'])
                    target_height = int(train_params['target_height'])
                    if 'img-slider.value' in changed_id:
                        ls_plot = dash.no_update
                    else:
                        ls_plot = get_bottleneck(ls_var, target_width, target_height)
            else:
                ls_plot = get_bottleneck(1,1,1, False)
                target_width = None
    if type(DATA) != dict:                        # loading from array
        slider_max = DATA[data_name].shape[0] - 1
        img_ind = min(slider_max, img_ind)
        origimg = Image.fromarray((np.squeeze(DATA[data_name][img_ind] * 255)).astype(np.uint8))
    else:                                               # loading from directory
        slider_max = len(DATA[data_name]) - 1
        img_ind = min(slider_max, img_ind)
        origimg = Image.open(DATA[data_name][img_ind])
    (width, height) = origimg.size
    if 'reconst_img' not in locals():
        reconst_img = Image.fromarray((np.zeros(origimg.size).astype(np.uint8)))
    if target_width:
        origimg = plot_figure(origimg.resize((target_width, target_height)))
        recimg = plot_figure(reconst_img.resize((target_width, target_height)))
        data_size = 'Original Image: (' + str(width) + 'x' + str(height) + '). Resized Image: (' + \
                    str(target_width) + 'x' + str(target_height) + ').'
    else:
        origimg = plot_figure(origimg)
        recimg = plot_figure(reconst_img)
        data_size = 'Original Image: (' + str(width) + 'x' + str(height) + \
                    '). Choose a trained model to update the graph.'
    return origimg, recimg, ls_plot, slider_max, img_ind, data_size


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
        active_cell:    Selected cell in table of jobs
        close_clicks:   Close pop-up window
    Returns:
        jobs-table:     Updates the job table
        loss-plot:      Updates the loss plot according to the job status (logs)
        show-plot:      Shows/hides the loss plot
        log-modal:      Open/close pop-up window
        log-display:    Contents of pop-up window
        jobs-table:     Selects/deselects the active cell in job table. Without this output, the pop-up window will not
                        close
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'modal-close.n_clicks' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update, False, dash.no_update, None
    job_list = get_job(USER, 'data_clinic')
    data_table = []
    if job_list is not None:
        for job in job_list:
            params = str(job['job_kwargs']['kwargs']['params'])
            if job['job_kwargs']['kwargs']['job_type'] != 'train_model':
                params = params + '\nTraining Parameters: ' + str(job['job_kwargs']['kwargs']['train_params'])
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  job_type=job['job_kwargs']['kwargs']['job_type'],
                                  status=job['status']['state'],
                                  parameters=params,
                                  experiment_id=job['job_kwargs']['kwargs']['experiment_id'],
                                  job_logs=job['logs'])
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
            log_display = dcc.Textarea(value=data_table[row_log]["parameters"],
                                       style={'width': '100%', 'height': '30rem', 'font-family': 'monospace'})
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
    return data_table, fig, show_plot, is_open, log_display, None


@app.callback(
    Output('resources-setup', 'is_open'),
    Input('execute', 'n_clicks'),
    Input('submit', 'n_clicks'),
    [State('app-parameters', 'children'),
     State('num-cpus', 'value'),
     State('num-gpus', 'value'),
     State('action', 'value'),
     State('jobs-table', 'data'),
     State('jobs-table', 'selected_rows')],
    prevent_intial_call=True)
def submit(execute, submit, children, num_cpus, num_gpus, action_selection, job_data, row):
    '''
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        execute:            Execute button
        submit:             Submit button
        children:           Model parameters
        num_cpus:           Number of CPUs assigned to job
        num_gpus:           Number of GPUs assigned to job
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
    Returns:
        None
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'execute.n_clicks' in changed_id:
        return True
    if 'submit.n_clicks' in changed_id:
        contents = []
        experiment_id = str(uuid.uuid4())
        out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        out_path.mkdir(parents=True, exist_ok=True)
        input_params = {}
        if bool(children):
            for child in children['props']['children']:
                key = child["props"]["children"][1]["props"]["id"]["param_key"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        json_dict = input_params
        kwargs = {}
        if action_selection == 'train_model':
            if type(DATA) != dict:
                data_path = DATA_PATH
            else:
                data_path = DATA_PATH + '/train'
            command = "python3 src/train_model.py"
            directories = [data_path, str(out_path)]
        else:
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
            kwargs = {'train_params': job_data[row[0]]['parameters']}
            train_params = str_to_dict(job_data[row[0]]['parameters'])
            json_dict['target_width'] = train_params['target_width']
            json_dict['target_height'] = train_params['target_height']
        if action_selection == 'prediction_model':
            if type(DATA) != dict:
                data_path = DATA_PATH
            else:
                data_path = DATA_PATH + '/test'
            command = "python3 src/predict_model.py"
            directories = [data_path, str(in_path) , str(out_path)]
        job = SimpleJob(service_type='backend',
                        working_directory='{}'.format(DATA_DIR),
                        uri='mlexchange/unsupervised-classifier',
                        cmd= ' '.join([command, str(json_dict)] + directories),
                        kwargs = {'job_type': action_selection,
                                  'experiment_id': experiment_id,
                                  'params': json_dict,
                                  **kwargs})
        job.submit(USER, num_cpus, num_gpus)
        return False
    return False


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8052)
