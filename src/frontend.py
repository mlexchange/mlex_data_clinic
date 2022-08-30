import json
import os
import logging
import pathlib
import shutil
import zipfile
import uuid, requests

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_uploader as du
import dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import PIL.Image as Image
import plotly.graph_objects as go

from file_manager import filename_list, move_a_file, move_dir, add_paths_from_dir, \
                         check_duplicate_filename, docker_to_local_path, local_to_docker_path, file_explorer
from helper_utils import SimpleJob
from helper_utils import plot_figure, get_bottleneck, get_job, generate_loss_plot, load_from_dir, str_to_dict, \
                         model_list_GET_call, get_gui_components, get_counter
from assets.kwarg_editor import JSONParameterEditor
import templates


### GLOBAL VARIABLES AND DATA LOADING
DATA_DIR = str(os.environ['DATA_DIR'])
MODEL_DATABASE = {"The Model": "path-to-model"} # hardcoded model database as dict
USER = 'admin'
MODELS = model_list_GET_call()
DOCKER_DATA = pathlib.Path.home() / 'data'
LOCAL_DATA = str(os.environ['DATA_DIR'])
DOCKER_HOME = str(DOCKER_DATA) + '/'
LOCAL_HOME = str(LOCAL_DATA)
UPLOAD_FOLDER_ROOT = DOCKER_DATA / 'upload'
SUPPORTED_FORMATS = ['tiff', 'tif', 'jpg', 'jpeg', 'png']

#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets) #, suppress_callback_exceptions=True)
# server = app.server
app.title = "Data Clinic"
app._favicon = 'mlex.ico'
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

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
                ]),
                dbc.FormGroup([
                    dbc.Label('Data'),
                    file_explorer,
                ]),
                dbc.Button('Execute',
                           id='execute',
                           n_clicks=0,
                           className='m-1',
                           style={'width': '100%', 'justify-content': 'center'})
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
                    "OK", id="ok-button", color='danger', outline=False,
                    className="ms-auto", n_clicks=0
                ),
            ]),
        ],
        id="warning-modal",
        is_open=False,
    ),
    dcc.Store(id='warning-cause', data=''),
    dcc.Store(id='counters', data=get_counter(USER))
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
                    dbc.Row(
                        [
                            dbc.Button("Deselect Row", id="deselect-row", style={'margin-left': '1rem'}),
                            dbc.Button("Delete Job", id="delete-row", color='danger'),
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
        dbc.Modal(
            [
                dbc.ModalHeader("Warning"),
                dbc.ModalBody('Models cannot be recovered after deletion.  \
                                Do you still want to proceed?"'),
                dbc.ModalFooter([
                    dbc.Button(
                        "OK", id="confirm-delete-row", color='danger', outline=False,
                        className="ms-auto", n_clicks=0
                    ),
                ]),
            ],
            id="delete-modal",
            is_open=False,
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
                    dbc.Label('Image: ', id='current-image-label'),
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
    dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0),
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
                                      value=0)]),
                        dbc.FormGroup([
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
                     dbc.Col(column_02, width=8),
                     html.Div(id='dummy-output')]
                ),
                RESOURCES_SETUP
            ],
            fluid=True
        )
    ]
)


##### FILE MANAGER CALLBACKS ####
@app.callback(
    Output("collapse", "is_open"),
    Input("collapse-button", "n_clicks"),
    Input("import-dir", "n_clicks"),
    State("collapse", "is_open")
)
def toggle_collapse(collapse_button, import_button, is_open):
    '''
    This callback toggles the file manager
    Args:
        collapse_button:    "Open File Manager" button
        import_button:      Import button
        is_open:            Open/close File Manager modal state
    '''
    if collapse_button or import_button:
        return not is_open
    return is_open


@app.callback(
    Output("warning-modal", "is_open"),
    Output("warning-msg", "children"),
    Input("warning-cause", "data"),
    Input("ok-button", "n_clicks"),
    State("warning-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_warning_modal(warning_cause, ok_n_clicks, is_open):
    '''
    This callback toggles a warning/error message
    Args:
        warning_cause:      Cause that triggered the warning
        ok_n_clicks:        Close the warning
        is_open:            Close/open state of the warning
    '''
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'ok-button.n_clicks' in changed_id:
        return not is_open, ""
    if warning_cause == 'wrong_dataset':
        return not is_open, "The dataset you have selected is not supported. Please select (1) a data directory " \
                        "where each subfolder corresponds to a given category, OR (2) an NPZ file."
    if warning_cause == 'different_size':
        return not is_open, "The number of images and labels do not match. Please select a different dataset."
    if warning_cause == 'no_row_selected':
        return not is_open, "Please select a trained model from the List of Jobs"
    if warning_cause == 'no_dataset':
        return not is_open, "Please upload the dataset before submitting the job."
    else:
        return False, ""


@app.callback(
    Output("modal", "is_open"),
    Input("delete-files", "n_clicks"),
    Input("confirm-delete", "n_clicks"),
    State("modal", "is_open")
)
def toggle_modal(n1, n2, is_open):
    '''
    This callback toggles a confirmation message for file manager
    Args:
        n1:         Delete files button
        n2:         Confirm delete button
        is_open:    Open/close confirmation modal state
    '''
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("npz-modal", "is_open"),
    Output("npz-img-key", "options"),

    Input("import-dir", "n_clicks"),
    Input("confirm-import", "n_clicks"),
    Input("npz-img-key", "value"),
    State("npz-modal", "is_open"),
    State("docker-file-paths", "data"),
)
def toggle_modal_keyword(import_button, confirm_import, img_key, is_open, npz_path):
    '''
    This callback opens the modal to select the keywords within the NPZ file. When a keyword is selected for images or
    labels, this option is removed from the options of the other.
    Args:
        import_button:      Import button
        confirm_import:     Confirm import button
        img_key:            Selected keyword for the images
        is_open:            Open/close status of the modal
        npz_path:           Path to NPZ file
    Returns:
        toggle_modal:       Open/close modal
        img_options:        Keyword options for images
    '''
    img_options = []
    toggle_modal = is_open
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if npz_path:
        if npz_path[0].split('.')[-1] == 'npz':
            data = np.load(npz_path[0])
            img_key_list = list(data.keys())
            df_img = pd.DataFrame({'c': img_key_list})
            img_options = [{'label':i, 'value':i} for i in df_img['c']]
            toggle_modal = True
    if is_open and 'confirm-import.n_clicks' in changed_id:
        toggle_modal = False
    return toggle_modal, img_options


@app.callback(
    Output('dummy-data', 'data'),
    Input('dash-uploader', 'isCompleted'),
    State('dash-uploader', 'fileNames')
)
def upload_zip(iscompleted, upload_filename):
    '''
    This callback uploads a ZIP file
    Args:
        iscompleted:        The upload operation is completed (bool)
        upload_filename:    Filename of the uploaded content
    '''
    if not iscompleted:
        return 0
    if upload_filename is not None:
        path_to_zip_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0]
        if upload_filename[0].split('.')[-1] == 'zip':  # unzip files and delete zip file
            zip_ref = zipfile.ZipFile(path_to_zip_file)  # create zipfile object
            path_to_folder = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0].split('.')[-2]
            if (upload_filename[0].split('.')[-2] + '/') in zip_ref.namelist():
                zip_ref.extractall(pathlib.Path(UPLOAD_FOLDER_ROOT))  # extract file to dir
            else:
                zip_ref.extractall(path_to_folder)
            zip_ref.close()  # close file
            os.remove(path_to_zip_file)
    return 0


@app.callback(
    Output('files-table', 'data'),
    Output('docker-file-paths', 'data'),
    Output('data-path', 'data'),

    Input('browse-format', 'value'),
    Input('browse-dir', 'n_clicks'),
    Input('import-dir', 'n_clicks'),
    Input('confirm-delete', 'n_clicks'),
    Input('move-dir', 'n_clicks'),
    Input('files-table', 'selected_rows'),
    Input('data-path', 'data'),
    Input('import-format', 'value'),
    Input('my-toggle-switch', 'value'),
    Input('jobs-table', 'selected_rows'),
    Input("clear-data", "n_clicks"),
    Input("refresh-data", "n_clicks"),

    State('dest-dir-name', 'value'),
    State('jobs-table', 'data')
)
def file_manager(browse_format, browse_n_clicks, import_n_clicks, delete_n_clicks, move_dir_n_clicks, rows,
                 selected_paths, import_format, docker_path, job_rows, clear_data, refresh_data, dest, job_data):
    '''
    This callback displays manages the actions of file manager
    Args:
        browse_format:      File extension to browse
        browse_n_clicks:    Browse button
        import_n_clicks:    Import button
        delete_n_clicks:    Delete button
        move_dir_n_clicks:  Move button
        rows:               Selected rows
        selected_paths:     Selected paths in cache
        import_format:      File extension to import
        docker_path:        [bool] docker vs local path
        job_rows:           Selected rows in job table. If it's not a "training" model, it will load its results
                            instead of the data uploaded through File Manager. This is so that the user can observe
                            previous evaluation results
        dest:               Destination path
        job_data:           Data in job table
        clear_data:         Clear the loaded images
        refresh_data:      Refresh the loaded images
    Returns
        files:              Filenames to be displayed in File Manager according to browse_format from docker/local path
        list_filename:      List of selected filenames in the directory AND SUBDIRECTORIES FROM DOCKER PATH
        selected_files:     List of selected filename FROM DOCKER PATH (no subdirectories)
        selected_row:       Selected row in jobs table
    '''
    changed_id = dash.callback_context.triggered[0]['prop_id']

    supported_formats = []
    import_format = import_format.split(',')
    if import_format[0] == '*':
        supported_formats = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
    else:
        for ext in import_format:
            supported_formats.append(ext.split('.')[1])

    files = []
    if browse_n_clicks or import_n_clicks:
        files = filename_list(DOCKER_DATA, browse_format)

    selected_files = []
    list_filename = []
    if bool(rows):
        for row in rows:
            file_path = files[row]
            selected_files.append(file_path)
            if file_path['file_type'] == 'dir':
                list_filename = add_paths_from_dir(file_path['file_path'], supported_formats, list_filename)
            else:
                list_filename.append(file_path['file_path'])

    if browse_n_clicks and changed_id == 'confirm-delete.n_clicks':
        for filepath in selected_files:
            if os.path.isdir(filepath['file_path']):
                shutil.rmtree(filepath['file_path'])
            else:
                os.remove(filepath['file_path'])
        selected_files = []
        files = filename_list(DOCKER_DATA, browse_format)

    if browse_n_clicks and changed_id == 'move-dir.n_clicks':
        if dest is None:
            dest = ''
        destination = DOCKER_DATA / dest
        destination.mkdir(parents=True, exist_ok=True)
        if bool(rows):
            sources = selected_paths
            for source in sources:
                if os.path.isdir(source['file_path']):
                    move_dir(source['file_path'], str(destination))
                    shutil.rmtree(source['file_path'])
                else:
                    move_a_file(source['file_path'], str(destination))
            selected_files = []
            files = filename_list(DOCKER_DATA, browse_format)
    if not docker_path:
        files = docker_to_local_path(files, DOCKER_HOME, LOCAL_HOME)
    
    if changed_id == 'refresh-data.n_clicks':
        list_filename, selected_files = [], []
        datapath = requests.get(f'http://labelmaker-api:8005/api/v0/import/datapath').json()
        if bool(datapath['datapath']) and os.path.isdir(datapath['datapath'][0]['file_path']):
            list_filename, selected_files = datapath['filenames'], datapath['datapath'][0]['file_path']
        return files,  list_filename, selected_files
        
    elif changed_id == 'import-dir.n_clicks':
        return files, list_filename, selected_files
        
    elif changed_id == 'clear-data.n_clicks':
        return [], [], []
        
    else:
        return files, dash.no_update, dash.no_update


##### DATA CLINIC CALLBACKS  ####
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
        action_selection:   Selected action (pre-defined actions in Data Clinic)
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
    Output('orig_img', 'src'),
    Output('rec_img', 'src'),
    Output('ls_graph', 'src'),
    Output('img-slider', 'max'),
    Output('img-slider', 'value'),
    Output('data-size-out', 'children'),
    Output('current-image-label', 'children'),

    Input('import-dir', 'n_clicks'),
    Input('confirm-import', 'n_clicks'),
    Input({'type': ALL, 'param_key': 'latent_dim', 'name': 'latent_dim', 'layer': 'input'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_width', 'name': 'target_width'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_height', 'name': 'target_height'}, 'value'),
    Input('img-slider', 'value'),
    Input('action', 'value'),
    Input('jobs-table', 'selected_rows'),
    Input('jobs-table', 'data'),
    Input("docker-file-paths", "data"),
    
    State("npz-img-key", "value"),
    State("npz-modal", "is_open"),
)
def refresh_image(import_dir, confirm_import, ls_var, target_width, target_height, img_ind, action_selection, row,
                  data_table, filenames, img_keyword, npz_modal):
    '''
    This callback updates the images in the display
    Args:
        import_dir:         Import button
        confirm_import:     Confirm import button
        ls_var:             Latent space value
        target_width:       Target data width (if resizing)
        target_height:      Target data height (if resizing)
        img_ind:            Index of image according to the slider value
        row:                Selected job (model) 
        data_table:         Data in table of jobs
        filenames:          Selected data files
        img_keyword:        Keyword for images in NPZ file
        npz_modal:          Open/close status of NPZ modal
        action_selection:   Action selection (train vs test)
    Returns:
        img-output:         Output figure
        img-reconst-output: Reconstructed output (if prediction is selected, ow. blank image)
        latent-space-plot:  Graphical representation of latent space definition
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
        img-slider-value:   Value of the slider according to the dataset length
        data-size-out:      Size of uploaded data
    '''
    current_im_label = ''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if action_selection in ['train_model', 'transfer_learning']:
        if len(ls_var) > 0:
            ls_var = int(ls_var[0])
            target_width = int(target_width[0])
            target_height = int(target_height[0])
            ls_plot = get_bottleneck(ls_var, target_width, target_height)
        else:
            ls_plot = dash.no_update
    else:
        ls_plot = get_bottleneck(1, 1, 1, False)
        target_width = None
    if row:
        if row[0] < len(data_table):
            if data_table[row[0]]['job_type'].split()[0] == 'train_model':
                if action_selection == 'prediction_model':
                    train_params = str_to_dict(data_table[row[0]]['parameters'])
                    ls_var = int(train_params['latent_dim'])
                    target_width = int(train_params['target_width'])
                    target_height = int(train_params['target_height'])
                    ls_plot = get_bottleneck(ls_var, target_width, target_height)
            else:
                supported_formats = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
                filenames = add_paths_from_dir(data_table[row[0]]['dataset'], supported_formats, [])
                job_id = data_table[row[0]]['experiment_id']
                reconstructed_path = 'data/mlexchange_store/{}/{}/reconstructed_images.npy'.format(USER, job_id)
                try:
                    reconstructed_data = np.load(reconstructed_path)
                    slider_max = reconstructed_data.shape[0]
                    img_ind = min(slider_max, img_ind)
                    reconst_img = Image.fromarray((np.squeeze((reconstructed_data[img_ind] -
                                                               np.min(reconstructed_data[img_ind])) * 255)).astype(np.uint8))
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

    if len(filenames) > 0:
        try:
            if filenames[0].split('.')[-1] == 'npz':    # npz file
                if img_keyword is not None:
                    current_im_label = filenames[0]
                    data_npz = np.load(filenames[0])
                    data_npy = np.squeeze(data_npz[img_keyword])
                    slider_max = len(data_npy) - 1
                    img_ind = min(slider_max, img_ind)
                    origimg = data_npy[img_ind]
            else:                                       # directory
                slider_max = len(filenames) - 1
                if img_ind > slider_max:
                    img_ind = 0
                origimg = Image.open(filenames[img_ind])
                current_im_label = filenames[img_ind]
        except Exception as e:
            print(f'Exception in refresh_image callback {e}')
    if 'origimg' not in locals():
        origimg = Image.fromarray((np.zeros((32,32)).astype(np.uint8)))
        slider_max = 0
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
    
    return origimg, recimg, ls_plot, slider_max, img_ind, data_size, 'Image: '+current_im_label


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

    State('jobs-table', 'data'),
    State('loss-plot', 'figure'),
    prevent_initial_call=True
)
def update_table(n, row, active_cell, close_clicks, current_job_table, current_fig):
    '''
    This callback updates the job table, loss plot, and results according to the job status in the compute service.
    Args:
        n:                  Time intervals that triggers this callback
        row:                Selected row (job)
        active_cell:        Selected cell in table of jobs
        close_clicks:       Close pop-up window
        current_job_table:  Current job table
        current_fig:        Current loss plot
    Returns:
        jobs-table:         Updates the job table
        loss-plot:          Updates the loss plot according to the job status (logs)
        show-plot:          Shows/hides the loss plot
        log-modal:          Open/close pop-up window
        log-display:        Contents of pop-up window
        jobs-table:         Selects/deselects the active cell in job table. Without this output, the pop-up window will not
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
            if job['job_kwargs']['kwargs']['job_type'].split()[0] != 'train_model':
                params = params + '\nTraining Parameters: ' + str(job['job_kwargs']['kwargs']['train_params'])
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  name=job['description'],
                                  job_type=job['job_kwargs']['kwargs']['job_type'],
                                  status=job['status']['state'],
                                  parameters=params,
                                  experiment_id=job['job_kwargs']['kwargs']['experiment_id'],
                                  dataset=job['job_kwargs']['kwargs']['dataset'],
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
        if row[0] < len(data_table):
            log = data_table[row[0]]["job_logs"]
            if log:
                if data_table[row[0]]['job_type'].split()[0] == 'train_model':
                    start = log.find('epoch')
                    if start > -1 and len(log) > start + 5:
                        fig = generate_loss_plot(log, start)
                        show_plot = True
    if current_fig:
        if current_fig['data'][0]['x']== list(fig['data'][0]['x']):
            fig = dash.no_update
    if data_table == current_job_table:
        data_table = dash.no_update
    return data_table, fig, show_plot, is_open, log_display, None


@app.callback(
    Output('jobs-table', 'selected_rows'),
    Input('deselect-row', 'n_clicks'),
    prevent_initial_call=True
)
def deselect_row(n_click):
    '''
    This callback deselects the row in the data table
    '''
    return []


@app.callback(
    Output('delete-modal', 'is_open'),
    Input('confirm-delete-row', 'n_clicks'),
    Input('delete-row', 'n_clicks'),
    State('jobs-table', 'selected_rows'),
    State('jobs-table', 'data'),
    prevent_initial_call=True
)
def delete_row(confirm_delete, delete, row, job_data):
    '''
    This callback deletes the selected model in the table
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'delete-row.n_clicks' == changed_id:
        return True
    else:
        job_uid = job_data[row[0]]['job_id']
        requests.delete(f'http://job-service:8080/api/v0/jobs/{job_uid}/delete')
        return False


@app.callback(
    Output('resources-setup', 'is_open'),
    Output('counters', 'data'),
    Output("warning-cause", "data"),

    Input('execute', 'n_clicks'),
    Input('submit', 'n_clicks'),

    State('app-parameters', 'children'),
    State('num-cpus', 'value'),
    State('num-gpus', 'value'),
    State('action', 'value'),
    State('jobs-table', 'data'),
    State('jobs-table', 'selected_rows'),
    State('data-path', 'data'),
    State("docker-file-paths", "data"),
    State("counters", "data"),
    State("npz-img-key", "value"),
    State("model-name", "value"),
    prevent_intial_call=True)
def execute(execute, submit, children, num_cpus, num_gpus, action_selection, job_data, row, data_path, filenames,
            counters, x_key, model_name):
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
        data_path:          Local path to data
        counters:           List of counters to assign a number to each job according to its action (train vs evaluate)
        filenames:          List of filenames within this dataset
        x_key:              Keyword for x data in NPZ file
    Returns:
        open/close the resources setup modal
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'execute.n_clicks' in changed_id:
        if len(filenames) == 0:
            return False, counters, 'no_dataset'
        if action_selection != 'train_model' and not row:
            return False, counters, 'no_row_selected'
        if row:
            if action_selection != 'train_model' and job_data[row[0]]['job_type'].split()[0] != 'train_model':
                return False, counters, 'no_row_selected'
        return True, counters, ''
    if 'submit.n_clicks' in changed_id:
        counters = get_counter(USER)
        experiment_id = str(uuid.uuid4())
        out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        out_path.mkdir(parents=True, exist_ok=True)
        input_params = {'data_key': x_key}
        if bool(children):
            for child in children['props']['children']:
                key = child["props"]["children"][1]["props"]["id"]["param_key"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        try:
            data_path = data_path[0]['file_path']
        except Exception as e:
            print(e)
        json_dict = input_params
        kwargs = {}
        if action_selection == 'train_model':
            counters[0] = counters[0] + 1
            count = counters[0]
            command = "python3 src/train_model.py"
            directories = [data_path, str(out_path)]
        else:
            counters[1] = counters[1] + 1
            count = counters[1]
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
            kwargs = {'train_params': job_data[row[0]]['parameters']}
            train_params = str_to_dict(job_data[row[0]]['parameters'])
            json_dict['target_width'] = train_params['target_width']
            json_dict['target_height'] = train_params['target_height']
        if action_selection == 'prediction_model':
            command = "python3 src/predict_model.py"
            directories = [data_path, str(in_path) , str(out_path)]
        if len(model_name)==0:      # if model_name was not defined
            model_name = f'{action_selection} {count}'
        job = SimpleJob(service_type='backend',
                        description=model_name,
                        working_directory='{}'.format(DATA_DIR),
                        uri='mlexchange1/unsupervised-classifier',
                        cmd= ' '.join([command] + directories + ['\''+json.dumps(json_dict)+'\'']),
                        kwargs = {'job_type': action_selection,
                                  'experiment_id': experiment_id,
                                  'dataset': data_path,
                                  'params': json_dict,
                                  **kwargs})
        job.submit(USER, num_cpus, num_gpus)
        return False, counters, ''
    return False, counters, ''


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8072)
