import copy
import os
import pathlib
import requests

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_table
import dash_uploader as du

DOCKER_DATA = pathlib.Path.home() / 'data'
LOCAL_DATA = str(os.environ['DATA_DIR'])
DOCKER_HOME = str(DOCKER_DATA) + '/'
LOCAL_HOME = str(LOCAL_DATA)

UPLOAD_FOLDER_ROOT = DOCKER_DATA / 'upload'
DATAPATH_DEFAULT, FILENAMES_DEFAULT = [], []
try:
    DATAPATH = requests.get(f'http://labelmaker-api:8005/api/v0/datapath/import_dataset').json()
except Exception as e:
    DATAPATH = False
    print(e)
if DATAPATH:
    if bool(DATAPATH['datapath']) and os.path.isdir(DATAPATH['datapath']['file_path'][0]):
        DATAPATH_DEFAULT = DATAPATH['datapath']['file_path'][0]
        FILENAMES_DEFAULT = DATAPATH['filenames']

# FILES DISPLAY
file_paths_table = html.Div(
        children=[
            dash_table.DataTable(
                id='files-table',
                columns=[
                    {'name': 'type', 'id': 'file_type'},
                    {'name': 'File Table', 'id': 'file_path'},
                ],
                data = [],
                hidden_columns = ['file_type'],
                row_selectable='single', #'multi',
                style_cell={'padding': '0.5rem', 'textAlign': 'left'},
                fixed_rows={'headers': False},
                css=[{"selector": ".show-hide", "rule": "display: none"}],
                style_data_conditional=[
                    {'if': {'filter_query': '{file_type} = dir'},
                     'color': 'blue'},
                 ],
                style_table={'height':'18rem', 'overflowY': 'auto'}
            )
        ]
    )


# UPLOAD DATASET OR USE PRE-DEFINED DIRECTORY
data_access = html.Div([
    dbc.Card([
        dbc.CardBody(id='data-body',
                      children=[
                          dbc.Label('Upload a new file or folder (zip) to work dir:', className='mr-2'),
                          html.Div([html.Div([du.Upload(
                                                    id="dash-uploader",
                                                    max_file_size=1800,  # 1800 Mb
                                                    cancel_button=True,
                                                    pause_button=True)],
                                                style={  # wrapper div style
                                                    'textAlign': 'center',
                                                    'width': '300px',
                                                    'padding': '5px',
                                                    'display': 'inline-block',
                                                    'margin-bottom': '30px',
                                                    'margin-right': '20px'}),
                                    html.Div([
                                        dbc.Col([
                                            dbc.Label("Dataset is by default uploaded to '{}'. \
                                                       You can move the selected files or dirs (from File Table) \
                                                       into a new dir.".format(UPLOAD_FOLDER_ROOT), className='mr-5'),
                                            dbc.Label("Home data dir (HOME) is '{}'.".format(DOCKER_DATA),
                                                      className='mr-5'),
                                            html.Div([
                                                dbc.Label('Move data into dir:'.format(DOCKER_DATA), className='mr-5'),
                                                dcc.Input(id='dest-dir-name', placeholder="Input relative path to HOME",
                                                                style={'width': '40%', 'margin-bottom': '10px'}),
                                                dbc.Button("Move",
                                                     id="move-dir",
                                                     className="ms-auto",
                                                     color="secondary",
                                                     outline=True,
                                                     n_clicks=0,
                                                     #disabled = True,
                                                     style={'width': '22%', 'margin': '5px'}),
                                            ],
                                            style = {'width': '100%', 'display': 'flex', 'align-items': 'center'},
                                            )
                                        ])
                                    ])
                                    ],
                            style = {'width': '100%', 'display': 'flex', 'align-items': 'center'}
                          ),
                          dbc.Label('Choose files/directories:', className='mr-2'),
                          html.Div(
                                  [dbc.Button("Browse",
                                             id="browse-dir",
                                             className="ms-auto",
                                             color="secondary",
                                             outline=True,
                                             n_clicks=0,
                                             style={'width': '15%', 'margin': '5px'}),
                                   html.Div([
                                        dcc.Dropdown(
                                                id='browse-format',
                                                options=[
                                                    {'label': 'dir', 'value': 'dir'},
                                                    {'label': 'all (*)', 'value': '*'},
                                                    {'label': '.png', 'value': '*.png'},
                                                    {'label': '.jpg/jpeg', 'value': '*.jpg,*.jpeg'},
                                                    {'label': '.tif/tiff', 'value': '*.tif,*.tiff'},
                                                    {'label': '.txt', 'value': '*.txt'},
                                                    {'label': '.csv', 'value': '*.csv'},
                                                    {'label': '.npz', 'value': '*.npz'},
                                                ],
                                                value='dir')
                                            ],
                                            style={"width": "15%", 'margin-right': '60px'}
                                    ),
                                  dbc.Button("Delete the Selected",
                                             id="delete-files",
                                             className="ms-auto",
                                             color="danger",
                                             outline=True,
                                             n_clicks=0,
                                             style={'width': '22%', 'margin-right': '10px'}
                                    ),
                                   dbc.Modal(
                                        [
                                            dbc.ModalHeader("Warning"),
                                            dbc.ModalBody("Files cannot be recovered after deletion.  \
                                                          Do you still want to proceed?"),
                                            dbc.ModalFooter([
                                                dbc.Button(
                                                    "Delete", id="confirm-delete", color='danger', outline=False,
                                                    className="ms-auto", n_clicks=0
                                                ),
                                            ]),
                                        ],
                                        id="modal",
                                        is_open=False,
                                        style = {'color': 'red'}
                                    ),
                                   dbc.Button("Import",
                                             id="import-dir",
                                             className="ms-auto",
                                             color="secondary",
                                             outline=True,
                                             n_clicks=0,
                                             style={'width': '22%', 'margin': '5px'}
                                   ),
                                   html.Div([
                                        dcc.Dropdown(
                                                id='import-format',
                                                options=[
                                                    {'label': 'all files (*)', 'value': '*'},
                                                    {'label': '.png', 'value': '*.png'},
                                                    {'label': '.jpg/jpeg', 'value': '*.jpg,*.jpeg'},
                                                    {'label': '.tif/tiff', 'value': '*.tif,*.tiff'},
                                                    {'label': '.txt', 'value': '*.txt'},
                                                    {'label': '.csv', 'value': '*.csv'},
                                                ],
                                                value='*')
                                            ],
                                            style={"width": "15%"}
                                    ),
                                 ],
                                style = {'width': '100%', 'display': 'flex', 'align-items': 'center'},
                                ),
                         html.Div([ html.Div([dbc.Label('Show Local/Docker Path')], style = {'margin-right': '10px'}),
                                    daq.ToggleSwitch(
                                        id='my-toggle-switch',
                                        value=False
                                    )],
                            style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'margin': '10px',
                                     'margin-left': '0px'},
                        ),
                        file_paths_table,
                        ]),
    ],
    id="data-access",
    )
])


file_explorer = html.Div(
    [
        dbc.Row([
            dbc.Col(dbc.Button(
                        "Refresh Images",
                        id="refresh-data",
                        size="lg",
                        className='m-1',
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={'width': '100%'}), width=6),
            dbc.Col(dbc.Button(
                        "Clear Images",
                        id="clear-data",
                        size="lg",
                        className='m-1',
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={'width': '100%'}), width=6) 
            ],
            justify = 'center'
        ),
        dbc.Button(
            "Open File Manager",
            id="collapse-button",
            size="lg",
            className='m-1',
            color="secondary",
            outline=True,
            n_clicks=0,
            style={'width': '100%', 'justify-content': 'center'}
        ),
        dbc.Modal(
            data_access,
            id="collapse",
            is_open=False,
            size='xl'
        ),
        dbc.Modal(
           [
               dbc.ModalHeader("You have selected a ZIP file"),
               dbc.ModalBody([
                   dbc.Label("Keyword for images:"),
                   dcc.Dropdown(id='npz-img-key'),
               ]),
               dbc.ModalFooter([
                    dbc.Button(
                        "Confirm Import", id="confirm-import", outline=False,
                        className="ms-auto", n_clicks=0
                    ),
                ]),
           ],
           id="npz-modal",
           is_open=False,
        ),
        dcc.Store(id='dummy-data', data=[]),
        dcc.Store(id='docker-file-paths', data=FILENAMES_DEFAULT),
        dcc.Store(id='data-path', data=DATAPATH_DEFAULT),
    ]
)


def move_a_file(source, destination):
    '''
    Args:
        source, str:          full path of a file from source directory
        destination, str:     full path of destination directory 
    '''
    pathlib.Path(destination).mkdir(parents=True, exist_ok=True)
    filename = source.split('/')[-1]
    new_destination = destination + '/' + filename
    os.rename(source, new_destination)


def move_dir(source, destination):
    '''
    Args:
        source, str:          full path of source directory
        destination, str:     full path of destination directory 
    '''
    dir_path, list_dirs, filenames = next(os.walk(source))
    original_dir_name = dir_path.split('/')[-1]
    destination = destination + '/' + original_dir_name
    pathlib.Path(destination).mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        file_source = dir_path + '/' + filename  
        move_a_file(file_source, destination)
    
    for dirname in list_dirs:
        dir_source = dir_path + '/' + dirname
        move_dir(dir_source, destination)


def add_paths_from_dir(dir_path, supported_formats, list_file_path):
    '''
    Args:
        dir_path, str:            full path of a directory
        supported_formats, list:  supported formats, e.g., ['tiff', 'tif', 'jpg', 'jpeg', 'png']
        list_file_path, [str]:     list of absolute file paths
    
    Returns:
        Adding unique file paths to list_file_path, [str]
    '''
    root_path, list_dirs, filenames = next(os.walk(dir_path))
    for filename in filenames:
        exts = filename.split('.')
        if exts[-1] in supported_formats and exts[0] != '':
            file_path = root_path + '/' + filename
            if file_path not in list_file_path:
                list_file_path.append(file_path)
            
    for dirname in list_dirs:
        new_dir_path = dir_path + '/' + dirname
        list_file_path = add_paths_from_dir(new_dir_path, supported_formats, list_file_path)
    
    return list_file_path


def filename_list(directory, format):
    '''
    Args:
        directory, str:     full path of a directory
        format, list(str):  list of supported formats
    Return:
        a full list of absolute file path (filtered by file formats) inside a directory. 
    '''
    hidden_formats = ['DS_Store']
    files = []
    if format == 'dir':
        if os.path.exists(directory):
            for filepath in pathlib.Path(directory).glob('**/*'):
                if os.path.isdir(filepath):
                    files.append({'file_path': str(filepath.absolute()), 'file_type': 'dir'})
    else:
        format = format.split(',')
        for f_ext in format:
            if os.path.exists(directory):
                for filepath in pathlib.Path(directory).glob('**/{}'.format(f_ext)):
                    if os.path.isdir(filepath):
                        files.append({'file_path': str(filepath.absolute()), 'file_type': 'dir'})
                    else:
                        filename = str(filepath).split('/')[-1]
                        exts = filename.split('.')
                        if exts[-1] not in hidden_formats and exts[0] != '':
                            files.append({'file_path': str(filepath.absolute()), 'file_type': 'file'})
    
    return files


def check_duplicate_filename(dir_path, filename):
    root_path, list_dirs, filenames = next(os.walk(dir_path))
    if filename in filenames:
        return True
    else:
        return False


def docker_to_local_path(paths, docker_home, local_home, type='list-dict'):
    '''
    Args:
        paths:              docker file paths
        docker_home, str:   full path of home dir (ends with '/') in docker environment
        local_home, str:    full path of home dir (ends with '/') mounted in local machine
        type:
            list-dict, default:  a list of dictionary (docker paths), e.g., [{'file_path': 'docker_path1'},{...}]
            str:                a single file path string
    Return: 
        replace docker path with local path.
    '''
    if type == 'list-dict':
        files = copy.deepcopy(paths)
        for file in files:
            if not file['file_path'].startswith(local_home):
                file['file_path'] = local_home + file['file_path'].split(docker_home)[-1]
    
    if type == 'str':
        if not paths.startswith(local_home):
            files = local_home + paths.split(docker_home)[-1]
        else:
            files = paths
        
    return files


def local_to_docker_path(paths, docker_home, local_home, type='list'):
    '''
    Args:
        paths:             selected local (full) paths 
        docker_home, str:  full path of home dir (ends with '/') in docker environment
        local_home, str:   full path of home dir (ends with '/') mounted in local machine
        type:
            list:          a list of path string
            str:           single path string 
    Return: 
        replace local path with docker path
    '''
    if type == 'list':
        files = []
        for i in range(len(paths)):
            if not paths[i].startswith(docker_home):
                files.append(docker_home + paths[i].split(local_home)[-1])
            else:
                files.append(paths[i])
    
    if type == 'str':
        if not paths.startswith(docker_home):
            files = docker_home + paths.split(local_home)[-1]
        else:
            files = paths

    return files

