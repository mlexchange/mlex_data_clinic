from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_uploader as du


def create_file_explorer(max_file_size):
    '''
    Creates the dash components for the file explorer
    Args:
        max_file_size:      Maximum file size to be uploaded
    Returns:
        file_explorer:      HTML.DIV with all the corresponding components of the file explorer
    '''
    file_explorer = html.Div([
        dbc.Card([
            dbc.CardBody(id={'base_id': 'file-manager', 'name': 'data-body'},
                        children=[
                            ######################## UPLOADING DATA ################################
                            dbc.Label('Upload a new file or a zipped folder:', className='mr-2'),
                            html.Div([
                                du.Upload(id={'base_id': 'file-manager', 'name': 'dash-uploader'},
                                          max_file_size=max_file_size,
                                          cancel_button=True,
                                          pause_button=True
                                        )],
                                style={'textAlign': 'center',
                                       'width': '100%',
                                       'padding': '5px',
                                       'display': 'inline-block',
                                       'margin-bottom': '10px',
                                       'margin-right': '20px'},
                            ),
                            ################## BROWSE/IMPORT DATA FORMATS ##########################
                            dbc.Label('Choose files/directories:', className='mr-2'),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Row([
                                        dbc.Col(dbc.InputGroupText('Browse Format: ',
                                                                    style={'height': '2.5rem', 
                                                                           'margin-bottom': '10px',
                                                                           'width': '100%'}),
                                                width=5),
                                        dbc.Col(dcc.Dropdown(
                                                    id={'base_id': 'file-manager', 
                                                        'name': 'browse-format'},
                                                    options=[
                                                        {'label': 'dir', 'value': '**/'},
                                                        {'label': 'all (*)', 'value': '*'},
                                                        {'label': '.png', 'value': '**/*.png'},
                                                        {'label': '.jpg/jpeg', 'value': '**/*.jpg'},
                                                        {'label': '.tif/tiff', 'value': '**/*.tif'},
                                                        {'label': '.txt', 'value': '**/*.txt'},
                                                        {'label': '.csv', 'value': '**/*.csv'},
                                                    ],
                                                    value='**/',
                                                    style={'height': '2.5rem', 'width': '100%'}),
                                                width=7)
                                    ], className='g-0'),
                                    width=5,
                                ),
                                dbc.Col(
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.InputGroupText('Import Format: ',
                                                               style={'height': '2.5rem', 
                                                                        'width': '100%'}),
                                                               width=5),
                                        dbc.Col(dcc.Dropdown(
                                                id={'base_id': 'file-manager', 
                                                    'name': 'import-format'},
                                                options=[
                                                        {'label': 'all (*)', 'value': '*'},
                                                        {'label': '.png', 'value': '**/*.png'},
                                                        {'label': '.jpg/jpeg', 'value': '**/*.jpg'},
                                                        {'label': '.tif/tiff', 'value': '**/*.tif'},
                                                        {'label': '.txt', 'value': '**/*.txt'},
                                                        {'label': '.csv', 'value': '**/*.csv'},
                                                    ],
                                                    value='*',
                                                style={'height': '2.5rem', 'width': '100%'}),
                                                width=7)
                                    ], className='g-0'),
                                    width=5
                                ),
                                dbc.Col(
                                    dbc.Button('Import',
                                               id={'base_id': 'file-manager', 'name': 'import-dir'},
                                               className='ms-auto',
                                               color='secondary',
                                               size='sm',
                                               outline=True,
                                               n_clicks=0,
                                               style={'width': '100%', 'height': '2.5rem'}
                                    ),
                                    width=2,
                                ),
                            ]),   
                            ###################### TILED FOR DATA ACCESS ###########################
                            dbc.Label('Load data through Tiled:', 
                                      style = {'margin-right': '10px',
                                               'margin-bottom': '10px'}),   
                            dbc.Row([
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText('URI'), 
                                        dbc.Input(placeholder='tiled uri', 
                                                  id={'base_id': 'file-manager', 
                                                      'name': 'tiled-uri'})
                                    ]),
                                    width=11
                                ),
                                dbc.Col(
                                    daq.BooleanSwitch(
                                        id={'base_id': 'file-manager', 'name': 'tiled-switch'},
                                        on=False,
                                        color='green'
                                    ),
                                    width=1
                                ),
                                dbc.Modal(
                                    children=[
                                        dbc.ModalHeader(dbc.ModalTitle("Warning")),
                                        dbc.ModalBody(children="Could not connect to Tiled"),
                                        # dbc.ModalFooter([dbc.Button("OK")])
                                        ], 
                                    id={'base_id': 'file-manager', 'name': 'tiled-error'},
                                    is_open=False,
                                    style = {'color': 'red'}
                                )
                            ]),
                            ############################ FILE TABLE ################################
                            html.Div(
                                children=[
                                    dash_table.DataTable(
                                        id={'base_id': 'file-manager', 'name': 'files-table'},
                                        columns=[
                                            {'name': 'Type', 'id': 'type'},
                                            {'name': 'URI', 'id': 'uri'}
                                        ],
                                        data = [],
                                        hidden_columns = ['type'],
                                        row_selectable='single',
                                        style_cell={'padding': '0.5rem', 'textAlign': 'left'},
                                        fixed_rows={'headers': False},
                                        css=[{'selector': '.show-hide', 'rule': 'display: none'}],
                                        style_data_conditional=[
                                            {'if': {'filter_query': '{file_type} = dir'},
                                                'color': 'blue'},
                                            ],
                                        style_table={'height':'18rem', 'overflowY': 'auto',
                                                     'margin-top': '10px'}
                                        ),
                                    dcc.Store(id={'base_id': 'file-manager', 
                                                  'name': 'confirm-update-data'}, 
                                              data=True),
                                    dcc.Store(id={'base_id': 'file-manager', 
                                                  'name': 'confirm-clear-data'}, 
                                              data=False),
                                    dcc.Store(id={'base_id': 'file-manager', 
                                                  'name': 'docker-file-paths'}, 
                                              data=[]),
                                    dcc.Store(id={'base_id': 'file-manager', 'name': 'upload-data'},
                                              data=False),
                                    dcc.Store(id={'base_id': 'file-manager', 'name': 'project-id'},
                                              data=''),
                                    ]
                                )
                            ]),
        ])
    ])
    return file_explorer