from dash import dash_table, dcc
import dash_bootstrap_components as dbc


def job_table():
    job_table = dbc.Card(
        children=[
                dbc.CardHeader("List of Jobs"),
                dbc.CardBody(
                    children=[
                        dbc.Alert(
                            "Your ML job is being prepared, it will be shown in the table shortly.",
                            id="job-alert",
                            dismissable=True,
                            is_open=False,
                        ),
                        dbc.Alert(
                            "Your ML job has been succesfully submitted.",
                            id="job-alert-confirm",
                            dismissable=True,
                            is_open=False,
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Deselect Row", 
                                        id="deselect-row",
                                        style={
                                            'width': '100%', 
                                            'margin-bottom': '1rem'
                                            }
                                        )
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Show Details", 
                                        id="show-info",
                                        style={'width': '100%', 'margin-bottom': '1rem'}
                                        )
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            'Download Results',
                                            id='download-button',
                                            style={'width': '100%', 'margin-bottom': '1rem'},
                                            disabled=True,
                                        ),
                                        dcc.Download(id='download-out'),
                                        dbc.Modal(
                                            [
                                                dbc.ModalBody('Download will start shortly'),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        'OK',
                                                        id='close-storage-modal'
                                                        )
                                                    )
                                            ],
                                            id="storage-modal",
                                            is_open=False
                                        )
                                    ]
                                ),
                                dbc.Col(
                                    dbc.Button("Stop Job", 
                                               id="stop-row", 
                                               color='warning',
                                               style={'width': '100%'})
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Delete Job", 
                                        id="delete-row", 
                                        color='danger',
                                        style={'width': '100%'}
                                        )
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
                            hidden_columns=['job_id',
                                            'experiment_id',
                                            'dataset',
                                            'parameters',
                                            'job_logs'],
                            row_selectable='single',
                            style_cell={'padding': '1rem',
                                        'textAlign': 'left',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'maxWidth': 0},
                            fixed_rows={'headers': True},
                            css=[{"selector": ".show-hide",
                                  "rule": "display: none"}],
                            page_size=4,
                            style_data_conditional=[
                                {'if': {
                                    'column_id': 'status',
                                    'filter_query': '{status} = complete'
                                    },
                                'backgroundColor': 'green',
                                'color': 'white'},
                                {'if': {
                                    'column_id': 'status',
                                    'filter_query': '{status} = failed'
                                    },
                                'backgroundColor': 'red',
                                'color': 'white'},
                            ],
                            style_table={'overflowY': 'auto'}
                        )
                    ],
                ),
            dbc.Modal(
                [
                    dbc.ModalHeader("Warning"),
                    dbc.ModalBody('Models cannot be recovered after deletion. Do you still want to \
                                  proceed?"'),
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
                dbc.ModalHeader("Job Information"),
                dbc.ModalBody(id='info-display'),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", 
                        id="modal-close", 
                        className="ml-auto"
                        )
                    ),
                ],
                id='info-modal',
                size='xl')
        ]
    )
    return job_table