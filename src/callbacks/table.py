from dash import dcc, Input, Output, State, callback
import dash
import plotly.graph_objects as go

from app_layout import USER
from utils.job_utils import TableJob
from utils.plot_utils import generate_loss_plot


@callback(
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
)
def update_table(n, row, active_cell, close_clicks, current_job_table, current_fig):
    '''
    This callback updates the job table, loss plot, and results according to the job status in the 
    compute service.
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
        jobs-table:         Selects/deselects the active cell in job table. Without this output, 
                            the pop-up window will not close
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'modal-close.n_clicks' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update, False, dash.no_update, None
    job_list = TableJob.get_job(USER, 'data_clinic')
    data_table = []
    if job_list is not None:
        for job in job_list:
            simple_job = TableJob.compute_job_to_table_job(job)
            data_table.insert(0, simple_job.__dict__)
    is_open = dash.no_update
    log_display = dash.no_update
    if active_cell:
        row_log = active_cell["row"]
        col_log = active_cell["column_id"]
        if col_log == 'job_logs':       # show job logs
            is_open = True
            log_display = dcc.Textarea(value=data_table[row_log]["job_logs"],
                                       style={'width': '100%', 
                                              'height': '30rem', 
                                              'font-family':'monospace'})
        if col_log == 'parameters':     # show job parameters
            is_open = True
            log_display = dcc.Textarea(value=data_table[row_log]["parameters"],
                                       style={'width': '100%', 
                                              'height': '30rem', 
                                              'font-family': 'monospace'})
    fig = go.Figure(go.Scatter(x=[], y=[]))
    show_plot = False
    if row:
        if row[0] < len(data_table):
            log = data_table[row[0]]["job_logs"]
            if log:
                if data_table[row[0]]['job_type'] in ['train_model', 'tune_model']:
                    start = log.find('epoch')
                    if start > -1 and len(log) > start + 25:
                        try:
                            fig = generate_loss_plot(log, start)
                            show_plot = True
                        except Exception as e:
                            print(f'Loss plot exception {e}')
    if current_fig:
        try:
            if current_fig['data'][0]['x'] == list(fig['data'][0]['x']):
                fig = dash.no_update
        except Exception as e:
            print(e)
    if data_table == current_job_table:
        data_table = dash.no_update
    return data_table, fig, show_plot, is_open, log_display, None


@callback(
    Output('jobs-table', 'selected_rows'),
    Input('deselect-row', 'n_clicks'),
    prevent_initial_call=True
)
def deselect_row(n_click):
    '''
    This callback deselects the row in the data table
    '''
    return []


@callback(
    Output('delete-modal', 'is_open'),

    Input('confirm-delete-row', 'n_clicks'),
    Input('delete-row', 'n_clicks'),
    Input('stop-row', 'n_clicks'),

    State('jobs-table', 'selected_rows'),
    State('jobs-table', 'data'),
    prevent_initial_call=True
)
def delete_row(confirm_delete, delete, stop, row, job_data):
    '''
    This callback deletes the selected model in the table
    Args:
        confirm_delete:     Number of clicks in confirm delete job button
        delete:             Number of clicks in delete job button
        stop:               Number of clicks in stop job button
        row:                Row (job) selected from table
        job_data:           Job table data
    Returns:
        Open/closes warning modal
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'delete-row.n_clicks' == changed_id:
        return True
    elif 'stop-row.n_clicks' == changed_id:
        job_uid = job_data[row[0]]['job_id']
        TableJob.terminate_job(job_uid)
        return False
    else:
        job_uid = job_data[row[0]]['job_id']
        TableJob.delete_job(job_uid)
        return False