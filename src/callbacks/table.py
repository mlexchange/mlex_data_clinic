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
    Input('interval', 'n_intervals'),
    Input('jobs-table', 'selected_rows'),
    State('jobs-table', 'data'),
    State('loss-plot', 'figure'),
)
def update_table(n, row, current_job_table, current_fig):
    '''
    This callback updates the job table and loss plot according to the job status in the 
    compute service.
    Args:
        n:                  Time intervals that triggers this callback
        row:                Selected row (job)
        current_job_table:  Current job table
        current_fig:        Current loss plot
    Returns:
        jobs-table:         Updates the job table
        loss-plot:          Updates the loss plot according to the job status (logs)
        show-plot:          Shows/hides the loss plot
    '''
    job_list = TableJob.get_job(USER, 'data_clinic')
    data_table = []
    if job_list is not None:
        for job in job_list:
            simple_job = TableJob.compute_job_to_table_job(job)
            data_table.insert(0, simple_job.__dict__)
    fig = go.Figure(go.Scatter(x=[], y=[]))
    show_plot = False
    if row and row[0] < len(data_table):
        log = data_table[row[0]]["job_logs"]
        if log and data_table[row[0]]['job_type'] in ['train_model', 'tune_model']:
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
    return data_table, fig, show_plot


@callback(
    Output('info-modal', 'is_open'),
    Output('info-display', 'children'),
    Input('show-info', 'n_clicks'),
    Input('modal-close', 'n_clicks'),
    State('jobs-table', 'data'),
    State('info-modal', 'is_open'),
    State('jobs-table', 'selected_rows'),
)
def open_job_modal(n_clicks, close_clicks, current_job_table, is_open, rows):
    '''
    This callback updates shows the job logs and parameters
    Args:
        n_clicks:           Number of clicks in "show details" button
        close_clicks:       Close modal with close-up details of selected cell
        current_job_table:  Current job table
        is_open:            Open/close modal state
        rows:               Selected rows in jobs table
    Returns:
        info-modal:         Open/closes the modal
        info-display:       Display the job logs and parameters
    '''
    if not is_open and rows is not None and len(rows) > 0:
        job_data = current_job_table[rows[0]]
        logs = job_data['job_logs']
        params = job_data['parameters']
        info_display = dcc.Textarea(
            value= f'Parameters: {params}\n\nLogs: {logs}',
            style={
                'width': '100%', 
                'height': '30rem', 
                'font-family':'monospace'
                }
            )
        return True, info_display
    else:
        return False, dash.no_update


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