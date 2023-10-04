import json
import pathlib

from dash import Input, Output, State, callback
import dash

from file_manager.data_project import DataProject
from app_layout import USER, DATA_DIR
from utils.job_utils import str_to_dict, MlexJob, TableJob
from utils.data_utils import prepare_directories


@callback(
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
    State({'base_id': 'file-manager', 'name': 'docker-file-paths'},'data'),
    State("counters", "data"),
    State("model-name", "value"),
    State({'base_id': 'file-manager', 'name': 'project-id'}, 'data'),
    prevent_intial_call=True)
def execute(execute, submit, children, num_cpus, num_gpus, action_selection, job_data, row, file_paths,
            counters, model_name, project_id):
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
        counters:           List of counters to assign a number to each job according to its action 
                            (train vs evaluate)
        file_paths:         List of filenames within this dataset
    Returns:
        open/close the resources setup modal
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    if 'execute.n_clicks' in changed_id:
        if len(data_project.data) == 0:
            return False, counters, 'no_dataset'
        elif file_paths[0]['uid']=='1234':
            return False, counters, 'data_project_not_ready'
        if action_selection != 'train_model' and not row:
            return False, counters, 'no_row_selected'
        if row:
            if action_selection != 'train_model' and \
                job_data[row[0]]['job_type'].split()[0] != 'train_model':
                return False, counters, 'no_row_selected'
        return True, counters, ''
    if 'submit.n_clicks' in changed_id:
        counters = TableJob.get_counter(USER)
        experiment_id, out_path, data_info = prepare_directories(USER, data_project, project_id)
        input_params = {}
        if bool(children):
            for child in children['props']['children']:
                key = child["props"]["children"][1]["props"]["id"]["param_key"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        json_dict = input_params
        kwargs = {}
        if action_selection == 'train_model':
            counters[0] = counters[0] + 1
            count = counters[0]
            command = f"python3 src/train_model.py -d {data_info} -o {out_path}"
        else:
            counters[1] = counters[1] + 1
            count = counters[1]
            training_exp_id = job_data[row[0]]['experiment_id']
            model_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
            kwargs = {'train_params': job_data[row[0]]['parameters']}
            train_params = str_to_dict(job_data[row[0]]['parameters'])
            json_dict['target_width'] = train_params['target_width']
            json_dict['target_height'] = train_params['target_height']
            command = f"python3 src/predict_model.py -d {data_info} -m {model_path} -o {out_path}"
        if len(model_name)==0:      # if model_name was not defined
            model_name = f'{action_selection} {count}'
        job = MlexJob(service_type='backend',
                        description=model_name,
                        working_directory='{}'.format(DATA_DIR),
                        uri='mlexchange1/autoencoder-pytorch:0.1',
                        cmd= f"{command} -p \'{json.dumps(json_dict)}\'",
                        kwargs = {'job_type': action_selection,
                                  'experiment_id': experiment_id,
                                  'dataset': project_id,
                                  'params': json_dict,
                                  **kwargs})
        job.submit(USER, num_cpus, num_gpus)
        return False, counters, ''
    return False, counters, ''