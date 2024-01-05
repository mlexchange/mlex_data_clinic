import pathlib
import re
import uuid
import pandas as pd


def prepare_directories(user_id, data_project, project_id, pattern = r'[/\\?%*:|"<>]'):
    '''
    Prepare data directories that host experiment results and data movements processes for tiled
    If data is served through tiled, a local copy will be made for ML training and inference 
    processes in file system located at data/mlexchange_store/user_id/tiledprojectid_localprojectid
    Args:
        user_id:        User ID
        data_project:   List of data sets in the application
        project_id:     Current project ID
        pattern:        List of patterns to remove from data set uri
    Returns:
        experiment_id:  ML experiment ID
        out_path:       Path were experiment results will be stored
        info_file:      Filename of a parquet file that contains the list of data sets within the
                        current project
    '''
    experiment_id = str(uuid.uuid4())
    out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(user_id, experiment_id))
    out_path.mkdir(parents=True, exist_ok=True)
    if data_project.data[0].type == 'tiled':
        data_info = data_project.tiled_to_local_project(project_id)
    else:
        uri_list = []
        for dataset in data_project.data:
            uri_list.append(dataset.uri)
        data_info = pd.DataFrame({'uri': uri_list})
        data_info['type'] = ['file']*len(data_project.data)
    data_info.to_parquet(f'{out_path}/data_info.parquet', engine='pyarrow')
    return experiment_id, out_path, f'{out_path}/data_info.parquet'


def get_input_params(children):
    '''
    Gets the model parameters and its corresponding values
    '''
    input_params = {}
    if bool(children):
        try:
            for child in children['props']['children']:
                key = child["props"]["children"][1]["props"]["id"]["param_key"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        except Exception:
            for child in children:
                key = child["props"]["children"][1]["props"]["id"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
    return input_params