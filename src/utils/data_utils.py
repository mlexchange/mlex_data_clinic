import pathlib
import uuid
import pandas as pd


def prepare_directories(user_id, data_project, project_id):
    '''
    Prepare data directories that host experiment results and data movements processes for tiled
    If data is served through tiled, a local copy will be made for ML training and inference 
    processes in file system located at data/mlexchange_store/user_id/tiledprojectid_localprojectid
    Args:
        user_id:        User ID
        data_project:   List of data sets in the application
        project_id:     Current project ID
    Returns:
        experiment_id:  ML experiment ID
        out_path:       Path were experiment results will be stored
        project_id:     Project ID - will be overwritten if a local copy of a tiled dataset is provided
    '''
    experiment_id = str(uuid.uuid4())
    out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(user_id, experiment_id))
    out_path.mkdir(parents=True, exist_ok=True)
    uri_list = []
    uid_list = []
    for dataset in data_project.data:
        uri_list.append(dataset.uri)
        uid_list.append(dataset.uid)
    data_info = pd.DataFrame({'uri': uri_list, 'uid': uid_list, 'type': ['file']*len(data_project.data)})
    if data_project.data[0].type == 'tiled':
        local_path = pathlib.Path(f'data/tiled_local_copy/{project_id}')
        data_info['local_uri'] = [f'{local_path}/{uid}.tif' for uid in uid_list]
        if not local_path.exists():
            local_path.mkdir(parents=True)
            data_project.tiled_to_local_project(local_path, project_id)
    data_info.to_parquet(f'{out_path}/data_info.parquet', engine='pyarrow')
    return experiment_id, out_path, f'{out_path}/data_info.parquet'