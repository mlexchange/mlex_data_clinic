import os

import json
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import urllib.request

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests


class SimpleJob:
    def __init__(self,
                 service_type,
                 description,
                 working_directory,
                 uri,
                 cmd,
                 kwargs = None,
                 mlex_app = 'data_clinic'):
        self.mlex_app = mlex_app
        self.description = description
        self.service_type = service_type
        self.working_directory = working_directory
        self.job_kwargs = {'uri': uri,
                           'type': 'docker',
                           'cmd': cmd,
                           'kwargs': kwargs}

    def submit(self, user, num_cpus, num_gpus):
        '''
        Sends job to computing service
        Args:
            user:       user UID
            num_cpus:   Number of CPUs
            num_gpus:   Number of GPUs
        Returns:
            Workflow status
        '''
        workflow = {'user_uid': user,
                    'job_list': [self.__dict__],
                    'host_list': ['mlsandbox.als.lbl.gov', 'local.als.lbl.gov', 'vaughan.als.lbl.gov'],
                    'dependencies': {'0': []},
                    'requirements': {'num_processors': num_cpus,
                                     'num_gpus': num_gpus,
                                     'num_nodes': 1}}
        url = 'http://job-service:8080/api/v0/workflows'
        return requests.post(url, json=workflow).status_code
    

def load_from_dir(data_path):
    '''
    Loads data from directory
    Args:
        data_path:      Path to data
    Returns:
        data_files:     Dictionary with the list of filenames for training and testing
    '''
    folders = ['/train', '/test']
    keys = ['x_train', 'x_test']
    data_type = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
    data_files = {keys[0]: [], keys[1]: []}        # dictionary of filenames
    for folder, key in zip(folders, keys):
        path, train_folders, extra_files = next(os.walk(data_path+folder))
        for subfolder in train_folders:
            path, list_dirs, filenames = next(os.walk(data_path+folder + '/' + subfolder))
            for filename in filenames:
                if filename.split('.')[-1] in data_type:
                    data_files[key].append(data_path + folder + '/' + subfolder + '/' + filename)
    return data_files


def get_job(user, mlex_app, job_type=None, deploy_location=None):
    '''
    Queries the job from the computing database
    Args:
        user:               username
        mlex_app:           mlexchange application
        job_type:           type of job
        deploy_location:    deploy location
    Returns:
        list of jobs that match the query
    '''
    url = 'http://job-service:8080/api/v0/jobs?'
    if user:
        url += ('&user=' + user)
    if mlex_app:
        url += ('&mlex_app=' + mlex_app)
    if job_type:
        url += ('&job_type=' + job_type)
    if deploy_location:
        url += ('&deploy_location=' + deploy_location)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


def generate_loss_plot(log, start):
    '''
    Generate loss plot
    Args:
        log:    job logs with the loss/accuracy per epoch
        start:  index where the list of loss values start
    Returns:
        loss plot
    '''
    end = log.find('Train process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=' ')
    df.set_index('epoch', inplace=True)
    try:
        fig = px.line(df, markers=True)
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss", margin=dict(l=20, r=20, t=20, b=20))
        return fig
    except Exception as e:
        print(e)
        return go.Figure(go.Scatter(x=[], y=[]))

def str_to_dict(text):
    text = text.replace('True', 'true')
    text = text.replace('False', 'false')
    text = text.replace('None', 'null')
    return json.loads(text.replace('\'', '\"'))


def model_list_GET_call():
    """
    Get a list of algorithms from content registry
    """
    url = 'http://content-api:8000/api/v0/models'
    response = urllib.request.urlopen(url)
    list = json.loads(response.read())
    models = []
    for item in list:
        if 'data_clinic' in item['application']:
            models.append({'label': item['name'], 'value': item['content_id']})
    return models


def get_gui_components(model_uid, comp_group):
    '''
    Returns the GUI components of the corresponding model and action
    Args:
        model_uid:  Model UID
        comp_group: Action, e.g. training, testing, etc
    Returns:
        params:     List of model parameters
    '''
    url = f'http://content-api:8000/api/v0/models/{model_uid}/model/{comp_group}/gui_params'
    response = urllib.request.urlopen(url)
    return json.loads(response.read())


def get_counter(username):
    job_list = get_job(username, 'data_clinic')
    job_types = ['train_model', 'prediction_model']
    counters = [-1, -1]
    if job_list is not None:
        for indx, job_type in enumerate(job_types):
            for job in reversed(job_list):
                last_job = job['job_kwargs']['kwargs']['job_type']
                if job['description']:
                     job_name = job['description'].split()
                else:
                    job_name = job['job_kwargs']['kwargs']['job_type'].split()
                if last_job == job_type and job_name[0] == job_type and len(job_name)==2 and job_name[-1].isdigit():
                    value = int(job_name[-1])
                    counters[indx] = value
                    break
    return counters


def get_host(host_nickname):
    hosts = requests.get(f'http://job-service:8080/api/v0/hosts?&nickname={host_nickname}').json()
    max_processors = hosts[0]['backend_constraints']['num_processors']
    max_gpus = hosts[0]['backend_constraints']['num_gpus']
    return max_processors, max_gpus

