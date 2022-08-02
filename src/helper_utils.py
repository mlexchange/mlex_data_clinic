import os
import base64
import json
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import urllib.request

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests


class SimpleJob:
    def __init__(self,
                 service_type,
                 working_directory,
                 uri,
                 cmd,
                 kwargs = None,
                 mlex_app = 'data_clinic'):
        self.mlex_app = mlex_app
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


def plot_figure(image):
    '''
    Plots images in frontend
    Args:
        image:  Image to plot
    Returns:
        plot in base64 format
    '''
    try:
        h,w = image.size
    except Exception:
        h,w,c = image.size
    fig = px.imshow(image, height=200, width=200*w/h)
    fig.update_xaxes(showgrid=False,
                     showticklabels=False,
                     zeroline=False,
                     fixedrange=True)
    fig.update_yaxes(showgrid=False,
                     showticklabels=False,
                     zeroline=False,
                     fixedrange=True)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(dict(showscale=False, coloraxis=None))
    png = plotly.io.to_image(fig)
    png_base64 = base64.b64encode(png).decode('ascii')
    return "data:image/png;base64,{}".format(png_base64)


def get_bottleneck(ls_var, width, height, annotations=True):
    '''
    Plots the latent space representation
    Args:
        ls_var:         latent space value
        width:          data width
        height:         data height
        annotations:    Bool
    Returns:
        plot with graphical representation of the latent space in base64 format
    '''
    ratio = 400 / (width * height)              # ratio between flatten input data and selected latent space size
    annotation1 = str(width)+'x'+str(height)    # target data size
    annotation2 = str(ls_var)+'x1'              # target latent space
    if ls_var>width*height:                     # if the latent space is larger than the data dimension (flatten),
        color = 'rgba(238, 69, 80, 1)'          # the bottleneck is shown in red
    else:
        color = 'rgba(168, 216, 234, 1)'
    ls_var = ls_var*ratio                       # adjusting the latent space with respect to the images size in frontend
    x = [-200, -200, 0, 200, 200, 0, -200]
    y = [-200, 200, ls_var / 2, 200, -200, -ls_var / 2, -200]
    fig = go.Figure(go.Scatter(x=x, y=y,
                               fill='toself',
                               fillcolor=color,
                               line_color=color))
    fig.add_shape(type="rect",
                  x0=-1, y0=ls_var/2,
                  x1=1, y1=-ls_var/2,
                  fillcolor="RoyalBlue",
                  line_color="RoyalBlue")
    fig.update_traces(marker_size=1, hoverinfo='skip')
    if annotations:
        fig.add_annotation(x=-187, y=-25, text=annotation1, textangle=270, font={'size': 28})
        fig.add_annotation(x=199, y=-25, text=annotation1, textangle=270, font={'size': 28})
        fig.add_annotation(x=-10, y=0, text=annotation2, textangle=270, font={'size': 28}, showarrow=False)
    fig.update_xaxes(range=[-200,200],
                     showgrid=False,
                     showticklabels=False,
                     zeroline=False,
                     fixedrange=True)
    fig.update_yaxes(range=[-200,200],
                     showgrid=False,
                     showticklabels=False,
                     zeroline=False,
                     fixedrange=True)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=0, b=0))
    png = plotly.io.to_image(fig)
    png_base64 = base64.b64encode(png).decode('ascii')
    return "data:image/png;base64,{}".format(png_base64)


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
                last_job = job['job_kwargs']['kwargs']['job_type'].split()
                value = int(last_job[-1])
                last_job = ' '.join(last_job[0:-1])
                if last_job == job_type:
                    counters[indx] = value
                    break
    return counters