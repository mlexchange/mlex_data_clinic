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
                 user,
                 job_type,
                 description,
                 deploy_location,
                 gpu,
                 data_uri,
                 container_uri,
                 container_cmd,
                 container_kwargs,
                 mlex_app = 'data_clinic'):
        self.user = user
        self.mlex_app = mlex_app
        self.job_type = job_type
        self.description = description
        self.deploy_location = deploy_location
        self.gpu = gpu
        self.data_uri = data_uri
        self.container_uri = container_uri
        self.container_cmd = container_cmd
        self.container_kwargs = container_kwargs

    def launch_job(self):
        '''
        Send job to computing service
        Returns:
            Job status
        '''
        url = 'http://job-service:8080/api/v0/jobs'
        return requests.post(url, json=self.__dict__).status_code


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
        w,h = image.size
    except Exception:
        w,h,c = image.size
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


def get_bottleneck(ls_var, width, height):
    '''
    Plots the latent space representation
    Args:
        ls_var:     latent space value
        width:      data width
        height:     data height
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
    return json.loads(text.replace('\'', '\"'))
