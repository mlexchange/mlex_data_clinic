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
        """
        Send job to computing service
        :return:
        """
        url = 'http://job-service:8080/api/v0/jobs'
        return requests.post(url, json=self.__dict__).status_code


def plot_figure(image):
    try:
        h,w = image.size
    except:
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


def get_bottleneck(ls_var):
    # x = [-200, -200,     -5,        0,    5, 200,  200,      5,         0,     -5, -200]
    # y = [-200,  200, 200/8, ls_var/2, 200/8, 200, -200, -200/8, -ls_var/2, -200/8, -200]
    x = [-200, -200, 0, 200, 200, 0, -200]
    y = [-200, 200,  ls_var/2, 200, -200, -ls_var/2, -200]
    fig = go.Figure(go.Scatter(x=x, y=y,
                               fill='toself',
                               fillcolor='rgba(168, 216, 234, 1)',
                               line_color='rgba(168, 216, 234, 1)'))
    fig.update_traces(marker_size=1, hoverinfo='skip')
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
    # return fig


# Queries the job from the computing database
def get_job(user, mlex_app, job_type=None, deploy_location=None):
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


# Generate loss plot
def generate_loss_plot(log, start):
    end = log.find('Train process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep='\t')
    df.set_index('epoch', inplace=True)
    try:
        fig = px.line(df, markers=True)
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss", margin=dict(l=20, r=20, t=20, b=20))
        return fig
    except Exception as e:
        print(e)
        return go.Figure(go.Scatter(x=[], y=[]))
