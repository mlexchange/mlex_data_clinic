import os
import pathlib

import dash
from dash import html
import dash_uploader as du
import dash_bootstrap_components as dbc

from file_manager.file_manager import FileManager
from components.header import header
from components.job_table import job_table
from components.loss import loss_plot
from components.main_display import main_display
from components.resources_setup import resources_setup
from components.sidebar import sidebar
from utils.job_utils import get_host, TableJob
from utils.model_utils import get_model_list

USER = 'admin'
DATA_DIR = str(os.environ['DATA_DIR'])
DOCKER_DATA = pathlib.Path.home() / 'data'
LOCAL_DATA = str(os.environ['DATA_DIR'])
DOCKER_HOME = str(DOCKER_DATA) + '/'
LOCAL_HOME = str(LOCAL_DATA)
UPLOAD_FOLDER_ROOT = DOCKER_DATA / 'upload'
SPLASH_URI = str(os.environ['SPLASH_URL'])
TILED_KEY = str(os.environ['TILED_KEY'])
HOST_NICKNAME = str(os.environ['HOST_NICKNAME'])
num_processors, num_gpus = get_host(HOST_NICKNAME)

#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/mlex-style.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Data Clinic"
app._favicon = 'mlex.ico'
dash_file_explorer = FileManager(DOCKER_DATA, UPLOAD_FOLDER_ROOT, open_explorer=False, 
                                 api_key=TILED_KEY)
dash_file_explorer.init_callbacks(app)
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)

##### DEFINE LAYOUT ####
app.layout = html.Div(
    [
        header("MLExchange | Data Clinic",
               "https://github.com/mlexchange/mlex_data_clinic"),
        dbc.Container(
            [
                dbc.Row(
                    [dbc.Col(sidebar(dash_file_explorer.file_explorer, get_model_list(), 
                                     TableJob.get_counter(USER)), 
                             width=4),
                     dbc.Col(main_display(loss_plot(), job_table()), 
                             width=8),
                     html.Div(id='dummy-output')]
                ),
                resources_setup(num_processors, num_gpus)
            ],
            fluid=True
        )
    ]
)