import os
import pathlib

import dash
import dash_bootstrap_components as dbc
import dash_uploader as du
import diskcache
from dash import html
from dash.long_callback import DiskcacheLongCallbackManager
from file_manager.main import FileManager

from components.header import header
from components.job_table import job_table
from components.loss import loss_plot
from components.main_display import main_display
from components.resources_setup import resources_setup
from components.sidebar import sidebar
from utils.job_utils import get_host
from utils.model_utils import get_model_list

USER = "admin"
DATA_DIR = os.getenv("DATA_DIR")
DOCKER_DATA = pathlib.Path.home() / "data"
UPLOAD_FOLDER_ROOT = DOCKER_DATA / "upload"
SPLASH_URL = os.getenv("SPLASH_URL")
DEFAULT_TILED_URI = os.getenv("DEFAULT_TILED_URI")
TILED_KEY = os.getenv("TILED_KEY")
if TILED_KEY == "":
    TILED_KEY = None
HOST_NICKNAME = os.getenv("HOST_NICKNAME")
num_processors, num_gpus = get_host(HOST_NICKNAME)

# SETUP DASH APP
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "../assets/mlex-style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    long_callback_manager=long_callback_manager,
)
app.title = "Data Clinic"
app._favicon = "mlex.ico"
dash_file_explorer = FileManager(
    DOCKER_DATA,
    UPLOAD_FOLDER_ROOT,
    open_explorer=False,
    api_key=TILED_KEY,
    splash_uri=SPLASH_URL,
    default_tiled_uri=DEFAULT_TILED_URI,
)
dash_file_explorer.init_callbacks(app)
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)

# DEFINE LAYOUT
app.layout = html.Div(
    [
        header(
            "MLExchange | Data Clinic", "https://github.com/mlexchange/mlex_data_clinic"
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            sidebar(dash_file_explorer.file_explorer, get_model_list()),
                            width=4,
                        ),
                        dbc.Col(main_display(loss_plot(), job_table()), width=8),
                        html.Div(id="dummy-output"),
                    ]
                ),
                resources_setup(num_processors, num_gpus),
            ],
            fluid=True,
        ),
    ]
)
