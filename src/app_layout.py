import logging
import os

import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import dcc, html
from dash.long_callback import DiskcacheLongCallbackManager
from dotenv import load_dotenv
from file_manager.main import FileManager
from mlex_utils.dash_utils.mlex_components import MLExComponents

from src.components.header import header
from src.components.main_display import main_display
from src.components.sidebar import sidebar
from src.components.training_stats import training_stats_plot
from src.utils.model_utils import Models

load_dotenv(".env")

USER = os.getenv("USER", "admin")
DATA_DIR = os.getenv("DATA_DIR", "data")
SPLASH_URL = os.getenv("SPLASH_URL")
DEFAULT_TILED_URI = os.getenv("DEFAULT_TILED_URI")
DEFAULT_TILED_SUB_URI = os.getenv("DEFAULT_TILED_SUB_URI")
TILED_KEY = os.getenv("TILED_KEY")
if TILED_KEY == "":
    TILED_KEY = None
MODELFILE_PATH = os.getenv("MODELFILE_PATH", "./examples/assets/models.json")
MODE = os.getenv("MODE", "dev")
PREFECT_TAGS = os.getenv("PREFECT_TAGS", ["data-clinic"])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# SETUP FILE MANAGER
dash_file_explorer = FileManager(
    DATA_DIR,
    open_explorer=False,
    api_key=TILED_KEY,
    logger=logger,
)
dash_file_explorer.init_callbacks(app)
file_explorer = dash_file_explorer.file_explorer

# GET MODELS
models = Models(modelfile_path="./src/assets/default_models.json")

# SETUP MLEx COMPONENTS
mlex_components = MLExComponents("dbc")
job_manager = mlex_components.get_job_manager(
    model_list=models.modelname_list,
    mode=MODE,
    aio_id="data-clinic-jobs",
    prefect_tags=PREFECT_TAGS,
)

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
                            sidebar(file_explorer, job_manager),
                            style={"flex": "0 0 500px"},
                        ),
                        dbc.Col(main_display(training_stats_plot())),
                        html.Div(id="dummy-output"),
                    ]
                ),
                dcc.Store(id="current-target-size", data=[0, 0]),
            ],
            fluid=True,
        ),
    ]
)
