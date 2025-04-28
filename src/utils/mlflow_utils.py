import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

logger = logging.getLogger(__name__)


def check_mlflow_ready():
    """
    Check if MLflow is reachable.

    Raises:
        Exception: If MLflow is not reachable
    """
    try:
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()  # noqa: F841
        return True
    except Exception as e:
        logger.warning(f"MLflow is not reachable: {e}")
        return False
