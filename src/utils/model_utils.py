import os

import requests

CONTENT_URL = str(os.environ["MLEX_CONTENT_URL"])


def get_model_list():
    """
    Get a list of algorithms from content registry
    """
    models = []
    response = requests.get(f"{CONTENT_URL}/models")
    if response.status_code == 200:
        for item in response.json():
            if "data_clinic" in item["application"]:
                models.append({"label": item["name"], "value": item["content_id"]})
    else:
        models.append({"label": "No models available", "value": ""})
    return models


def get_gui_components(model_uid, comp_group):
    """
    Returns the GUI components of the corresponding model and action
    Args:
        model_uid:      Model UID
        comp_group:     Action, e.g. training, testing, etc
    Returns:
        params:         List of model parameters
    """
    response = requests.get(
        f"{CONTENT_URL}/models/{model_uid}/model/{comp_group}/gui_params"
    )
    return response.json()


def get_model_content(content_id):
    """
    Get the model content: uri and commands
    """
    response = requests.get(f"{CONTENT_URL}/contents/{content_id}/content").json()
    return response["uri"], response["cmd"]
