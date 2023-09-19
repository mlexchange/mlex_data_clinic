import os
import json
import urllib

CONTENT_URL = str(os.environ['MLEX_CONTENT_URL'])

def get_model_list():
    """
    Get a list of algorithms from content registry
    """
    response = urllib.request.urlopen(f'{CONTENT_URL}/models')
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
        model_uid:      Model UID
        comp_group:     Action, e.g. training, testing, etc
    Returns:
        params:         List of model parameters
    '''
    url = f'{CONTENT_URL}/models/{model_uid}/model/{comp_group}/gui_params'
    response = urllib.request.urlopen(url)
    return json.loads(response.read())