from dash.dependencies import Input, Output, State
from uuid import uuid4

from app_layout import app
from callbacks.display import refresh_image, toggle_warning_modal
from callbacks.execute import execute
from callbacks.table import update_table, delete_row
from dash_component_editor import JSONParameterEditor
from utils.model_utils import get_gui_components


@app.callback(
    Output('app-parameters', 'children'),
    Input('model-selection', 'value'),
    Input('action', 'value')
)
def load_parameters_and_content(model_selection, action_selection):
    '''
    This callback dynamically populates the parameters and contents of the website according to the 
    selected action & model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in Data Clinic)
    Returns:
        app-parameters:     Parameters according to the selected model & action
    '''
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(_id={'type': str(uuid4())}, # pattern match _id (base id), name
                                   json_blob=parameters,
                                   )
    gui_item.init_callbacks(app)
    return gui_item


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
