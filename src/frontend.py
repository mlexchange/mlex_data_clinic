from dash.dependencies import Input, Output

from callbacks.display import *
from callbacks.execute import *
from callbacks.table import *
from helper_utils import get_gui_components
from dash_component_editor import JSONParameterEditor
from app_layout import app


##### DATA RELATED CALLBACKS ####

# @app.callback(
#     Output("warning-modal", "is_open"),
#     Output("warning-msg", "children"),
#     Input("warning-cause", "data"),
#     Input("ok-button", "n_clicks"),
#     State("warning-modal", "is_open"),
#     prevent_initial_call=True
# )
# def toggle_warning_modal(warning_cause, ok_n_clicks, is_open):
#     '''
#     This callback toggles a warning/error message
#     Args:
#         warning_cause:      Cause that triggered the warning
#         ok_n_clicks:        Close the warning
#         is_open:            Close/open state of the warning
#     '''
#     changed_id = dash.callback_context.triggered[0]['prop_id']
#     if 'ok-button.n_clicks' in changed_id:
#         return not is_open, ""
#     if warning_cause == 'wrong_dataset':
#         return not is_open, "The dataset you have selected is not supported. Please select (1) a data directory " \
#                         "where each subfolder corresponds to a given category, OR (2) an NPZ file."
#     if warning_cause == 'different_size':
#         return not is_open, "The number of images and labels do not match. Please select a different dataset."
#     if warning_cause == 'no_row_selected':
#         return not is_open, "Please select a trained model from the List of Jobs"
#     if warning_cause == 'no_dataset':
#         return not is_open, "Please upload the dataset before submitting the job."
#     else:
#         return False, ""


##### DATA CLINIC CALLBACKS  ####
@app.callback(
    Output('app-parameters', 'children'),
    Input('model-selection', 'value'),
    Input('action', 'value')
)
def load_parameters_and_content(model_selection, action_selection):
    '''
    This callback dynamically populates the parameters and contents of the website according to the selected action &
    model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in Data Clinic)
    Returns:
        app-parameters:     Parameters according to the selected model & action
    '''
    print('I am here')
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(_id={'type': 'parameter_editor'}, # pattern match _id (base id), name
                                   json_blob=parameters,
                                   )
    gui_item.init_callbacks(app)
    return gui_item


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8072)
