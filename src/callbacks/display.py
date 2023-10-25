import pathlib
import time

from dash import Input, Output, State, callback, ALL
import dash
import numpy as np
import pandas as pd
from PIL import Image

from app_layout import USER, TILED_KEY
from file_manager.data_project import DataProject
from utils.job_utils import str_to_dict
from utils.plot_utils import get_bottleneck, plot_figure


@callback(
    Output('orig_img', 'src'),
    Output('rec_img', 'src'),
    Output('ls_graph', 'src'),
    Output('img-slider', 'max'),
    Output('img-slider', 'value'),
    Output('data-size-out', 'children'),

    Input({'base_id': 'file-manager', 'name': "docker-file-paths"}, "data"),
    Input({'type': ALL, 'param_key': 'latent_dim', 'name': 'latent_dim', 'layer': 'input'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_width', 'name': 'target_width'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_height', 'name': 'target_height'}, 'value'),
    Input('img-slider', 'value'),
    Input('action', 'value'),
    Input('jobs-table', 'selected_rows'),
    Input('jobs-table', 'data'),
)
def refresh_image(file_paths, ls_var, target_width, target_height, img_ind, action_selection, row,
                  data_table):
    '''
    This callback updates the images in the display
    Args:
        file_paths:         Selected data files
        ls_var:             Latent space value
        target_width:       Target data width (if resizing)
        target_height:      Target data height (if resizing)
        img_ind:            Index of image according to the slider value
        action_selection:   Action selection (train vs test)
        row:                Selected job (model) 
        data_table:         Data in table of jobs
    Returns:
        img-output:         Output figure
        img-reconst-output: Reconstructed output (if prediction is selected, ow. blank image)
        latent-space-plot:  Graphical representation of latent space definition
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
        img-slider-value:   Value of the slider according to the dataset length
        data-size-out:      Size of uploaded data
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data_project = DataProject()
    if row and len(row)>0 and data_table[row[0]]['job_type']=='train_model' and \
        action_selection=='prediction_model':
        data_project.init_from_dict(file_paths)
        train_params = str_to_dict(data_table[row[0]]['parameters'])
        ls_var = int(train_params['latent_dim'])
        target_width = int(train_params['target_width'])
        target_height = int(train_params['target_height'])
        if changed_id != 'img-slider.value':
            ls_plot = get_bottleneck(ls_var, target_width, target_height)
        else:
            ls_plot = dash.no_update
    elif row and len(row)>0 and data_table[row[0]]['job_type']=='prediction_model':
        job_id = data_table[row[0]]['experiment_id']
        data_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, job_id))
        data_info = pd.read_parquet(f'{data_path}/data_info.parquet', engine='pyarrow')
        data_project.init_from_dict(data_info.to_dict('records'), api_key=TILED_KEY)
        reconstructed_path = 'data/mlexchange_store/{}/{}/'.format(USER, job_id)
        try:
            slider_max = len(data_project.data)
            img_ind = min(slider_max, img_ind)
            uri = data_project.data[img_ind].uri.split('/')[-1]
            reconst_img= Image.open(f'{reconstructed_path}reconstructed_{uri}.jpg')
        except Exception as e:
            print(f'Reconstructed images are not ready due to {e}')
        train_params = data_table[row[0]]['parameters'].split('Training Parameters:')[-1]
        train_params = str_to_dict(train_params)
        ls_var = int(train_params['latent_dim'])
        target_width = int(train_params['target_width'])
        target_height = int(train_params['target_height'])
        if changed_id != 'img-slider.value':
            ls_plot = get_bottleneck(ls_var, target_width, target_height)
        else:
            ls_plot = dash.no_update
    elif action_selection == 'train_model':
        data_project.init_from_dict(file_paths)
        target_width = int(target_width[0])
        target_height = int(target_height[0])
        ls_var = int(ls_var[0])
        if changed_id != 'img-slider.value':
            ls_plot = get_bottleneck(ls_var, target_width, target_height)
        else:
            ls_plot = dash.no_update
    else:
        data_project.init_from_dict(file_paths)
        if changed_id != 'img-slider.value':
            ls_plot = get_bottleneck(1, 1, 1, False)
        else:
            ls_plot = dash.no_update
        target_width = None
    if len(data_project.data) > 0:
        slider_max = len(data_project.data) - 1
        if img_ind > slider_max:
            img_ind = 0
        origimg, _ = data_project.data[img_ind].read_data(export='pillow')
    else:
        origimg = Image.fromarray((np.zeros((32,32)).astype(np.uint8)))
        slider_max = 0
    (width, height) = origimg.size
    if 'reconst_img' not in locals():
        reconst_img = Image.fromarray((np.zeros(origimg.size).astype(np.uint8)))
    origimg = plot_figure(origimg.resize((target_width, target_height)))
    recimg = plot_figure(reconst_img.resize((target_width, target_height)))
    data_size = f'Original Image: ({width}x{height}). Resized Image: ({target_width}x{target_height}).'
    return origimg, recimg, ls_plot, slider_max, img_ind, data_size


@callback(
    Output("warning-modal", "is_open"),
    Output("warning-msg", "children"),

    Input("warning-cause", "data"),
    Input("ok-button", "n_clicks"),
    State("warning-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_warning_modal(warning_cause, ok_n_clicks, is_open):
    '''
    This callback toggles a warning/error message
    Args:
        warning_cause:      Cause that triggered the warning
        ok_n_clicks:        Close the warning
        is_open:            Close/open state of the warning
    Returns:
        is_open:            Close/open state of the warning
         warning_msg:       Warning message
    '''
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'ok-button.n_clicks' in changed_id:
        return not is_open, ""
    if warning_cause == 'wrong_dataset':
        return not is_open, "The dataset you have selected is not supported."
    if warning_cause == 'no_row_selected':
        return not is_open, "Please select a trained model from the List of Jobs"
    if warning_cause == 'no_dataset':
        return not is_open, "Please upload the dataset before submitting the job."
    if warning_cause == 'data_project_not_ready':
        return not is_open, "The data project is still being created. Please try again \
                             in a couple minutes."
    else:
        return False, ""