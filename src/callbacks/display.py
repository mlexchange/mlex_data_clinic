import base64
from dash import Input, Output, State, callback, ALL
import dash
import numpy as np
from PIL import Image
import plotly
import plotly.graph_objects as go
import plotly.express as px

from file_manager.data_project import DataProject
from app_layout import USER
from helper_utils import str_to_dict


def plot_figure(image):
    '''
    Plots images in frontend
    Args:
        image:  Image to plot
    Returns:
        plot in base64 format
    '''
    try:
        h,w = image.size
    except Exception:
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
    try:
        fig.update_traces(dict(showscale=False, coloraxis=None))
    except Exception as e:
        print('plot error')
    png = plotly.io.to_image(fig, format='jpg')
    png_base64 = base64.b64encode(png).decode('ascii')
    return "data:image/jpg;base64,{}".format(png_base64)


def get_bottleneck(ls_var, width, height, annotations=True):
    '''
    Plots the latent space representation
    Args:
        ls_var:         latent space value
        width:          data width
        height:         data height
        annotations:    Bool
    Returns:
        plot with graphical representation of the latent space in base64 format
    '''
    ratio = 400 / (width * height)              # ratio between flatten input data and selected latent space size
    annotation1 = str(width)+'x'+str(height)    # target data size
    annotation2 = str(ls_var)+'x1'              # target latent space
    if ls_var>width*height:                     # if the latent space is larger than the data dimension (flatten),
        color = 'rgba(238, 69, 80, 1)'          # the bottleneck is shown in red
    else:
        color = 'rgba(168, 216, 234, 1)'
    ls_var = ls_var*ratio                       # adjusting the latent space with respect to the images size in frontend
    x = [-200, -200, 0, 200, 200, 0, -200]
    y = [-200, 200, ls_var / 2, 200, -200, -ls_var / 2, -200]
    fig = go.Figure(go.Scatter(x=x, y=y,
                               fill='toself',
                               fillcolor=color,
                               line_color=color))
    fig.add_shape(type="rect",
                  x0=-1, y0=ls_var/2,
                  x1=1, y1=-ls_var/2,
                  fillcolor="RoyalBlue",
                  line_color="RoyalBlue")
    fig.update_traces(marker_size=1, hoverinfo='skip')
    if annotations:
        fig.add_annotation(x=-187, y=-25, text=annotation1, textangle=270, font={'size': 28})
        fig.add_annotation(x=199, y=-25, text=annotation1, textangle=270, font={'size': 28})
        fig.add_annotation(x=-10, y=0, text=annotation2, textangle=270, font={'size': 28}, showarrow=False)
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


@callback(
    Output('orig_img', 'src'),
    Output('rec_img', 'src'),
    Output('ls_graph', 'src'),
    Output('img-slider', 'max'),
    Output('img-slider', 'value'),
    Output('data-size-out', 'children'),
    Output('current-image-label', 'children'),

    Input({'base_id': 'file-manager', 'name': "docker-file-paths"}, "data"),
    Input({'type': ALL, 'param_key': 'latent_dim', 'name': 'latent_dim', 'layer': 'input'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_width', 'name': 'target_width'}, 'value'),
    Input({'type': ALL, 'param_key': 'target_height', 'name': 'target_height'}, 'value'),
    Input('img-slider', 'value'),
    Input('action', 'value'),
    Input('jobs-table', 'selected_rows'),
    Input('jobs-table', 'data'),
)
def refresh_image(file_paths, ls_var, target_width, target_height, img_ind, action_selection, row, \
                  data_table):
    '''
    This callback updates the images in the display
    Args:
        import_dir:         Import button
        confirm_import:     Confirm import button
        ls_var:             Latent space value
        target_width:       Target data width (if resizing)
        target_height:      Target data height (if resizing)
        img_ind:            Index of image according to the slider value
        row:                Selected job (model) 
        data_table:         Data in table of jobs
        file_paths:          Selected data files
        img_keyword:        Keyword for images in NPZ file
        npz_modal:          Open/close status of NPZ modal
        action_selection:   Action selection (train vs test)
    Returns:
        img-output:         Output figure
        img-reconst-output: Reconstructed output (if prediction is selected, ow. blank image)
        latent-space-plot:  Graphical representation of latent space definition
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
        img-slider-value:   Value of the slider according to the dataset length
        data-size-out:      Size of uploaded data
    '''
    current_im_label = ''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    if action_selection in ['train_model', 'transfer_learning']:
        if len(ls_var) > 0:
            ls_var = int(ls_var[0])
            target_width = int(target_width[0])
            target_height = int(target_height[0])
            ls_plot = get_bottleneck(ls_var, target_width, target_height)
        else:
            ls_plot = dash.no_update
    else:
        ls_plot = get_bottleneck(1, 1, 1, False)
        target_width = None
    if row:
        if row[0] < len(data_table):
            if data_table[row[0]]['job_type'].split()[0] == 'train_model':
                if action_selection == 'prediction_model':
                    train_params = str_to_dict(data_table[row[0]]['parameters'])
                    ls_var = int(train_params['latent_dim'])
                    target_width = int(train_params['target_width'])
                    target_height = int(train_params['target_height'])
                    ls_plot = get_bottleneck(ls_var, target_width, target_height)
            else:
                job_id = data_table[row[0]]['experiment_id']
                dataset = data_table[row[0]]['dataset']
                data_project.init_from_splash((f'http://splash:80/api/v0/datasets/search', dataset))
                reconstructed_path = 'data/mlexchange_store/{}/{}/'.format(USER, job_id)
                try:
                    slider_max = len(data_project.data)
                    img_ind = min(slider_max, img_ind)
                    filename = data_project.data[img_ind].uri.split('/')[-1].split('.')[0]
                    reconst_img= Image.open(f'{reconstructed_path}/{filename}_reconstructed.jpg')
                except Exception as e:
                    print(f'Reconstructed images are not ready due to {e}')
                indx = data_table[row[0]]['parameters'].find('Training Parameters:')
                train_params = str_to_dict(data_table[row[0]]['parameters'][indx + 21:])
                ls_var = int(train_params['latent_dim'])
                target_width = int(train_params['target_width'])
                target_height = int(train_params['target_height'])
                if 'img-slider.value' in changed_id:
                    ls_plot = dash.no_update
                else:
                    ls_plot = get_bottleneck(ls_var, target_width, target_height)
    if len(data_project.data) > 0:
        try:
            slider_max = len(data_project.data) - 1
            if img_ind > slider_max:
                img_ind = 0
            origimg, current_im_label = data_project.data[img_ind].read_data(export='pillow')
        except Exception as e:
            print(f'Exception in refresh_image callback {e}')
    if 'origimg' not in locals():
        origimg = Image.fromarray((np.zeros((32,32)).astype(np.uint8)))
        slider_max = 0
    (width, height) = origimg.size
    if 'reconst_img' not in locals():
        reconst_img = Image.fromarray((np.zeros(origimg.size).astype(np.uint8)))
    if target_width:
        origimg = plot_figure(origimg.resize((target_width, target_height)))
        recimg = plot_figure(reconst_img.resize((target_width, target_height)))
        data_size = 'Original Image: (' + str(width) + 'x' + str(height) + '). Resized Image: (' + \
                    str(target_width) + 'x' + str(target_height) + ').'
    else:
        origimg = plot_figure(origimg)
        recimg = plot_figure(reconst_img)
        data_size = 'Original Image: (' + str(width) + 'x' + str(height) + \
                    '). Choose a trained model to update the graph.'
    return origimg, recimg, ls_plot, slider_max, img_ind, data_size, 'Image: '+current_im_label
