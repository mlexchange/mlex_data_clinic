import base64
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px


def generate_loss_plot(log, start):
    '''
    Generate loss plot
    Args:
        log:    job logs with the loss/accuracy per epoch
        start:  index where the list of loss values start
    Returns:
        loss plot
    '''
    end = log.find('Train process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=',')
    df.set_index('epoch', inplace=True)
    try:
        fig = px.line(df, markers=True)
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss", margin=dict(l=20, r=20, t=20, b=20))
        return fig
    except Exception as e:
        print(e)
        return go.Figure(go.Scatter(x=[], y=[]))
    

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
    # ratio between flatten input data and selected latent space size
    ratio = 400 / (width * height)
    annotation1 = str(width)+'x'+str(height)    # target data size
    annotation2 = str(ls_var)+'x1'              # target latent space
    # if the latent space is larger than the data dimension (flatten), the bottleneck is shown in red
    if ls_var>width*height:
        color = 'rgba(238, 69, 80, 1)'
    else:
        color = 'rgba(168, 216, 234, 1)'
    # adjusting the latent space with respect to the images size in frontend
    ls_var = ls_var*ratio
    x = [-200, -200, 0, 200, 200, 0, -200]
    y = [-200, 200, ls_var / 2, 200, -200, -ls_var / 2, -200]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            fill='toself',
            fillcolor=color,
            line_color=color
            )
        )
    fig.add_shape(type="rect",
                  x0=-1,
                  y0=ls_var/2,
                  x1=1,
                  y1=-ls_var/2,
                  fillcolor="RoyalBlue",
                  line_color="RoyalBlue")
    fig.update_traces(marker_size=1,
                      hoverinfo='skip')
    if annotations:
        fig.add_annotation(x=-187,
                           y=-25,
                           text=annotation1,
                           textangle=270,
                           font={'size': 28}
                           )
        fig.add_annotation(x=199,
                           y=-25,
                           text=annotation1,
                           textangle=270,
                           font={'size': 28}
                           )
        fig.add_annotation(x=-10,
                           y=0,
                           text=annotation2,
                           textangle=270,
                           font={'size': 28},
                           showarrow=False)
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