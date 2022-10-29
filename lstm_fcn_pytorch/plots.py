import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(x, y):
    
    '''
    Plot the time series in each class.
    
    Parameters:
    __________________________________
    x: np.array.
        Time series, array with shape (samples, length) where samples is the number of time series
        and length is the length of the time series.
    
    y: np.array.
        Predicted labels, array with shape (samples,) where samples is the number of time series.
        
    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of time series, one subplot for each class.
    '''
    
    # extract the distinct classes
    c = np.unique(y)
    
    fig = make_subplots(
        subplot_titles=['Class ' + str(i + 1) for i in c],
        vertical_spacing=0.15,
        rows=len(c),
        cols=1
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(
            color='#000000',
            size=10,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#000000',
            ),
        ),
    )
    
    fig.update_annotations(
        font=dict(
            size=13
        )
    )
    
    for i in range(len(c)):
        
        # extract the time series in the ith class
        x_ = x[y == c[i], :]
        
        for j in range(x_.shape[0]):
            fig.add_trace(
                go.Scatter(
                    y=x_[j, :],
                    showlegend=False,
                    mode='lines',
                    line=dict(
                        color='rgba(194, 194, 194, 0.5)',
                        width=0.1
                    )
                ),
                row=i + 1,
                col=1
            )
        
        fig.add_trace(
            go.Scatter(
                y=np.nanmean(x_, axis=0),
                showlegend=True if i == 0 else False,
                name='Average',
                legendgroup='Average',
                mode='lines',
                line=dict(
                    color='rgb(130, 80, 223)',
                    width=1
                )
            ),
            row=i + 1,
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )
        
        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )
    
    return fig