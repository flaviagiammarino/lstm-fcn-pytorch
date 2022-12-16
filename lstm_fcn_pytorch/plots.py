import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(x, y):
    '''
    Plot the time series in each class.

    Parameters:
    __________________________________
    x: np.array.
        Time series, array with shape (samples, channels, length) where samples is the number of time series,
        channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
        is the length of the time series.

    y: np.array.
        Predicted labels, array with shape (samples,) where samples is the number of time series.

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of time series, one subplot for each class.
    '''
    
    # extract the distinct clusters
    c = np.unique(y).astype(int)
    
    fig = make_subplots(
        subplot_titles=['Class ' + str(i + 1) for i in c],
        vertical_spacing=0.125,
        rows=len(c),
        cols=1
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=30, r=30),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'
        ),
    )
    
    fig.update_annotations(
        font=dict(
            color='#1b1f24',
            size=12,
        )
    )

    for i in range(len(c)):
    
        # extract the time series in the ith class
        x_ = x[y == c[i], :]
        
        for j in range(x_.shape[0]):
            fig.add_trace(
                go.Scatter(
                    y=x_[j, :],
                    name='Actual',
                    showlegend=False,
                    mode='lines',
                    line=dict(
                        color='rgba(175,184,193,0.2)',
                        width=0.5
                    )
                ),
                row=i + 1,
                col=1
            )
        
        fig.add_trace(
            go.Scatter(
                y=np.nanmean(x_, axis=0),
                showlegend=True if i == 0 else False,
                name='Class Average',
                mode='lines',
                line=dict(
                    color='#cf222e',
                    width=1,
                    shape='spline',
                    dash='dot',
                )
            ),
            row=i + 1,
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )
        
        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )
    
    return fig