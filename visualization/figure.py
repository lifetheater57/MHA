import json

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from visualization.config import get_at


def figure(df: pd.DataFrame, config: dict):
    # Initialize figure with subplots
    n_row = config["rows"]["count"]
    n_columns = config["columns"]["count"]
    fig = make_subplots(
        rows=n_row,
        cols=n_columns,
        column_titles=config["columns"]["title"],
        row_titles=config["rows"]["title"],
    )
    
    default_colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    colors = dict(zip(sum(config["columns"]["Method"], []), default_colors))
    
    for i, row_title in enumerate(config["rows"]["title"]):
        for j, column_title in enumerate(config["columns"]["title"]):
            for method in get_at(config["columns"]["Method"], j):
                # Add trace
                mask = (df["row"] == row_title) & (df["column"] == column_title) & (df["Method"] == method)
                show_legend = (j == n_columns - 1) and (i == 0)
                color = colors[method]
                x = df.loc[mask, get_at(config["columns"]["x"], j)]
                y = df.loc[mask, get_at(config["columns"]["y"], j)]
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=method, showlegend=show_legend, marker=dict(color=color), legendgroup=method), row=i+1, col=j+1)

                # Update axis properties
                fig.update_xaxes(title_text=get_at(config["columns"]["x"], j), row=i+1, col=j+1)
                fig.update_yaxes(title_text=get_at(config["columns"]["y"], j), row=i+1, col=j+1)

    # Update title and height
    fig.update_layout(height=650, width=650, template="plotly_dark")
    
    return fig