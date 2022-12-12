import json

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from visualization.config import get_at


def figure(df: pd.DataFrame, config: dict):
    # Initialize figure with subplots
    n_row = len(config["rows"]["title"])
    n_columns = len(config["columns"]["title"])
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
                y_confidence = df.loc[mask, get_at(config["columns"]["y_confidence"], j)]
                fig.add_trace(
                    go.Scatter(
                        x=x, 
                        y=y, 
                        mode="lines", 
                        name=method, 
                        showlegend=show_legend, 
                        marker=dict(color=color), 
                        legendgroup=method
                    ), 
                    row=i+1, 
                    col=j+1
                )
                
                if y_confidence.notna().all() and (y_confidence > 0).all():
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([x, x[::-1]]), 
                            y=pd.concat([y + y_confidence, (y - y_confidence)[::-1]]),
                            fill="toself",
                            fillcolor=f"rgba{color[3:-1]},0.2)",
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=i+1, 
                        col=j+1
                    )

                # Update axis properties
                fig.update_xaxes(title_text=get_at(config["columns"]["x"], j), row=i+1, col=j+1)
                fig.update_yaxes(title_text=get_at(config["columns"]["y"], j), row=i+1, col=j+1)

    # Update title and height
    fig.update_layout(height=650, width=650, template="plotly_dark")#title_text="Customizing Subplot Axes", 


    filename = f"output/plot-"# + datetime.now().strftime("%Y%m%d%H%M%S")
    pio.write_html(fig, filename + ".html")
    pio.write_image(fig, filename + ".png")

    return fig

df = pd.read_csv("output/data.csv")
with open("output/config.json") as file:
    figure_config = json.load(file)

figure(df, figure_config)