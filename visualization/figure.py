import pandas as pd
import plotly.express as px


def figure(df: pd.DataFrame, log_x: bool=False, log_y: bool=False):
    fig = px.line(df, x="x", y="y", facet_row="row", facet_col="column", color="Method", log_x=log_x, log_y=log_y)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    return fig