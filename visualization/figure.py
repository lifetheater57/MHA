import plotly.express as px
import pandas as pd

def figure(df: pd.DataFrame, params: dict=None):
    fig = px.line(df, x="x", y="y", facet_row="row", facet_col="column", color="Method")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    return fig