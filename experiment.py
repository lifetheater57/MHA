from datetime import datetime

import numpy as np
import pandas as pd
import plotly.io as pio
from rich.progress import track

from data.gaussian_generator import GaussianGenerator
from model.model import Connectivity
from visualization.figure import figure

# Constants
W_title = r"$W \text{ recovery}$"
G_i_title = r"$G^{(i)}\text{ recovery}$"
NLL_title = r"$\text{Negative log-likelihood}$"

MHA_label = "MHA"
FA_label = "Factor Anal."
NN_PCA_label = "Non-neg. PCA"
SC_label = "Sample Cov."
LW_label = "Ledoit-Wolf"
G_label = "Glasso"
DC_label = "Diag. Cov."

lg10_label = r"$log_{10}(n)$"
lg10_sq_err_label = r"$log_{10}(\text{Sq. Error})$"
rel_NLL_label = r"$\text{Relative NLL}$"

# Data config
N = [1, 2]
p = 50
k = 5
seed = 6269
sizes = [100, 200, 1000, 2000, 4000]
split_ratio = 0.9

# Figure config
figure_config = {
    "rows": {
        "count": len(N),
        "title": [f"$\\text{{{N[i]} class{'es' if N[i] > 1 else ''}}}$" for i in range(len(N))],
    },
    "columns": {
        "count": 3,
        "title": [W_title, G_i_title, NLL_title],
        "x": "log_10(n)",
        "y": ["log_10(Sq. Error)", "log_10(Sq. Error)", "Relative NLL"],
        method_column: [[MHA_label]] * 3,
    },
}

def get_at(param, index):
    """
    Function to ease the access to parameters.
    """
    if type(param) == str:
        return param
    elif type(param) == list:
        return param[index]
    else:
        raise Exception("Unsupported parameter type. Should be either str or list(str).")

print("Running experiments...")
df = pd.DataFrame()
for i in range(len(N)):
    print(f"\tUsing {N[i]} classes.")
    for j in track(range(len(sizes)), "Running"):
        # Generate data
        data = next(GaussianGenerator(N[i], p, k, seed, np.round(sizes[j] / N[i]).astype(int)))
        # Split data
        train_size = np.round(split_ratio * sizes[j] / N[i]).astype(int)
        data_train = data[:, :train_size]
        data_test = data[:, train_size:]
        ## MHA
        if MHA_label in get_at(figure_config["columns"][method_column], -1):
            # Fit the model
            init_params = {
                "X": data_train,
                "k": k,
            }
            fit_params = {}
            model = Connectivity(**init_params)
            model.fit(**fit_params)
            if NLL_title in figure_config["columns"]["title"]:
                # Compute and save the metrics
                nll = model.negative_log_likelihood(data_test)
                c = figure_config["columns"]["title"].index(NLL_title)
                row = pd.DataFrame([{
                    "x": sizes[j],
                    "y": nll,
                    "row": get_at(figure_config["rows"]["title"], i),
                    "column": "Negative log-likelihood",
                    "Method": "MHA",
                }])
                df = pd.concat([df, row])
fig = figure(df, figure_config)
filename = f"plot-" + datetime.now().strftime("%Y%m%d%H%M%S")
pio.write_html(fig, filename + ".html")
pio.write_image(fig, filename + ".png")