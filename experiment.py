import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.io as pio
from rich.progress import track
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis

from data.gaussian_generator import GaussianGenerator
from model.model import Connectivity, ConnectivityEM
from visualization.config import get_at
from visualization.figure import figure

# Constants
W_title = r"$W \text{ recovery}$"
G_i_title = r"$G^{(i)}\text{ recovery}$"
NLL_title = r"$\text{Negative log-likelihood}$"

MHA_label = "MHA SM"
FA_label = "Factor Anal."
NN_PCA_label = "Non-neg. PCA"
SC_label = "Sample Cov."
LW_label = "Ledoit-Wolf"
G_label = "Glasso"
DC_label = "Diag. Cov."
EM_label = "MHA EM"

lg10_label = r"$log_{10}(n)$"
lg10_sq_err_label = r"$log_{10}(\text{Sq. Error})$"
rel_NLL_label = r"$\text{Relative NLL}$"

# Data config
N = [1, 10]
p = 50
k = 5
seed = 420
sizes = [100, 200, 1000, 2000, 4000]
split_ratio = 0.5
fixed_test_size = True
repetition = 10

# Figure config
figure_config = {
    "rows": {
        "title": [f"$\\text{{{N[i]} class{'es' if N[i] > 1 else ''}}}$" for i in range(len(N))],
    },
    "columns": {
        "title": [W_title, NLL_title],
        "x": lg10_label,
        "y": [lg10_sq_err_label, lg10_sq_err_label, rel_NLL_label],
        "y_confidence": ["range"] * 3,
        "Method": [[MHA_label, EM_label, FA_label]] * 3,
    },
}

def find_assignment(W, W_true):
    rows, cols = W.shape
    C = np.zeros((cols,cols))
    for i in range(cols):
        for j in range(cols):
            C[i,j] = np.linalg.norm(W[:,i] - W_true[:,j]) 

    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind, col_ind

print("Running experiments...")
df = pd.DataFrame()
for i in range(len(N)):
    print(f"\tUsing {N[i]} classes.")
    for j in track(range(len(sizes)), "Running"):
        # Generate data
        generator = GaussianGenerator(N[i], p, k, seed, np.round(sizes[j] / N[i]).astype(int))
        df_iteration = pd.DataFrame()
        for iteration in range(repetition):
            data = next(generator)
            # Split data
            if not fixed_test_size:
                train_size = np.round(split_ratio * sizes[j] / N[i]).astype(int)
            else:
                train_size = np.round((sizes[j] - (1 - split_ratio) * min(sizes)) / N[i]).astype(int)
            data_test = data[:, train_size:]
            data_train = data[:, :train_size]
            # Factor Analysis
            if FA_label in get_at(figure_config["columns"]["Method"], -1):
                # Fit the model
                init_params = {
                    "n_components": k,
                    "rotation": "varimax",
                }
                fit_params = {
                    "X": np.reshape(data_train, (np.prod(data_train.shape[:2]), data_train.shape[2])),
                }
                model = FactorAnalysis(**init_params)
                model.fit(**fit_params)

                measures = {
                    "iteration": iteration,
                    "Method": FA_label,
                    }
                for c, title in enumerate(figure_config["columns"]["title"]):
                    # Compute the metric
                    w = model.components_.T
                    if title == W_title:
                        measures[W_title] = np.log10(np.mean((w - generator.W)**2))
                    elif title == G_i_title:
                        #TODO: Compute the value here
                        measures[G_i_title] = 0#np.log10(np.sum((model.get_covariance() - generator.G)**2))
                    elif title == NLL_title:
                        measures[NLL_title] = -model.score(np.reshape(data_test, (np.prod(data_test.shape[:2]), data_train.shape[2])))
                df_iteration = pd.concat([df_iteration, pd.DataFrame([measures])])
            
            # EM
            if EM_label in get_at(figure_config["columns"]["Method"], -1):
                # Fit the model
                init_params = {
                    "X": data_train,
                    "k": k,
                }
                fit_params = {}
                model = ConnectivityEM(**init_params)
                model.fit(**fit_params)

                measures = {
                    "iteration": iteration,
                    "Method": EM_label,
                    }
                for c, title in enumerate(figure_config["columns"]["title"]):
                    # Compute the metric
                    if title == W_title:
                        W_aligned = model.W[:, find_assignment(model.W, generator.W)[1]]
                        measures[W_title] = np.log10(np.mean((W_aligned - generator.W)**2))
                    elif title == G_i_title:
                        G_aligned = model.G[:, find_assignment(model.W, generator.W)[1]]
                        measures[G_i_title] = np.log10(np.sum((G_aligned - generator.G)**2))
                    elif title == NLL_title:
                        measures[NLL_title] = model.negative_log_likelihood(data_test)# / (min(sizes) - train_size)
                df_iteration = pd.concat([df_iteration, pd.DataFrame([measures])])
            
            ## MHA
            if MHA_label in get_at(figure_config["columns"]["Method"], -1):
                # Fit the model
                init_params = {
                    "X": data_train,
                    "k": k,
                }
                fit_params = {}
                model = Connectivity(**init_params)
                model.fit(**fit_params)

                measures = {
                    "iteration": iteration,
                    "Method": MHA_label,
                    }
                for c, title in enumerate(figure_config["columns"]["title"]):
                    # Compute the metric
                    if title == W_title:
                        W_aligned = model.W[:, find_assignment(model.W, generator.W)[1]]
                        #TODO: implement log-sum-exp
                        measures[W_title] = np.log10(np.mean((W_aligned - generator.W)**2))
                    elif title == G_i_title:
                        #TODO: implement log-sum-exp
                        G_aligned = model.G[:, find_assignment(model.W, generator.W)[1]]
                        measures[G_i_title] = np.log10(np.sum((G_aligned - generator.G)**2))
                    elif title == NLL_title:
                        measures[NLL_title] = model.negative_log_likelihood(data_test)# / (min(sizes) - train_size)
                df_iteration = pd.concat([df_iteration, pd.DataFrame([measures])])
        
        for method in df_iteration["Method"].unique():
            for c, title in enumerate(figure_config["columns"]["title"]):
                # Save the metric
                mean = None
                var = None
                method_mask = (df_iteration["Method"] == method)
                if title == W_title:
                    mean = df_iteration.loc[method_mask, W_title].mean()
                    #TODO: check if this is the right confidence measure for 95% CI
                    var = 1.96 * np.sqrt(df_iteration.loc[method_mask, W_title].var() / method_mask.sum())
                elif title == G_i_title:
                    mean = df_iteration.loc[method_mask, G_i_title].mean()
                    #TODO: check if this is the right confidence measure for 95% CI
                    var = 1.96 * np.sqrt(df_iteration.loc[method_mask, G_i_title].var() / method_mask.sum())
                elif title == NLL_title:
                    mean = df_iteration.loc[method_mask, NLL_title].mean()
                    #TODO: check if this is the right confidence measure for 95% CI
                    var = 1.96 * np.sqrt(df_iteration.loc[method_mask, NLL_title].var() / method_mask.sum())
                row = pd.DataFrame([{
                    get_at(figure_config["columns"]["x"], c): np.log10(sizes[j]),
                    get_at(figure_config["columns"]["y"], c): mean,
                    get_at(figure_config["columns"]["y_confidence"], c): var,
                    "row": get_at(figure_config["rows"]["title"], i),
                    "column": get_at(figure_config["columns"]["title"], c),
                    "Method": method,
                }])
                df = pd.concat([df, row])
                
if not os.path.exists("output"):
    os.makedirs("output")
df.to_csv("output/data.csv")
with open("output/config.json", "w+") as file:
    file.write(json.dumps(figure_config))
fig = figure(df, figure_config)
filename = f"output/plot-" + datetime.now().strftime("%Y%m%d%H%M%S")
pio.write_html(fig, filename + ".html")
pio.write_image(fig, filename + ".png")
