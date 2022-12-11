import numpy as np
import pandas as pd
from datetime import datetime

from data.gaussian_generator import GaussianGenerator
from visualization.figure import figure
from model.model import Connectivity
from rich.progress import track

from plotly import io

N = [1]#, 2]
p = 50
k = 5
seed = 6269
sizes = [100, 200, 1000, 2000, 4000]
split_ratio = 0.9

df = pd.DataFrame(columns=["x", "y", "row", "column", "Method"])
print("Running experiments...")
for i in range(len(N)):
    print(f"\tUsing {N[i]} classes.")
    for j in track(range(len(sizes)), "Running"):
        data = next(GaussianGenerator(N[i], p, k, seed, np.round(sizes[j] / N[i]).astype(int)))

        train_size = np.round(split_ratio * sizes[j] / N[i]).astype(int)
        data_train = data[:, :train_size]
        data_test = data[:, train_size:]

        init_params = {
            "X": data_train,
            "k": k,
        }
        fit_params = {}
        model = Connectivity(**init_params)
        model.fit(**fit_params)
        nll = model.negative_log_likelihood(data_test)

        df_temp = pd.Series({
                "x": sizes[j],
                "y": nll,
                "row": f"{i} class{'es' if i > 1 else ''}",
                "column": "Relative NLL",
                "Method": j,
            })
        df = pd.concat([df, df_temp])
fig = figure(df)
filename = f"plot-" + datetime.now().strftime("%Y%m%d%H%M%S")
io.write_html(fig, filename + ".html")
io.write_image(fig, filename + ".png")
