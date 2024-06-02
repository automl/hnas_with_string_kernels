import pathlib
from typing import Sequence

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import utils


################################


def search_exp_nb201(
    *,
    models: Sequence[utils.Model],
    datasets: Sequence[utils.Dataset],
    metric,
    seeds: Sequence[int],
    num_configs,
    data_base_path: pathlib.Path,
    save_pdf=True,
):
    if save_pdf:
        utils.set_general_plot_style(presentation=False)

    for didx, dataset in enumerate(datasets):
        for seed in seeds:
            results = {}

            for sidx, surrogate_model in enumerate(models):
                data = surrogate_model.read_search_results(
                    data_base_path=data_base_path,
                    dataset_code=dataset.code,
                    seed=seed,
                    num_configs=num_configs,
                )
                if data is None:
                    continue

                metric_data = {}
                for k, v in data.items():
                    if v == "error":
                        metric_data[k] = float("inf")
                        print("PLOT SEARCH: INF: ", dataset.name, surrogate_model.name, seed, k)
                    else:
                        metric_data[k] = v[metric]
                results[surrogate_model.name] = metric_data

            for k, v in results.items():
                a = {iteration: vals for iteration, vals in v.items() if vals != float("inf")}
                x = np.array(sorted(list(a.keys())))
                y = np.array([a[k] for k in x])
                plt.title(f"{dataset.name} - {k} - {seed}")
                plt.scatter(x, y)
                plt.show()


def search_exp_act(
    *,
    models: Sequence[utils.Model],
    datasets: Sequence[utils.Dataset],
    metric,
    seeds: Sequence[int],
    num_configs,
    data_base_path: pathlib.Path,
    save_pdf=True,
):
    if save_pdf:
        utils.set_general_plot_style(presentation=False)

    for didx, dataset in enumerate(datasets):
        for seed in seeds:
            results = {}

            for sidx, surrogate_model in enumerate(models):
                data = surrogate_model.read_search_results(
                    data_base_path=data_base_path,
                    dataset_code=dataset.code,
                    seed=seed,
                    num_configs=num_configs,
                )
                if data is None:
                    continue

                metric_data = {}
                for k, v in data.items():
                    if v == "error":
                        metric_data[k] = float("inf")
                        print("PLOT SEARCH: INF: ", dataset.name, surrogate_model.name, seed, k)
                    else:
                        metric_data[k] = v[metric]
                results[surrogate_model.name] = metric_data

            for k, v in results.items():
                # show all results
                # a = {iteration: vals for iteration, vals in v.items() if vals != float("inf")}
                # show only better results
                a = {iteration: vals for iteration, vals in v.items() if vals != float("inf") and -3 <= vals <= -2.3}
                x = np.array(sorted(list(a.keys())))
                y = np.array([a[k] for k in x])
                plt.title(f"{dataset.name} - {k} - {seed}")
                plt.ylim(-2.7, -2.3)
                plt.scatter(x, y)
                plt.show()
