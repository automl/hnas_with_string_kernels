import math
import pathlib

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Sequence

import utils


################################

METRICS = {
    "spearman": "Rank correlation",
    "pearson": "Pearson correlation",
    "kendalltau": r"Kendall's $\tau$",
    "nll": "-log marginal likelihood",
    "rmse": "RMSE",
    "mae": "MAE",
    "fitting_time": "Fitting time (s)",
    "predicting_time": "Predicting time (s)",
    "total_time": "Total time (s)",
}


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mae(predictions, targets):
    return np.absolute(predictions - targets).mean()


################################


def surrogate_exp_nb201(
    *,
    models: Sequence[utils.Model],
    datasets: Sequence[utils.Dataset],
    metric,
    n_train_ticks: Sequence[int],
    seeds: Sequence[int],
    data_base_path: pathlib.Path,
    save_pdf=True,
):
    if metric not in METRICS.keys():
        raise ValueError(f"Metric {metric} is not supported.")

    if save_pdf:
        utils.set_general_plot_style(presentation=False)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(datasets),
        sharex="row",
        figsize=(utils.get_size(utils.WIDTH_PT)[0], 1.5),
    )

    for didx, dataset in enumerate(datasets):
        ax = axs[didx]

        ax.set_facecolor("white")
        ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", pad=2, direction="out")

        for sidx, surrogate_model in enumerate(models):
            vals = []

            for nt in n_train_ticks:
                data = surrogate_model.read_regression_results(
                    data_base_path=data_base_path,
                    dataset_code=dataset.code,
                    n_train_tick=nt,
                )
                if data is None:
                    continue

                present_seeds = set(data.keys())
                missing_seeds = set(map(str, seeds)).difference(present_seeds)
                if missing_seeds:
                    print(
                        f"Warning missing seeds for {surrogate_model.code} for {nt} training points: {missing_seeds}"
                    )

                if metric == "rmse" or metric == "mae":
                    predicted = np.array(
                        [data[str(seed)]["y_pred"] for seed in present_seeds]
                    )
                    true = np.array(
                        [data[str(seed)]["y_test"] for seed in present_seeds]
                    )
                    metric_func = rmse if metric == "rmse" else mae
                    vals.append([metric_func(p, t) for p, t in zip(predicted, true)])
                else:
                    vals.append([data[str(seed)][metric] for seed in present_seeds])

            y_mean = np.mean(vals, axis=1)
            std_error = stats.sem(vals, axis=1)
            std_dev = np.std(vals, axis=1, ddof=1)
            ci_95 = (
                y_mean - (1.96 * std_dev / math.sqrt(len(vals))),
                y_mean + (1.96 * std_dev / math.sqrt(len(vals))),
            )
            one_std_err_range = (
                y_mean - std_error,
                y_mean + std_error,
            )
            assert len(vals) == len(y_mean)

            label = surrogate_model.name
            ax.plot(
                n_train_ticks[: len(y_mean)],
                y_mean,
                label=label,
                linestyle=utils.LINESTYLE[sidx],
            )
            ax.fill_between(
                n_train_ticks[: len(y_mean)],
                one_std_err_range[0],
                one_std_err_range[1],
                alpha=0.4,
            )

            ax.set_title(dataset.name, fontsize=utils.TITLE_FONTSIZE)
            ax.set_xlabel("\\#train samples", fontsize=utils.LABEL_FONTSIZE)
            if didx == 0:
                label = METRICS[metric]
                ax.set_ylabel(label, fontsize=utils.LABEL_FONTSIZE)

            ax.set_xlim([0, 400])
            ax.set_xticks([0, 200, 400])

            ax.set_ylim(0.1, 0.8)
            ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

            # comment the lines below when plotting NASK variants (so that ylimits are better)
            if metric == "spearman":
                if "cifar100" in dataset.code:
                    ax.set_ylim(0.5, 0.95)
                    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
                elif "cifar10" in dataset.code:
                    ax.set_ylim(0.5, 0.95)
                    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
                elif "addNIST" in dataset.code:
                    ax.set_ylim(0.4, 0.95)
                    ax.set_yticks([0.4, 0.6, 0.8])
                elif "cifarTile" in dataset.code:
                    ax.set_ylim(0.4, 0.95)
                    ax.set_yticks([0.4, 0.6, 0.8])
                elif "ImageNet" in dataset.code:
                    ax.set_ylim(0.5, 0.95)
                    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
            elif metric == "pearson":
                if "cifar100" in dataset.code:
                    ax.set_ylim(0.35, 0.75)
                    ax.set_yticks([0.4, 0.5, 0.6, 0.7])
                elif "cifar10" in dataset.code:
                    ax.set_ylim(0.37, 0.95)
                    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                elif "addNIST" in dataset.code:
                    ax.set_ylim(0.35, 0.95)
                    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
                elif "cifarTile" in dataset.code:
                    ax.set_ylim(0.3, 0.95)
                    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                elif "ImageNet" in dataset.code:
                    ax.set_ylim(0.4, 0.95)
                    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            elif metric == "kendalltau":
                if "cifar100" in dataset.code:
                    ax.set_ylim(0.35, 0.72)
                    ax.set_yticks([0.4, 0.5, 0.6, 0.7])
                elif "cifar10" in dataset.code:
                    ax.set_ylim(0.35, 0.82)
                    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
                elif "addNIST" in dataset.code:
                    ax.set_ylim(0.3, 0.75)
                    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
                elif "cifarTile" in dataset.code:
                    ax.set_ylim(0.3, 0.75)
                    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
                elif "ImageNet" in dataset.code:
                    ax.set_ylim(0.4, 0.82)
                    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
            elif metric == "rmse":
                if "cifar100" in dataset.code:
                    pass
                elif "cifar10" in dataset.code:
                    pass
                elif "addNIST" in dataset.code:
                    pass
                elif "cifarTile" in dataset.code:
                    pass
                elif "ImageNet" in dataset.code:
                    pass
            elif metric == "mae":
                if "cifar100" in dataset.code:
                    pass
                elif "cifar10" in dataset.code:
                    pass
                elif "addNIST" in dataset.code:
                    pass
                elif "cifarTile" in dataset.code:
                    pass
                elif "ImageNet" in dataset.code:
                    pass
            elif metric == "nll":
                pass
            elif "time" in metric:
                pass
            else:
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])

    sns.despine(fig)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    if save_pdf:
        plot_file_name_prefix = "_".join(m.name for m in models)
        utils.save_fig(
            fig, f"surrogate__{plot_file_name_prefix}__{metric}", output_dir="figures"
        )
    else:
        plt.show()
    plt.close()
