import dataclasses
import json
import pathlib
import yaml

import seaborn as sns
import matplotlib.pyplot as plt
from path import Path
from cycler import cycler

plt.style.use("seaborn")

WIDTH_PT = 397.48499

LEGEND_FONTSIZE = 10
TICK_FONTSIZE = 7
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 10

line_cycler = cycler(
    color=["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
) + cycler(linestyle=["-", "--", ":", "-.", (0, (1, 1)), (0, (5, 1))])
marker_cycler = (
        cycler(
            color=[
                "#0072B2",
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#D55E00",
                "#CC79A7",
                "#F0E442",
            ]
        )
        + cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"])
        + cycler(marker=["4", "2", "3", "1", "+", "x", "."])
)

LINESTYLE = ["-", "--", ":", "-.", (0, (1, 1)), (0, (5, 1))]


def set_general_plot_style(presentation: bool = False):
    """summary"""
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    plt.switch_backend("pgf")
    if presentation:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
                "font.family": "serif",
                "font.serif": [],
                "font.sans-serif": [],
                "font.monospace": [],
                "font.size": "10",
                "legend.fontsize": "9.90",
                "xtick.labelsize": "small",
                "ytick.labelsize": "small",
                "legend.title_fontsize": "small",
                "pgf.preamble": r"""
                    \usepackage[T1]{fontenc}
                    \usepackage[utf8x]{inputenc}
                    \usepackage{microtype}
                    \usepackage{mathptmx}
                """,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.sans-serif": [],
                # "font.sans-serif": ["Times New Roman"],
                "font.monospace": [],
                "font.size": max(
                    LEGEND_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE
                ),
                "legend.fontsize": LEGEND_FONTSIZE,
                "xtick.labelsize": TICK_FONTSIZE - 1,
                "ytick.labelsize": TICK_FONTSIZE - 1,
                "legend.title_fontsize": LEGEND_FONTSIZE,
                "pgf.preamble": r"""
                    \usepackage[utf8]{inputenc}
                    \usepackage[T1]{fontenc}
                    \usepackage{microtype}
                    \usepackage{nicefrac}
                    \usepackage{amsfonts}
                """,
            }
        )


def get_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def save_fig(fig, filename, output_dir, dpi: int = 600):
    output_dir = Path(output_dir)
    output_dir.makedirs_p()
    filename = filename.replace(" ", "_").replace("/", "")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight", dpi=dpi)
    print(f'Saved to "{output_dir}/{filename}.pdf"')


################################


@dataclasses.dataclass
class Model:
    experiment_name: str
    code: str
    name: str

    def read_regression_results(
            self,
            data_base_path: pathlib.Path,
            dataset_code: str,
            n_train_tick: int,
    ):
        json_path = (
                data_base_path
                / self.experiment_name
                / f"{dataset_code}__{self.code}"
                / f"surrogate_{self.code}_{n_train_tick}.json"
        )
        if not json_path.is_file():
            print(
                f"Warning: missing files for {self.code} with {n_train_tick} training samples"
            )
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def read_search_results(
            self,
            data_base_path: pathlib.Path,
            dataset_code: str,
            seed: int,
            num_configs: int,
    ):
        configs_path = (
                data_base_path
                / self.experiment_name
                / dataset_code
                / f"{self.code}__evolution__200"
                / str(seed)
                / "results"
        )
        if not configs_path.is_dir():
            raise ValueError(f"Error: missing configs dir for {self.code}")

        configs = {
            int(str(d).split("config_", maxsplit=1)[1]): (d / "result.yaml")
            for d in configs_path.iterdir()
            if d.is_dir() and "config_" in str(d)
        }
        configs = {k: v for k, v in configs.items() if k in range(1, num_configs + 1)}

        data = {}
        for k in range(1, num_configs + 1):
            v = configs.get(k, pathlib.Path("placeholder"))
            if v.is_file():
                with open(v, "r") as f:
                    data[k] = yaml.load(f, yaml.Loader)
            else:
                data[k] = "error"

        result = {}
        for k in sorted(list(data.keys())):
            result[k] = data[k]

        return result


@dataclasses.dataclass
class Dataset:
    code: str
    name: str
