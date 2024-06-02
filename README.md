# Hierarchical NAS With String Kernels

This repository contains the code related to the experiments of our AutoML 2024 workshop paper "Beyond Graph-Based Modeling for Hierarchical Neural Architecture Search". 

For more information, please check out the paper on [OpenReview](https://openreview.net/forum?id=gze7ISazsz).

## Installation

### Prerequisites

- poetry (for installing python dependencies)
- conda (for managing the venv)

Create a directory `automl` which will be used as the root of the project.

### Set up an appropriate version of NEPS

Inside the created `automl` directory, run:
```text
git clone --branch skanas git@github.com:automl/neps.git
```
### Set up the experiments repository

Inside the created `automl` directory, run:
1. Clone this repository:
    ```text
    git clone git@github.com:automl/hnas_with_string_kernels.git
    ```
2. Set up the venv for `nask`:
    ```text
    conda create -y --name nask python=3.7 && \
    conda activate nask && \
    cd hnas_with_string_kernels && \
    poetry install && \
    conda deactivate && \
    cd ..
    ```

## The data needed for surrogate experiments and the paper results

Please extract the downloaded files into the `automl/hnas_with_string_kernels` directory.

The data needed for the surrogate experiments can be downloaded from [Google Drive](https://drive.google.com/file/d/1TTPgI14qsJAbKsfC9c-I4XkB9cbDSmfe/view?usp=sharing).
<br>
The data for the paper results can be downloaded from [Google Drive](https://drive.google.com/file/d/1YN3C_OlLz9TlmiPmAORlVVMvNp3nImV0/view).

## Running the experiments

### Surrogate experiments

The commands to run the hNASK surrogate experiments for all datasets as in the paper are given below.
<br>
Run any of these commands from the `automl/hnas_with_string_kernels` directory.

- For `hNASK`: 
    ```text
    conda activate nask
    
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifar10' surrogate_model='gp_string_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifar10/' surrogate_model.kernel_type='nask' surrogate_model.hierarchy_considered='[-7, -6, -5, -4, -3, -2, -1]' surrogate_model.fit_num_iterations='20' surrogate_model.optimize_kernel_weights='True' experiment_group='nask_all_hierarchies'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifar100' surrogate_model='gp_string_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifar100/' surrogate_model.kernel_type='nask' surrogate_model.hierarchy_considered='[-7, -6, -5, -4, -3, -2, -1]' surrogate_model.fit_num_iterations='20' surrogate_model.optimize_kernel_weights='True' experiment_group='nask_all_hierarchies'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_ImageNet16-120' surrogate_model='gp_string_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_ImageNet16-120/' surrogate_model.kernel_type='nask' surrogate_model.hierarchy_considered='[-7, -6, -5, -4, -3, -2, -1]' surrogate_model.fit_num_iterations='20' surrogate_model.optimize_kernel_weights='True' experiment_group='nask_all_hierarchies'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifarTile' surrogate_model='gp_string_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifarTile/' surrogate_model.kernel_type='nask' surrogate_model.hierarchy_considered='[-7, -6, -5, -4, -3, -2, -1]' surrogate_model.fit_num_iterations='20' surrogate_model.optimize_kernel_weights='True' experiment_group='nask_all_hierarchies'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_addNIST' surrogate_model='gp_string_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_addNIST/' surrogate_model.kernel_type='nask' surrogate_model.hierarchy_considered='[-7, -6, -5, -4, -3, -2, -1]' surrogate_model.fit_num_iterations='20' surrogate_model.optimize_kernel_weights='True' experiment_group='nask_all_hierarchies'
    ```
- For `hWL`:
    ```text
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifar10' surrogate_model='gp_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifar10/' experiment_group='hWL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifar100' surrogate_model='gp_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifar100/' experiment_group='hWL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_ImageNet16-120' surrogate_model='gp_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_ImageNet16-120/' experiment_group='hWL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifarTile' surrogate_model='gp_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifarTile/' experiment_group='hWL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_addNIST' surrogate_model='gp_hierarchical' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_addNIST/' experiment_group='hWL'
    ```
- For `WL`:
    ```text
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifar10' surrogate_model='gp' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifar10/' experiment_group='WL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifar100' surrogate_model='gp' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifar100/' experiment_group='WL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_ImageNet16-120' surrogate_model='gp' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_ImageNet16-120/' experiment_group='WL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_cifarTile' surrogate_model='gp' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_cifarTile/' experiment_group='WL'
    python -m skanas.scripts.run_surrogate_regression objective='nb201_addNIST' surrogate_model='gp' n_train_values='[10,25,50,75,100,150,200,300,400]' data_path='towards_nas_data/nb201_addNIST/' experiment_group='WL'
    ```

### Search experiments `Hierarchical NB201`

Use the code below to create a shell script for your case, which you will execute from the `automl/hnas_with_string_kernels` directory.

```text
conda activate nask

DATASET=$1  # one of {"nb201_cifar10", "nb201_cifar100", "nb201_ImageNet16-120", "nb201_cifarTile", "nb201_addNIST"} 
MODEL=$2  # one of {"gp_string_hierarchical", "gp_hierarchical", "gp"}
SEED=$3  # one of {777, 888, 999}
MAX_EVALUATIONS_PER_RUN="null"
DATA_PATH="search_input_data/${DATASET}/"

if [ "${DATASET}" != "nb201_cifar10" ] && [ "${DATASET}" != "nb201_cifar100" ] && [ "${DATASET}" != "nb201_ImageNet16-120" ] && [ "${DATASET}" != "nb201_cifarTile" ] && [ "${DATASET}" != "nb201_addNIST" ]; then
    >&2 echo "Invalid dataset argument."
    exit 1
fi

if [ "${MODEL}" = "gp_string_hierarchical" ]; then
    SURROGATE_MODEL_ARGS=(surrogate_model.kernel_type=nask surrogate_model.fit_num_iterations=20 surrogate_model.optimize_kernel_weights=True surrogate_model.hierarchy_considered='[-7,-6,-5,-4,-3,-2,-1]')
elif [ "${MODEL}" = "gp_hierarchical" ]; then
    SURROGATE_MODEL_ARGS=()
elif [ "${MODEL}" = "gp" ]; then
    SURROGATE_MODEL_ARGS=()
else
    >&2 echo "Invalid model argument."
    exit 1
fi

python -m skanas.scripts.run_search \
    objective="${DATASET}" \
    data_path="${DATA_PATH}" \
    n_init=10 \
    max_evaluations_total=100 \
    max_evaluations_per_run="${MAX_EVALUATIONS_PER_RUN}" \
    pool_size=200 \
    experiment_group="final" \
    seed="${SEED}" \
    surrogate_model="${MODEL}" \
    "${SURROGATE_MODEL_ARGS[@]}"
```

### Search experiment `Activation Search`

Use the code below to create a shell script for your case, which you will execute from the `automl/hnas_with_string_kernels` directory.

```text
conda activate nask

SEARCH_SPACE="act_cifar10"
DATASET="act_cifar10"
DATA_PATH="search_input_data/nb201_cifar10/"
MODEL=$1  # one of {"gp_string_hierarchical", "gp_hierarchical", "gp"}
SEED=$2  # one of 777, 888, 999
MAX_EVALUATIONS_PER_RUN="null"

if [ "${MODEL}" = "gp_string_hierarchical" ]; then
    SURROGATE_MODEL_ARGS=(surrogate_model.kernel_type=nask surrogate_model.fit_num_iterations=20 surrogate_model.optimize_kernel_weights=True surrogate_model.hierarchy_considered='[-3,-2,-1]')
elif [ "${MODEL}" = "gp_hierarchical" ]; then
    SURROGATE_MODEL_ARGS=()
elif [ "${MODEL}" = "gp" ]; then
    SURROGATE_MODEL_ARGS=()
else
    >&2 echo "Invalid model argument."
    exit 1
fi

python -m skanas.scripts.run_search \
    search_space="${SEARCH_SPACE}" \
    objective="${DATASET}" \
    data_path="${DATA_PATH}" \
    n_init=50 \
    max_evaluations_total=1000 \
    max_evaluations_per_run="${MAX_EVALUATIONS_PER_RUN}" \
    pool_size=200 \
    experiment_group="final" \
    seed="${SEED}" \
    surrogate_model="${MODEL}" \
    "${SURROGATE_MODEL_ARGS[@]}"
```

## The notebooks for the paper results

The notebooks for the paper results can be found at
```text
automl/hnas_with_string_kernels/src/skanas/plotting/search.ipynb
automl/hnas_with_string_kernels/src/skanas/plotting/surrogate_regression.ipynb
```
