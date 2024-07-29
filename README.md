# SiBraR - A Multi-Modal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios

This repository accompanies our corresponding RecSys2024 submission
`A Multi-Modal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios`.

Originally, this repository is a fork of the [Hassaku](https://github.com/karapostK/hassaku) framework, extended and refactored to 
support the training and evaluation of content-based recommender systems for cold-start scenarios.

## Before you start
Please note that we use different names for the algorithms mentioned and experiemented with in the paper, than we use here in the code. While different, they still function the same way as described in their respective publications. Thus, here is a list of algorithm names and their shorthands:

| Algorithm name (paper) | Algorithm shorthand (code) | Algorithm class (code) | Source file |
|----------------|-------------------------------------------------------|--------------------------------|--------------------------|
| $SiBraR$       | SBNet                                                 | SingleBranchNet                | [algorithms/sgd_alg.py](algorithms/sgd_alg.py)    |
| $CLCRec$       | (depending on whether we use user or item side-information) |                                |                          |
|                | IFMF                                                  | ItemFeatureMatrixFactorization | [algorithms/sgd_alg.py](algorithms/sgd_alg.py) |
|                | UFMF                                                  | UserFeatureMatrixFactorization | [algorithms/sgd_alg.py](algorithms/sgd_alg.py) |
| $DropoutNet$   | DropoutNet                                            | DropoutNet                     | [algorithms/sgd_alg.py](algorithms/sgd_alg.py) |
| $MF$           | MF                                                    | SGDMatrixFactorization         | [algorithms/sgd_alg.py](algorithms/sgd_alg.py) |
| $DeepMF$       | DMF                                                   | DeepMatrixFactorization        | [algorithms/sgd_alg.py](algorithms/sgd_alg.py) |
| $Pop$          | Pop                                                   | PopularItems                   | [algorithms/naive_algs.py](algorithms/naive_algs.py) |
| $Rand$         | Rand                                                  | RandomItems                    | [algorithms/naive_algs.py](algorithms/naive_algs.py) |

## Installation
1) Clone the repo `git clone <this-repo-url>`
2) Update your conda installation
   1) `conda install python=3.10`
   2) `conda update conda`
   3) `conda config --set solver libmamba`
3) Install the environment with all its requirements `conda env create --file=environment.yml`
4) Activate the environment `conda activate hassaku`

## Using the framework
The following commands assume that the conda environment is already activated (`conda activate hassaku`).

### Retrieving the datasets
This framework supports 3 different, publicly available datasets (`Onion18, ML-1M and AmazonVideo2024`). 
For more information, we wish to refer you to our paper. For each dataset, there is an individual 
directory `data/<dataset-name>`, which contains all config files for this dataset. 
Moreover, it contains a script `<dataset-name>_preprocessor.py` or `<dataset-name>_downloader` to download 
the dataset.

For the `onion18` dataset, you have to execute the following
```
python data/onion/onion1mon_downloader.py \
--zenodo_access_token <your-zenodo-access-token> \ 
--config_file "data/onion/download_config.yaml" \
--save_path <your-data-storage-location> \
--year 2018
```

### Preprocess the datasets
Once a dataset is downloaded, you can start its preprocession. Check out 
[data/preprocess_dataset.py](data/preprocess_dataset.py) for more information:
```
usage: preprocess_dataset.py [-h] --config_file CONFIG_FILE [--data_path DATA_PATH] [--split_path SPLIT_PATH]

options:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE, -c CONFIG_FILE
                        .yaml configuration file defining the preprocessing
  --data_path DATA_PATH, -d DATA_PATH
                        The path where the data is stored
  --split_path SPLIT_PATH, -s SPLIT_PATH
                        The path where to store the split data to. If not specified, it will default to
                        {data_path}/{split_config}
```

### Run an experiment
For sweeping, you need to have a [Weights and Biases](https://wandb.ai/) account. 
For running a single experiment, simply select one of the configs provided 
in [conf/single/algorithms](conf/single/algorithms) or create your own config file and run it:

```
usage: run_experiment.py [-h]
                         [--algorithm {uknn,iknn,ifknn,mf,ifeatmf,sgdbias,pop,rand,rbmf,uprotomf,iprotomf,uiprotomf,acf,svd,als,p3alpha,ease,slim,uprotomfs,iprotomfs,uiprotomfs,ecf,dmf,dropoutnet,sbnet,ufeatmf}]
                         [--dataset {ml100k,ml1m,ml10m,amazonvid2018,lfm2b2020,deliveryherosg,onion,onion18,onion18g,kuai,amazonvid2024}]
                         [--dataset_path DATASET_PATH]
                         [--split_type {random,temporal,cold_start_user,cold_start_item,cold_start_both}]
                         [--conf_path CONF_PATH] [--run_type {train_val,test,train_val_test,gather}] [--log LOG]

Start an experiment

options:
  -h, --help            show this help message and exit
  --algorithm {uknn,iknn,ifknn,mf,ifeatmf,sgdbias,pop,rand,rbmf,uprotomf,iprotomf,uiprotomf,acf,svd,als,p3alpha,ease,slim,uprotomfs,iprotomfs,uiprotomfs,ecf,dmf,dropoutnet,sbnet,ufeatmf}, -a {uknn,iknn,ifknn,mf,ifeatmf,sgdbias,pop,rand,rbmf,uprotomf,iprotomf,uiprotomf,acf,svd,als,p3alpha,ease,slim,uprotomfs,iprotomfs,uiprotomfs,ecf,dmf,dropoutnet,sbnet,ufeatmf}
                        Recommender Systems Algorithm
  --dataset {ml100k,ml1m,ml10m,amazonvid2018,lfm2b2020,deliveryherosg,onion,onion18,onion18g,kuai,amazonvid2024}, -d {ml100k,ml1m,ml10m,amazonvid2018,lfm2b2020,deliveryherosg,onion,onion18,onion18g,kuai,amazonvid2024}
                        Recommender Systems Dataset
  --dataset_path DATASET_PATH, -p DATASET_PATH
                        The path to the dataset in case it is not located in the regular directory. All required data
                        must be placed directly in the root of this directory.
  --split_type {random,temporal,cold_start_user,cold_start_item,cold_start_both}, -s {random,temporal,cold_start_user,cold_start_item,cold_start_both}
                        Which dataset split to use
  --conf_path CONF_PATH, -c CONF_PATH
                        Path to the .yml containing the configuration
  --run_type {train_val,test,train_val_test,gather}, -t {train_val,test,train_val_test,gather}
                        Type of experiment to carry out
  --log LOG
```
Note that while other datasets are also visible, they are not yet supported due to the extensive changes to the framework.  

### Run a sweep
For sweeping, you need to have a [Weights and Biases](https://wandb.ai/) account. You can then run any of
the provided config files* in the [conf/sweeps](conf/sweeps) directory:
1) Start the sweep: `wandb sweep <your-sweep-config>`
2) Start the sweep agent(s):
   1) `wandb agent {sweep_id from the previous step}`
   2) or to run multiple agents in parallel, see [`run_agent.py`](run_agent.py):
```
usage: run_agent.py [-h] [--sweep_id SWEEP_ID] [--gpus GPUS] [--n_parallel N_PARALLEL]

Start an experiment

options:
  -h, --help            show this help message and exit
  --sweep_id SWEEP_ID, -s SWEEP_ID
                        The W&B sweep id used to start the agents.
  --gpus GPUS, -g GPUS  Which GPUs to use for running agents on. This will internally set the CUDA_VISIBLE_DEVICES
                        environmentvariable.
  --n_parallel N_PARALLEL, -p N_PARALLEL
                        The number of agents to run in parallel on each GPU
```


\***Although there are lots of configs defined, due to inplace modification of them, they are by far not an 
exhaustive list of the experiments that we performed!**
