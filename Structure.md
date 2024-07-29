# Hassaku
Folder structure
```
.
├── algorithms
├── conf
├── data
├── eval
├── train
├── utilities
├── README.md
├── Structure.md
├── experiment_helper.py
├── run_experiment.py
├── run_agent.py
├── sweep_agent.py
├── hassaku.yml
├── wandb_api_key
└── wandb_conf.py
```

### ```algorithms```
Hosts the code of the implemented algorithms. Roughly divided into classes.

```
.
├── algorithms_utils.py
├── base_classes.py
├── graph_algs.py
├── knn_algs.py
├── linear_algs.py
├── mf_algs.py
├── naive_algs.py
└── sgd_alg.py
```

### ```conf```
Directory to host the configuration for the experiments. 
```
.
├── <here you can place your .yml files>
└── conf_parser.py
```

### ```data```
Directory to host raw dataset, processed dataset, dataloaders, dataset classes, and dataset processing code. 
```
.
├── amazonvid2018
│   ├── processed_dataset
│   ├── raw_dataset
│   └── amazonvid2018_processor.py
├── lfm2b2020
│   ├── processed_dataset
│   ├── raw_dataset
│   └── lfm2b2020_processor.py
├── ml10m
│   ├── processed_dataset
│   ├── raw_dataset
│   └── movielens10m_processor.py
├── ml1m
│   ├── processed_dataset
│   ├── raw_dataset
│   └── movielens1m_processor.py
├── dataloader.py
├── dataset.py
└── data_utils.py

```

### ```eval```
Directory to host the evaluation metrics and evaluation procedure 
```
.
├── eval.py
├── eval_utils.py
└── metrics.py

```

### ```train```
Directory to host the code to train a SGD-based recommendere system
```
.
├── rec_losses.py
├── trainer.py
└── utils.py
```

### ```utilities```
Directory to host the miscellaneous code.
```
.
├── similarities.py
└── utils.py
```