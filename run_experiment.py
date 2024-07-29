import argparse
import logging

from algorithms.algorithms_utils import AlgorithmsEnum
from data.config_classes import DatasetSplitType, DatasetsEnum
from experiment_helper import run_train_val_test_experiment, run_train_val_experiment, run_test_experiment, \
    run_gather_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start an experiment')

    parser.add_argument('--algorithm', '-a', type=str, help='Recommender Systems Algorithm',
                        choices=[*AlgorithmsEnum])

    parser.add_argument('--dataset', '-d', type=str, help='Recommender Systems Dataset',
                        choices=[*DatasetsEnum], required=False, default='ml1m')

    parser.add_argument('--dataset_path', '-p', type=str,
                        help='The path to the dataset in case it is not located in the regular directory. '
                             'All required data must be placed directly in the root of this directory.',
                        required=False, default=None)

    parser.add_argument('--split_type', '-s', type=DatasetSplitType, help='Which dataset split to use',
                        choices=[*DatasetSplitType], required=False, default=DatasetSplitType.Random)

    parser.add_argument('--conf_path', '-c', type=str, help='Path to the .yml containing the configuration')

    parser.add_argument('--run_type', '-t', type=str,
                        choices=['train_val', 'test', 'train_val_test', 'gather'],
                        default='train_val_test', help='Type of experiment to carry out')

    parser.add_argument('--log', type=str, default='WARNING')

    args = parser.parse_args()

    alg = AlgorithmsEnum[args.algorithm]
    dataset = DatasetsEnum[args.dataset]
    dataset_path = args.dataset_path
    split_type = args.split_type
    conf_path = args.conf_path
    run_type = args.run_type
    log = args.log

    logging.basicConfig(level=log)
    match run_type:
        case 'train_val':
            run_train_val_experiment(alg, dataset, split_type, conf_path, dataset_path)
        case 'test':
            run_test_experiment(alg, dataset, split_type, conf_path)
        case 'gather':
            run_gather_experiment(alg, dataset, split_type, conf_path)
        case _:
            run_train_val_test_experiment(alg, dataset, split_type, conf_path, dataset_path)
