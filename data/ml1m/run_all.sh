# execute from root dir
python data/preprocess_dataset.py --config_file data/ml1m/split_config_random.yaml --data_path ./../datasets/hassaku/ml-1m/processed_dataset/
python data/preprocess_dataset.py --config_file data/ml1m/split_config_temporal.yaml --data_path ./../datasets/hassaku/ml-1m/processed_dataset/
python data/preprocess_dataset.py --config_file data/ml1m/split_config_coldstart_user.yaml --data_path ./../datasets/hassaku/ml-1m/processed_dataset/
python data/preprocess_dataset.py --config_file data/ml1m/split_config_coldstart_item.yaml --data_path ./../datasets/hassaku/ml-1m/processed_dataset/
python data/preprocess_dataset.py --config_file data/ml1m/split_config_coldstart_both.yaml --data_path ./../datasets/hassaku/ml-1m/processed_dataset/
