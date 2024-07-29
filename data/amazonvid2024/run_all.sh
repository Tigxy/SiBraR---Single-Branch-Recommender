# execute from root dir
python data/preprocess_dataset.py --config_file data/amazonvid2024/split_config_random.yaml --data_path data/amazonvid2024/processed_dataset/
python data/preprocess_dataset.py --config_file data/amazonvid2024/split_config_coldstart_user.yaml --data_path data/amazonvid2024/processed_dataset/
python data/preprocess_dataset.py --config_file data/amazonvid2024/split_config_coldstart_item.yaml --data_path data/amazonvid2024/processed_dataset/
