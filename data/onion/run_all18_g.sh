# execute from root dir
python data/preprocess_dataset.py --config_file data/onion/split_config_random_g.yaml --data_path /share/hel/datasets/hassaku-sbrec/onion18/processed_dataset/ --split_path /share/hel/datasets/hassaku-sbrec/onion18g/processed_dataset/random_split
python data/preprocess_dataset.py --config_file data/onion/split_config_coldstart_user_g.yaml --data_path /share/hel/datasets/hassaku-sbrec/onion18/processed_dataset/ --split_path /share/hel/datasets/hassaku-sbrec/onion18g/processed_dataset/cold_start_user
python data/preprocess_dataset.py --config_file data/onion/split_config_coldstart_item_g.yaml --data_path /share/hel/datasets/hassaku-sbrec/onion18/processed_dataset/ --split_path /share/hel/datasets/hassaku-sbrec/onion18g/processed_dataset/cold_start_item
