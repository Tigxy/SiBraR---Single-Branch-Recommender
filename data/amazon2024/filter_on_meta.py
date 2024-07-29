import json
import requests
from tqdm import tqdm
import os
import pickle
import csv


def filter_on_verified(save_path: str = './', category_official: str = 'Video_Games',
                       category_short: str = 'videogames'):
    category_short = f'{category_short}_verified'
    directory = os.path.join(save_path, f"raw_dataset/{category_short}")

    csv_path = os.path.join(directory, f"{category_official}.csv")
    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['user_id', 'parent_asin', 'rating', 'timestamp'])

        review_file = os.path.join(directory, f"{category_official}.jsonl")

        lines_verified = []
        total = 0
        verified = 0
        with open(review_file, 'r') as fp:
            for i, line in enumerate(tqdm(fp)):
                total += 1
                line = json.loads(line.strip())
                if line['verified_purchase']:
                    verified += 1
                    to_write = [line['user_id'], line['parent_asin'], line['rating'], line['timestamp']]
                    writer.writerow(to_write)
                    lines_verified.append(i)
        with open(f"{directory}/lines_verified", "wb") as fp:  # Pickling
            pickle.dump(lines_verified, fp)
        print(f'{verified / total} lines verified')


def filter_on_meta(save_path: str = './', category_official: str = 'All_Beauty', category_short: str = 'beauty',
                   crawl_images=False):
    directory = os.path.join(save_path, f"raw_dataset/{category_short}")
    meta_file = os.path.join(directory, f'meta_{category_official}.jsonl')
    ids_all_featuers = {}

    with open(meta_file, 'r') as fp:
        for line in fp:
            line = json.loads(line.strip())
            if len(line['images']) > 0:
                if line['images'][0]['large'] and line['title'] and line['description']:
                    ids_all_featuers[line['parent_asin']] = {
                        'image_url': line['images'][0]['large'],
                        'title': line['title'],
                        'description': line['description']
                    }
    if crawl_images:
        images_path = os.path.join(save_path, f'/raw_dataset/{category_short.removesuffix("_verified")}_images/')
        crawled_ids = [image_file.split('.')[0] for image_file in os.listdir(images_path)]
        missing_ids = []
        for item_id, features in tqdm(ids_all_featuers.items()):
            if item_id not in crawled_ids:
                image_url = features['image_url']
                try:
                    img_data = requests.get(image_url).content
                    with open(f'{images_path}/{item_id}.jpg', 'wb+') as handler:
                        handler.write(img_data)
                except:
                    missing_ids.append(item_id)
                    pass
    # with open(f'{directory}/missing_images.pkl', "wb") as fp:
    #    pickle.dump(missing_ids, fp)

    with open(f'{directory}/ids_all_featuers.json', 'w') as fp:
        json.dump(ids_all_featuers, fp)
        fp.close

    return ids_all_featuers.keys()


if __name__ == '__main__':
    filter_on_verified(category_official='Video_Games', category_short='videogames', )
