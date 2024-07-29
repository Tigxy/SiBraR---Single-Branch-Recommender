import os
import re
import numpy as np
import pandas as pd

import wikipedia
from sentence_transformers import SentenceTransformer

from data.config_classes import DatasetsEnum
from data_paths import get_processed_dataset_path
from utilities.notebook_utils import batch_samples, store_embeddings


def get_wikipedia_page_name(raw_name):
    names = wikipedia.search(raw_name)
    if len(names) == 0:
        return ''
    else:
        return names[0]


def get_movie_plot(page_name):
    try:
        try:
            movie_page_content = str(wikipedia.page(page_name, auto_suggest=False).content)
        except wikipedia.DisambiguationError as e:
            for option in e.options:
                if 'film' in option:
                    movie_page_content = str(wikipedia.page(option, auto_suggest=False).content)
            return ''
    except (wikipedia.PageError, KeyError):
        return ''
    re_groups = re.search('Plot ==(.*?)=+ [A-Z]', str(movie_page_content).replace('\n', ''))
    if re_groups:
        return re_groups.group(1)
    else:
        return ''


def extract_title_and_year(row):
    t = row['title']
    pattern = r'(.*)[(](\d+)[)]'
    m = re.search(pattern, t)
    return m[1].strip(), int(m[2])


def process_ml_items(df_movies: pd.DataFrame):
    df_movies[['title_wo_year', 'year']] = df_movies.apply(extract_title_and_year, axis=1, result_type='expand')
    df_movies['genres'] = df_movies['genres'].apply(lambda s: s.split('|'))
    return df_movies


def get_ml1m_items():
    data_path = get_processed_dataset_path(DatasetsEnum.ml1m)
    df_movies = pd.read_csv(os.path.join(data_path, 'item_features.csv'), index_col=False)
    return process_ml_items(df_movies)


def get_ml1m_users():
    data_path = get_processed_dataset_path(DatasetsEnum.ml1m)
    # ignore occupation and zip code, as they seem unusable
    df_users = pd.read_csv(os.path.join(data_path, 'user_features.csv'), index_col=False)
    df_users['gender'] = df_users['gender'].replace({'M': 'male', 'F': 'female'})
    return df_users


def enrich_df_by_plots(df_movies: pd.DataFrame, force_download=False):
    storage_file = os.path.join(get_processed_dataset_path(DatasetsEnum.ml1m), 'wikipedia_enriched_movies.csv')

    if os.path.isfile(storage_file) and not force_download:
        print('(loading cached plots...)')
        df = pd.read_csv(storage_file)
    else:
        df = df_movies.copy()

        print('retrieving wikipedia page names')
        df['wikipedia_page_name'] = df['title'].progress_apply(lambda name: get_wikipedia_page_name(name))

        print('retrieving plots')
        df['plot'] = df['wikipedia_page_name'].progress_apply(lambda name: get_movie_plot(name))

        # store csv
        df.to_csv(storage_file, index=False)

    # drop columns that don't matter
    df = df.drop(columns=['wikipedia_page_name'])

    # finally return enriched DataFrame
    return df


def store_ml1m_embeddings(entity: str, feature_name: str, indices: np.ndarray, values: np.ndarray):
    return store_embeddings(DatasetsEnum.ml1m, entity, feature_name, indices, values)


if __name__ == '__main__':
    movies = get_ml1m_items()
    movies = enrich_df_by_plots(movies, force_download=False)

    # load embedding network
    # note the maximum of 384 word pieces, which leads to truncated plots
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:0')

    # embed the plots
    plot_embeddings = []
    for plots in batch_samples(movies['plot'].tolist(), batch_size=64, desc='embedding plots'):
        plot_embeddings.append(model.encode(plots))
        break
    plot_embeddings = np.concatenate(plot_embeddings)

    # store results
    store_ml1m_embeddings('item', 'plot_mpnet', indices=movies['item'], values=plot_embeddings)
