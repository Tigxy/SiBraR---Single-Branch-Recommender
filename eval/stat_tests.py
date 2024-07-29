from scipy import stats
import itertools as it
import numpy as np
import pandas as pd

datasets = ['amazonvid2024', 'ml1m', 'onion18']
scenarios = ['cold_start_item', 'random', 'cold_start_user']
metrics = ['ndcg10', 'precision10', 'recall10']

files_names = [f'./data/results/{dataset}_{scenario}_{metric}.csv' for dataset in datasets for scenario in scenarios for metric in metrics]

#%%


threshold = 0.05

for file_name in files_names:
    metric_df = pd.read_csv(file_name)
    means_series = metric_df.mean()
    other_models = list(means_series.index)

    best_model = means_series.idxmax()
    other_models.remove(best_model) # this happens in place
    num_comparisons = len(other_models)
    bonferroni_threshold = threshold / num_comparisons


    best_model_metrics = user_metric_model_1 = metric_df[best_model]
    for other_model in other_models:
        other_model_metrics = metric_df[other_model]

        ttest = stats.ttest_rel(best_model_metrics, other_model_metrics, nan_policy='omit')

        if ttest.pvalue > bonferroni_threshold:
            print(f'\t{file_name}\n{best_model}\t{other_model}\t{ttest.pvalue} > {bonferroni_threshold}')
        else:
            # print('all good\n')
            pass
#%%