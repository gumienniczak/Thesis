from prepare_examples_exp1 import df, df_fin
from model_pipeline_exp1 import obtain_predictions
import pandas as pd

pd.set_option('display.max_colwidth', None)

nat_only = df_fin[df_fin['cat'] == 'NAT']

models = ['bert-base-uncased', 'google-bert/bert-large-uncased', 'google-bert/bert-base-cased',
          'google-bert/bert-large-cased', 'FacebookAI/roberta-base','FacebookAI/roberta-large',
          'distilbert/distilbert-base-uncased', 'distilbert/distilbert-base-cased', 'albert/albert-base-v2']

scores = {}
# nat_only = nat_only.sample(1000)

for model in models:
    results = obtain_predictions(nat_only, model, 5)
    mean_score_by_raised_top1 = results.groupby(['new_condi', 'raised_position'])['score_top1'].mean() * 100
    mean_score_by_raised_top5 = results.groupby(['new_condi', 'raised_position'])['score_top5'].mean() * 100
    mean_score_by_raised_top5_2 = results.groupby(['new_condi', 'raised_position'])['score2_top5'].mean() * 100
    mean_score_by_raised_top5_3 = results.groupby(['new_condi', 'raised_position'])['score3_top5'].mean() * 100
    scores[f'{model}, top_k = 1'] = mean_score_by_raised_top1
    scores[f'{model}, top_k = 5'] = mean_score_by_raised_top5
    scores[f'{model}, top_k = 5, metric2'] = mean_score_by_raised_top5_2
    scores[f'{model}, top_k = 5, metric3'] = mean_score_by_raised_top5_3


