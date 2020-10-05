#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append('..')


# In[4]:


import seaborn as sns
from matplotlib import pyplot as plt
from model.utils import evaluate_model, output_datapath
from datetime import datetime, date
import pandas as pd
import numpy as np


sns.set_style("whitegrid")

from tqdm import tqdm_notebook
from sklearn.metrics import mean_absolute_error


# In[4]:


# 参数
#     batch_size = 64
#     epochs = 150
#     optimizer = optimizers.Adam(learning_rate=.001)
#     loss = losses.mean_absolute_error
#     val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
#     callback = [callbacks.EarlyStopping(patience=25)]


# ##### feats_2

# In[5]:


results = []
for label in [0,1,2,3,4]:
    results.append(pd.read_csv(output_datapath / f'feats_2_{label}_foc_me_1.csv', index_col=0, parse_dates=['utc']))
df = (pd
      .concat(results, axis=0)
      .sort_values(by = ['ship_id', 'record_index', 'utc'])
     )


# In[7]:


type_train_val = 'train_val'
type_test = 'test'
for label in [0, 1, 2, 3, 4]:
    print('Label : ', int(label))
    train_val_pred = df.query('label == @label').query('is_train_val == True')
    test_pred = df.query('label == @label').query('is_test == True')
    print('Number of train and val : ', len(train_val_pred))
    print('Number of test : ', len(test_pred))
    evaluate_model(y_test=test_pred['foc_me'].values, y_test_pred=test_pred['pred'].values,
                   y_train=train_val_pred['foc_me'].values, y_train_pred=train_val_pred['pred'].values)


# In[8]:


colors = sns.color_palette(palette='muted', n_colors=3)
labels = ['foc_me', 'foc_me_train_pred', 'foc_me_test_pred']


def plot_pred(df, colors, labels, output_figpath):
    y = df[['utc', 'foc_me']]
    y_train_pred = df.loc[lambda df: df['is_train_val']
                          == True, ['utc', 'pred']]
    y_test_pred = df.loc[lambda df: df['is_test'] == True, ['utc', 'pred']]
    if len(y_train_pred) == 0 and len(y_test_pred) == 0:
        raise ValueError

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='utc', y='foc_me', data=y,
                 color=colors[0], label=labels[0], ax=ax)
    train_mae, test_mae = 'NA', 'NA'
    if len(y_train_pred) != 0:
        if len(y_train_pred) == len(y):
            train_mae = mean_absolute_error(
                y['foc_me'].values, y_train_pred['pred'].values)
            train_mae = np.round(train_mae, 3)
        sns.lineplot(x='utc', y='pred', data=y_train_pred,
                     color=colors[1], label=labels[1], ax=ax)
    if len(y_test_pred) != 0:
        if len(y_test_pred) == len(y):
            test_mae = mean_absolute_error(
                y['foc_me'].values, y_test_pred['pred'].values)
            test_mae = np.round(test_mae, 3)
        sns.lineplot(x='utc', y='pred', data=y_test_pred,
                     color=colors[2], label=labels[2], ax=ax)

    ship_id = df.query('record_index == @idx').iloc[0]['ship_id']
    label = df.query('record_index == @idx').iloc[0]['label']
    ax.set_title(
        f'record : {idx}, ship_id : {ship_id}, label : {label}\n mae@train : {train_mae}, mae@test : {test_mae}')
    fig.savefig(output_figpath +
                f'record : {idx}, ship_id : {ship_id}, label : {label}')


# ###### test

# In[10]:


df_ = df.query('is_test == True')
record_index= df_['record_index'].unique()

testmae_by_record = df_.groupby('record_index').apply(lambda col : mean_absolute_error(col['foc_me'], col['pred']))


# In[11]:


testmae_by_record


# In[ ]:


for idx in tqdm_notebook(testmae_by_record.sort_values().index):
    plot_pred(df=df_.query('record_index == @idx'), colors=colors,
              labels=labels, output_figpath='fig/fig_2_foc_2/')


# In[ ]:


def plot_pred_2(df, colors, labels, output_figpath):
    y = df[['utc', 'og_speed', 'rpm']]
    y_train_pred = df.loc[lambda df: df['is_train_val']
                          == True, ['utc', 'pred']]
    y_test_pred = df.loc[lambda df: df['is_test'] == True, ['utc', 'pred']]
    if len(y_train_pred) == 0 and len(y_test_pred) == 0:
        raise ValueError

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='utc', y='og_speed', data=y,
                 color=colors[0], label=labels[0], ax=ax)
    train_mae, test_mae = 'NA', 'NA'
    if len(y_train_pred) != 0:
        if len(y_train_pred) == len(y):
            train_mae = mean_absolute_error(
                y['og_speed'].values, y_train_pred['pred'].values)
            train_mae = np.round(train_mae, 3)
        sns.lineplot(x='utc', y='pred', data=y_train_pred,
                     color=colors[1], label=labels[1], ax=ax)
    if len(y_test_pred) != 0:
        if len(y_test_pred) == len(y):
            test_mae = mean_absolute_error(
                y['og_speed'].values, y_test_pred['pred'].values)
            test_mae = np.round(test_mae, 3)
        sns.lineplot(x='utc', y='pred', data=y_test_pred,
                     color=colors[2], label=labels[2], ax=ax)

    ship_id = df.query('record_index == @idx').iloc[0]['ship_id']
    label = df.query('record_index == @idx').iloc[0]['label']
    ax.set_title(
        f'record : {idx}, ship_id : {ship_id}, label : {label}\n mae@train : {train_mae}, mae@test : {test_mae}')
    fig.savefig(output_figpath +
                f'record : {idx}, ship_id : {ship_id}, label : {label}')


# ###### train

# In[90]:


df_ = df.query('is_train_val == True')
record_index= df_['record_index'].unique()

trainmae_by_record = df_.groupby('record_index').apply(lambda col : mean_absolute_error(col['foc_me'], col['pred']))


# In[ ]:


for idx in tqdm_notebook(trainmae_by_record.sort_values().index):
    plot_pred(df=df_.query('record_index == @idx'), colors=colors,
              labels=labels , output_figpath='fig/feats_2_train/')


# ##### feats_1

# In[3]:


results = []
for label in [0,1,2,3,4]:
    results.append(pd.read_csv(output_datapath / f'feats_1_{label}_2.csv', index_col=0, parse_dates=['utc']))
df = (pd
      .concat(results, axis=0)
      .sort_values(by = ['ship_id', 'record_index', 'utc'])
     )


# In[5]:


type_train_val = 'train_val'
type_test = 'test'
for label in [0, 1, 2, 3, 4]:
    print('Label : ', int(label))
    train_val_pred = df.query('label == @label').query('is_train_val == True')
    test_pred = df.query('label == @label').query('is_test == True')
    print('Number of train and val : ', len(train_val_pred))
    print('Number of test : ', len(test_pred))
    evaluate_model(y_test=test_pred['foc_me'].values, y_test_pred=test_pred['pred'].values,
                   y_train=train_val_pred['foc_me'].values, y_train_pred=train_val_pred['pred'].values)


# ###### test

# In[6]:


df_ = df.query('is_test == True')
record_index= df_['record_index'].unique()

testmae_by_record = df_.groupby('record_index').apply(lambda col : mean_absolute_error(col['foc_me'], col['pred']))


# In[11]:


df_


# In[12]:


for idx in tqdm_notebook(testmae_by_record.sort_values().index):
    plot_pred(df=df_.query('record_index == @idx'), colors=colors,
              labels=labels, output_figpath='fig/feats_1/')


# In[ ]:




