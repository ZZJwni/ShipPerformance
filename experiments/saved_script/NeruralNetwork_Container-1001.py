#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sys
sys.path.append('..')

from datetime import datetime, date

from model.utils import evaluate_model
from model.utils import plot_train_val_loss, plot_pred_test
from model.utils import split_train_test, standardize_train_test, kfold_by_shipid
from model.utils import feature_path, output_figpath

from model import NN
from model.NN import layer_param

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow import optimizers, losses, metrics
from tensorflow.keras import callbacks

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# In[3]:


# Configure guildai
import guild.ipy as guild
import os

GUILD_HOME = 'guild-env'
DELETE_RUNS_ON_INIT = True


if not os.path.exists(GUILD_HOME):
    os.mkdir(GUILD_HOME)

guild.set_guild_home(GUILD_HOME)

if DELETE_RUNS_ON_INIT:
    deleted = guild.runs().delete(permanent=True)
    print("Deleted %i run(s)" % len(deleted))


# In[4]:


features = ['draft_aft', 'draft_fore', 'sig_ht', 
            'crnt_L', 'crnt_T','wind_L', 'wind_T', 
            'sea_ht_L', 'sea_ht_T', 'sea_per_L', 'sea_per_T',
            'swl_ht_L', 'swl_ht_T', 'swl_per_L', 'swl_per_T',
            'normalized_rpm', 'draft_sum_norm', 'draft_diff_norm']

targets = ['tw_speed', 'normalized_tw_speed',
           'normalized_foc_me', 'foc_me', 'og_speed']

variables = ['ship_id', 'client', 'record_length', 'record_index', 'utc', 'rpm']

ship_features = ['length', 'breadth', 'depth', 
                 'draft', 'dwt', 'gt', 'power_at_mcr', 'rpm_at_mcr',
                 'speed_at_mcr']


# In[4]:


feature_filename = 'features_container.csv'
ship_filename = 'db_container_ship.csv'


# In[6]:


df_feats = pd.read_csv(feature_path / feature_filename,
                       index_col=0, parse_dates=['utc'])

df_ships = pd.read_csv(feature_path / ship_filename, 
                       index_col=0)


# In[7]:


print('Number of records : ', len(df_feats))


# In[8]:


from sklearn.preprocessing import StandardScaler

# Standardize ship specification features
ship_scaler = StandardScaler()
df_ships_ = df_ships.copy()
df_ships_.loc[:, ship_features] = ship_scaler.fit_transform(df_ships_[ship_features].values)


# In[9]:


# merge ship specifition features and other features
df = pd.merge(left=df_feats, right=df_ships_[ship_features + ['ship_id']], on='ship_id', how='inner')


# In[10]:


print('The number of ships : ', len(df['ship_id'].unique()))
print('The number of records : ', len(df))


# In[11]:


df_ = df.sort_values(by=['ship_id', 'utc'])
df_ = df.reset_index(drop=True)


# In[12]:


df_ = df_.reset_index()
train_val_idx, test_idx = list(kfold_by_shipid(df=df_, n_splits=5))[-1] # test_idx is the last 20% records each ship_id
train_idx_, val_idx_ = list(kfold_by_shipid(df=df_.loc[train_val_idx, :], n_splits=6))[-1] # val_idx is the last 16.6% records each ship_id
train_idx =  df_.loc[train_val_idx, 'index'].iloc[train_idx_].values
val_idx = df_.loc[train_val_idx, 'index'].iloc[val_idx_].values

assert set(train_idx) & set(test_idx) == set()
assert set(val_idx) & set(test_idx) == set()
assert set(val_idx) & set(train_idx) == set()
df_ = df_.drop('index', axis=1)


# In[13]:


# Standardize draft features and weather features
# performance features and targets do not require Standardization.
feat_scaler = StandardScaler()
features_need_norm = ['draft_aft','draft_fore','sig_ht',
                      'crnt_L','crnt_T','wind_L','wind_T',
                      'sea_ht_L','sea_ht_T','sea_per_L','sea_per_T',
                      'swl_ht_L','swl_ht_T','swl_per_L','swl_per_T']
df_.loc[train_val_idx, features_need_norm] = feat_scaler.fit_transform(df_.loc[train_val_idx, features_need_norm].values)
df_.loc[test_idx, features_need_norm] = feat_scaler.transform(df_.loc[test_idx, features_need_norm].values)


# In[14]:


all_features = ['draft_aft', 'draft_fore', 'sig_ht', 'crnt_L', 'crnt_T', 'wind_L',
                'wind_T', 'sea_ht_L', 'sea_ht_T', 'sea_per_L', 'sea_per_T', 'swl_ht_L',
                'swl_ht_T', 'swl_per_L', 'swl_per_T', 'normalized_rpm',
                'draft_sum_norm', 'draft_diff_norm','length', 'breadth',
                'depth', 'draft', 'dwt', 'gt', 'power_at_mcr', 'rpm_at_mcr',
                'speed_at_mcr']

target_1 = ['normalized_tw_speed']
target_2 = ['normalized_foc_me']
target_3 = ['foc_me']


# In[15]:


X_train = df_.loc[train_idx, all_features]
y1_train = df_.loc[train_idx, target_1]
y2_train = df_.loc[train_idx, target_2]
y3_train = df_.loc[train_idx, target_3]
train_weight = df_.loc[train_idx, 'speed_at_mcr'].values

X_val = df_.loc[val_idx, all_features]
y1_val = df_.loc[val_idx, target_1]
y2_val = df_.loc[val_idx, target_2]
y3_val = df_.loc[val_idx, target_3]
val_weight = df_.loc[val_idx, 'speed_at_mcr'].values

X_test = df_.loc[test_idx, all_features]
y1_test = df_.loc[test_idx, target_1]
y2_test = df_.loc[test_idx, target_2]
y3_test = df_.loc[test_idx, target_3]
test_weight = df_.loc[test_idx, 'speed_at_mcr'].values


# In[16]:


print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# ##### tw_speed

# ###### train model

# In[17]:


batch_size = 64
epochs = 200
optimizer = optimizers.Adam(learning_rate=.001)
loss = losses.mean_absolute_percentage_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callback = [callbacks.EarlyStopping(patience=25)]


# In[18]:


model = NN(input_size=X_train.shape[1], output_size=1, layer_params=[
           layer_param(256, 'relu', .36), layer_param(256, 'relu', .47),
           layer_param(64, 'sigmoid', .04), layer_param(512, 'relu', .53)],
           loss=loss, optimizer=optimizer, metrics=val_metrics)

model.build()


# In[19]:


# def train():
#     model.train(X_train=X_train.values, y_train=y1_train.values, 
#             X_val=X_val.values, y_val=y1_val.values, 
#             callbacks=callback, batch_size=batch_size, epochs=epochs)
#     y_test_pred = model.predict(X_test.values, rescaler=None)
#     y_train_pred = model.predict(X_train.values, rescaler=None)
#     y_val_pred = model.predict(X_val.values, rescaler=None)
#     return (y_train_pred, y_val_pred, y_test_pred)

# run, returns = guild.run(train)


# In[20]:


# 40min
model.train(X_train=X_train.values, y_train=y1_train.values, 
            X_val=X_val.values, y_val=y1_val.values, 
            callbacks=callback, batch_size=batch_size, epochs=epochs)


# ###### error analysis

# In[21]:


y_test_pred = model.predict(X_test.values, rescaler=None)
y_train_pred = model.predict(X_train.values, rescaler=None)
y_val_pred = model.predict(X_val.values, rescaler=None)


# In[22]:


result_ = df_[['ship_id', 'utc', 'record_index',
               'tw_speed', 'normalized_tw_speed', 'og_speed']]
result_.loc[:, 'crnt_L'] = df['crnt_L']
result = pd.merge(left=result_, right=df_ships[['ship_id', 'rpm_at_mcr', 'speed_at_mcr']],
                  how='left', on='ship_id')
result.loc[:, 'pred_tw_speed'] = 0.
result.loc[:, 'pred_normalized_tw_speed'] = 0.
result.loc[:, 'pred_og_speed'] = 0.

result.loc[train_idx, 'pred_normalized_tw_speed'] = y_train_pred
result.loc[test_idx, 'pred_normalized_tw_speed'] = y_test_pred
result.loc[val_idx, 'pred_normalized_tw_speed'] = y_val_pred

result.loc[:, 'pred_tw_speed'] = result['pred_normalized_tw_speed'] * result['speed_at_mcr']
result.loc[:, 'pred_og_speed'] = result['pred_tw_speed'] + result['crnt_L']
result.loc[:, 'type'] = ''
result.loc[test_idx, 'type'] = 'test'
result.loc[val_idx, 'type'] = 'val'
result.loc[train_idx, 'type'] = 'train'


# In[23]:


result


# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.array, y_pred: np.array):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('The units of rmse and mae are [knot].')
print('Prediction results on train set, RMSE : {}; MAE : {}, R^2 : {}; mape : {}.'
      .format(np.sqrt(mean_squared_error(result.loc[train_idx, 'tw_speed'], result.loc[train_idx, 'pred_tw_speed'])),
              mean_absolute_error(
                  result.loc[train_idx, 'tw_speed'], result.loc[train_idx, 'pred_tw_speed']),
              r2_score(result.loc[train_idx, 'tw_speed'],
                       result.loc[train_idx, 'pred_tw_speed']),
              mape(result.loc[train_idx, 'tw_speed'],
                   result.loc[train_idx, 'pred_tw_speed'])
              )
      )

print('Prediction results on val set, RMSE : {}; MAE : {}, R^2 : {}; mape : {}.'
      .format(np.sqrt(mean_squared_error(result.loc[val_idx, 'tw_speed'], result.loc[val_idx, 'pred_tw_speed'])),
              mean_absolute_error(
                  result.loc[val_idx, 'tw_speed'], result.loc[val_idx, 'pred_tw_speed']),
              r2_score(result.loc[val_idx, 'tw_speed'],
                       result.loc[val_idx, 'pred_tw_speed']),
              mape(result.loc[val_idx, 'tw_speed'],
                   result.loc[val_idx, 'pred_tw_speed'])
              )
      )

print('Prediction results on test set, RMSE : {}; MAE : {}, R^2 : {}; mape : {}.'
      .format(np.sqrt(mean_squared_error(result.loc[test_idx, 'tw_speed'], result.loc[test_idx, 'pred_tw_speed'])),
              mean_absolute_error(
                  result.loc[test_idx, 'tw_speed'], result.loc[test_idx, 'pred_tw_speed']),
              r2_score(result.loc[test_idx, 'tw_speed'],
                       result.loc[test_idx, 'pred_tw_speed']),
              mape(result.loc[test_idx, 'tw_speed'],
                   result.loc[test_idx, 'pred_tw_speed'])
              )
      )


# In[25]:


print('The units of rmse and mae are [knot].')
print('Prediction results on train set, RMSE : {}; MAE : {}, R^2 : {}; mape : {}.'
      .format(np.sqrt(mean_squared_error(result.loc[train_idx, 'og_speed'], result.loc[train_idx, 'pred_og_speed'])),
              mean_absolute_error(
                  result.loc[train_idx, 'og_speed'], result.loc[train_idx, 'pred_og_speed']),
              r2_score(result.loc[train_idx, 'og_speed'],
                       result.loc[train_idx, 'pred_og_speed']),
              mape(result.loc[train_idx, 'og_speed'],
                   result.loc[train_idx, 'pred_og_speed'])
              )
      )

print('Prediction results on val set, RMSE : {}; MAE : {}, R^2 : {}; mape : {}.'
      .format(np.sqrt(mean_squared_error(result.loc[val_idx, 'og_speed'], result.loc[val_idx, 'pred_og_speed'])),
              mean_absolute_error(
                  result.loc[val_idx, 'og_speed'], result.loc[val_idx, 'pred_og_speed']),
              r2_score(result.loc[val_idx, 'og_speed'],
                       result.loc[val_idx, 'pred_og_speed']),
              mape(result.loc[val_idx, 'og_speed'],
                   result.loc[val_idx, 'pred_og_speed'])
              )
      )

print('Prediction results on test set, RMSE : {}; MAE : {}, R^2 : {}; mape : {}.'
      .format(np.sqrt(mean_squared_error(result.loc[test_idx, 'og_speed'], result.loc[test_idx, 'pred_og_speed'])),
              mean_absolute_error(
                  result.loc[test_idx, 'og_speed'], result.loc[test_idx, 'pred_og_speed']),
              r2_score(result.loc[test_idx, 'og_speed'],
                       result.loc[test_idx, 'pred_og_speed']),
              mape(result.loc[test_idx, 'og_speed'],
                   result.loc[test_idx, 'pred_og_speed'])
              )
      )


# In[1]:


result.to_csv(output_figpath / 'pred_container.csv')


# In[5]:


result = pd.read_csv(output_figpath / 'pred_container.csv', index_col=0, parse_dates=['utc'])
df_ships = pd.read_csv(feature_path / ship_filename, 
                       index_col=0)


# In[6]:


result_ = pd.merge(left=result, right=df_ships[['ship_id', 'label']], on='ship_id', how='left')


# In[7]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.array, y_pred: np.array):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[8]:


print(result_.query('type == \'test\'').groupby('label').apply(lambda x: mean_absolute_error(x['tw_speed'], x['pred_tw_speed'])))
print(result_.query('type == \'test\'').groupby('label').apply(lambda x: np.sqrt(mean_squared_error(x['tw_speed'], x['pred_tw_speed']))))
print(result_.query('type == \'test\'').groupby('label').apply(lambda x: mape(x['tw_speed'], x['pred_tw_speed'])))
print(result_.query('type == \'test\'').groupby('label').apply(lambda x: r2_score(x['tw_speed'], x['pred_tw_speed'])))


# In[44]:


colors = sns.color_palette(palette='muted', n_colors=4)
labels = ['tw_speed', 'pred_tw_speed:train', 'pred_tw_speed:val', 'pred_tw_speed:test']

def plot_tw_speed(df, colors, labels, output_figpath):
    
    y_train_pred = df.loc[lambda df: df['type'] == 'train', :]
    y_val_pred = df.loc[lambda df: df['type'] == 'val', :]
    y_test_pred = df.loc[lambda df: df['type'] == 'test', :]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    _ = ax.set_ylim(bottom=0, top=24)
    sns.lineplot(x='utc', y='tw_speed', data=df, color=colors[0], label=labels[0], ax=ax,
                 markers='.', dashes=False)
    
    train_mape, val_mape, test_mape = -1, -1, -1
    if len(y_train_pred) != 0:
        sns.lineplot(x='utc', y='pred_tw_speed', data=y_train_pred,
                     color=colors[1], label=labels[1], ax=ax,
                     markers='.', dashes=False)
        train_mape = mape(y_train_pred['tw_speed'], y_train_pred['pred_tw_speed'])
    if len(y_val_pred) != 0:
        sns.lineplot(x='utc', y='pred_tw_speed', data=y_val_pred,
                     color=colors[2], label=labels[2], ax=ax,
                     markers='.', dashes=False)
        val_mape = mape(y_val_pred['tw_speed'], y_val_pred['pred_tw_speed'])
    if len(y_test_pred) != 0:
        sns.lineplot(x='utc', y='pred_tw_speed', data=y_test_pred,
                     color=colors[3], label=labels[3], ax=ax,
                     markers='.', dashes=False)
        test_mape = mape(y_test_pred['tw_speed'], y_test_pred['pred_tw_speed'])
    
    record_index = df.iloc[0]['record_index']
    ship_id = df.iloc[0]['ship_id']
    ax.set_title(f'record : {record_index}, ship_id : {ship_id},\n' 
                 f'train_mape : {train_mape}, val_mape : {val_mape}, test_mape : {test_mape}.')
    fig.savefig(output_figpath + f'{ship_id}_{record_index}')


# In[39]:


labels_2 = ['og_speed', 'pred_og_speed:train',
            'pred_og_speed:val', 'pred_og_speed:test']


def plot_og_speed(df, colors, labels, output_figpath):

    y_train_pred = df.loc[lambda df: df['type'] == 'train', :]
    y_val_pred = df.loc[lambda df: df['type'] == 'val', :]
    y_test_pred = df.loc[lambda df: df['type'] == 'test', :]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='utc', y='og_speed', data=df,
                 color=colors[0], label=labels[0], ax=ax)

    train_mape, val_mape, test_mape = -1, -1, -1
    if len(y_train_pred) != 0:
        sns.lineplot(x='utc', y='pred_og_speed', data=y_train_pred,
                     color=colors[1], label=labels[1], ax=ax)
        train_mape = mape(y_train_pred['og_speed'],
                          y_train_pred['pred_og_speed'])
    if len(y_val_pred) != 0:
        sns.lineplot(x='utc', y='pred_og_speed', data=y_val_pred,
                     color=colors[2], label=labels[2], ax=ax)
        val_mape = mape(y_val_pred['og_speed'], y_val_pred['pred_og_speed'])
    if len(y_test_pred) != 0:
        sns.lineplot(x='utc', y='pred_og_speed', data=y_test_pred,
                     color=colors[3], label=labels[3], ax=ax)
        test_mape = mape(y_test_pred['og_speed'], y_test_pred['pred_og_speed'])

    ship_id = df.iloc[0]['ship_id']
    record_index = df.iloc[0]['record_index']
    ax.set_title(f'record : {record_index}, ship_id : {ship_id},\n'
                 f'train_mape : {train_mape}, val_mape : {val_mape}, test_mape : {test_mape}.')
    fig.savefig(output_figpath + f'{ship_id}_{record_index}_2')


# In[29]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true: np.array, y_pred: np.array):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


result_ = result.loc[lambda df: df['type'] == 'train', :]
mae_by_record = result_.groupby('record_index').apply(
    lambda y: mean_absolute_error(y['tw_speed'], y['pred_tw_speed']))
rmse_by_record = result_.groupby('record_index').apply(
    lambda y: np.sqrt(mean_squared_error(y['tw_speed'], y['pred_tw_speed'])))
mape_by_record = result_.groupby('record_index').apply(
    lambda y: mape(y['tw_speed'], y['pred_tw_speed']))
train_error_by_record_ = pd.DataFrame(
    {'mae': mae_by_record, 'rmse': rmse_by_record, 'mape': mape_by_record})
# train_error_by_record = train_error_by_record_.melt(
#     value_vars=['mae', 'rmse', 'mape'], value_name='error', var_name='type')
train_error_by_record = train_error_by_record_.melt(
    value_vars=['mae', 'rmse'], value_name='error', var_name='type')

result_ = result.loc[lambda df: df['type'] == 'test', :]
mae_by_record = result_.groupby('record_index').apply(
    lambda y: mean_absolute_error(y['tw_speed'], y['pred_tw_speed']))
rmse_by_record = result_.groupby('record_index').apply(
    lambda y: np.sqrt(mean_squared_error(y['tw_speed'], y['pred_tw_speed'])))
mape_by_record = result_.groupby('record_index').apply(
    lambda y: mape(y['tw_speed'], y['pred_tw_speed']))
mae_by_record = result_.groupby('record_index').apply(
    lambda y: mean_absolute_error(y['tw_speed'], y['pred_tw_speed']))
rmse_by_record = result_.groupby('record_index').apply(
    lambda y: np.sqrt(mean_squared_error(y['tw_speed'], y['pred_tw_speed'])))
mape_by_record = result_.groupby('record_index').apply(
    lambda y: mape(y['tw_speed'], y['pred_tw_speed']))
test_error_by_record_ = pd.DataFrame(
    {'mae': mae_by_record, 'rmse': rmse_by_record, 'mape': mape_by_record})
# test_error_by_record = train_error_by_record_.melt(
#     value_vars=['mae', 'rmse', 'mape'], value_name='error', var_name='type')
test_error_by_record = train_error_by_record_.melt(
    value_vars=['mae', 'rmse'], value_name='error', var_name='type')

result_ = result.loc[lambda df: df['type'] == 'val', :]
mae_by_record = result_.groupby('record_index').apply(
    lambda y: mean_absolute_error(y['tw_speed'], y['pred_tw_speed']))
rmse_by_record = result_.groupby('record_index').apply(
    lambda y: np.sqrt(mean_squared_error(y['tw_speed'], y['pred_tw_speed'])))
mape_by_record = result_.groupby('record_index').apply(
    lambda y: mape(y['tw_speed'], y['pred_tw_speed']))
val_error_by_record_ = pd.DataFrame(
    {'mae': mae_by_record, 'rmse': rmse_by_record, 'mape': mape_by_record})
# val_error_by_record = test_error_by_record_.melt(
#     value_vars=['mae', 'rmse', 'mape'], value_name='error', var_name='type')
val_error_by_record = test_error_by_record_.melt(
    value_vars=['mae', 'rmse'], value_name='error', var_name='type')


fig, axes_1 = plt.subplots(1, 3, figsize=(3 * 8, 8), sharey=True)
sns.boxplot(x='type', y='error', data=train_error_by_record, ax=axes_1[0])
axes_1[0].set_title('error on train data')
sns.boxplot(x='type', y='error', data=val_error_by_record, ax=axes_1[1])
axes_1[1].set_title('error on val data')
sns.boxplot(x='type', y='error', data=test_error_by_record, ax=axes_1[2])
axes_1[2].set_title('error on test data')

fig, axes_2 = plt.subplots(1, 3, figsize=(3 * 8, 8), sharey=True)
sns.histplot(data=train_error_by_record, x='error', hue='type', ax=axes_2[0])
axes_2[0].set_title('error on train data')
sns.histplot(data=val_error_by_record, x='error', hue='type', ax=axes_2[1])
axes_2[1].set_title('error on val data')
sns.histplot(data=test_error_by_record, x='error', hue='type', ax=axes_2[2])
axes_2[2].set_title('error on test data')


# In[33]:


# result_ = result.loc[lambda df: df['type'] == 'train', :]
# mae_by_record = result_.groupby('record_index').apply(
#     lambda y: mean_absolute_error(y['og_speed'], y['pred_og_speed']))
# rmse_by_record = result_.groupby('record_index').apply(
#     lambda y: np.sqrt(mean_squared_error(y['og_speed'], y['pred_og_speed'])))
# mape_by_record = result_.groupby('record_index').apply(
#     lambda y: mape(y['og_speed'], y['pred_og_speed']))
# train_error_by_record_ = pd.DataFrame(
#     {'mae': mae_by_record, 'rmse': rmse_by_record, 'mape': mape_by_record})
# train_error_by_record = train_error_by_record_.melt(
#     value_vars=['mae', 'rmse', 'mape'], value_name='error', var_name='type')

# result_ = result.loc[lambda df: df['type'] == 'test', :]
# mae_by_record = result_.groupby('record_index').apply(
#     lambda y: mean_absolute_error(y['og_speed'], y['pred_og_speed']))
# rmse_by_record = result_.groupby('record_index').apply(
#     lambda y: np.sqrt(mean_squared_error(y['og_speed'], y['pred_og_speed'])))
# mape_by_record = result_.groupby('record_index').apply(
#     lambda y: mape(y['og_speed'], y['pred_og_speed']))
# test_error_by_record_ = pd.DataFrame(
#     {'mae': mae_by_record, 'rmse': rmse_by_record, 'mape': mape_by_record})
# test_error_by_record = test_error_by_record_.melt(
#     value_vars=['mae', 'rmse', 'mape'], value_name='error', var_name='type')

# result_ = result.loc[lambda df: df['type'] == 'val', :]
# mae_by_record = result_.groupby('record_index').apply(
#     lambda y: mean_absolute_error(y['og_speed'], y['pred_og_speed']))
# rmse_by_record = result_.groupby('record_index').apply(
#     lambda y: np.sqrt(mean_squared_error(y['og_speed'], y['pred_og_speed'])))
# mape_by_record = result_.groupby('record_index').apply(
#     lambda y: mape(y['og_speed'], y['pred_og_speed']))
# val_error_by_record_ = pd.DataFrame(
#     {'mae': mae_by_record, 'rmse': rmse_by_record, 'mape': mape_by_record})
# val_error_by_record = val_error_by_record_.melt(
#     value_vars=['mae', 'rmse', 'mape'], value_name='error', var_name='type')


# fig, axes_1 = plt.subplots(1, 3, figsize=(3 * 8, 8), sharey=True)
# sns.boxplot(x='type', y='error', data=train_error_by_record, ax=axes_1[0])
# axes_1[0].set_title('error on train data')
# sns.boxplot(x='type', y='error', data=val_error_by_record, ax=axes_1[1])
# axes_1[1].set_title('error on val data')
# sns.boxplot(x='type', y='error', data=test_error_by_record, ax=axes_1[2])
# axes_1[2].set_title('error on test data')

# fig, axes_2 = plt.subplots(1, 3, figsize=(3 * 8, 8), sharey=True)
# sns.histplot(data=train_error_by_record, x='error', hue='type', ax=axes_2[0])
# axes_2[0].set_title('error on train data')
# sns.histplot(data=val_error_by_record, x='error', hue='type', ax=axes_2[1])
# axes_2[1].set_title('error on val data')
# sns.histplot(data=test_error_by_record, x='error', hue='type', ax=axes_2[2])
# axes_2[2].set_title('error on test data')


# In[45]:


from tqdm.notebook import tqdm_notebook
for idx in tqdm_notebook(test_error_by_record_.index):
    plot_tw_speed(df=result.query('record_index == @idx'), colors=colors,
                  labels=labels, output_figpath='fig/tw_speed_container_1001/')
#     plot_og_speed(df=result.query('record_index == @idx'), colors=colors,
#                   labels=labels_2, output_figpath='fig/tw_speed_container0_0924/')


# In[ ]:




