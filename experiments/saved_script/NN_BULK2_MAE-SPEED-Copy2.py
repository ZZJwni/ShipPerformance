#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[1]:


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


# In[2]:


feature_filename = 'feat_bulk.csv'
label = 4
target_1 = ['tw_speed']
target_2 = ['normalized_tw_speed']
features = [
    'sig_ht', 
    'dwt', 'draft', 'breadth', 'length', 'depth', 'gt',
    'rpm_at_mcr', 'speed_at_mcr','power_at_mcr', 
    'crnt_L', 'crnt_T','wind_L', 'wind_T', 
    'sea_ht_L', 'sea_ht_T', 'sea_per_L', 'sea_per_T',
    'swl_ht_L', 'swl_ht_T', 'swl_per_L', 'swl_per_T',
    'normalized_rpm', 'trim', 'draft_mean']

feats_1 = features
feats_2 = [
    'dwt', 'draft', 'breadth', 'length', 'depth', 'gt',
    'rpm_at_mcr', 'speed_at_mcr','power_at_mcr', 
    'crnt_L', 'crnt_T','wind_L', 'wind_T', 
    'sea_ht_L', 'sea_ht_T', 'sea_per_L', 'sea_per_T',
    'swl_ht_L', 'swl_ht_T', 'swl_per_L', 'swl_per_T',
    'normalized_rpm', 'trim', 'draft_mean']


# In[3]:


df = (pd
      .read_csv(feature_path / feature_filename,
                index_col=0, parse_dates=['utc']))


# In[ ]:


#横川さん、draft_aft(vovage)はdraft(ship_db)より大きい記録があります。


# In[12]:


df.sort_values(by='trim')[['draft_aft', 'draft_fore', 'draft', 'ship_id', 'utc']].head(50)


# In[ ]:





# In[7]:


df['trim'].max()


# In[5]:


df = (pd
      .read_csv(feature_path / feature_filename,
                index_col=0, parse_dates=['utc'])
      .loc[lambda df: df['label'] == label, :]
     )
ship_ids = np.unique(df['ship_id'])

print('The number of ship id is : ', len(ship_ids))


# ######  feats_1 and target_2

# In[27]:


df.query('record_index == 232945').sort_values(by='tw_speed')


# In[42]:


df.loc[(df['crnt_u'] > 9000) | (df['crnt_v'] > 9000), 'ship_id'].unique()


# In[6]:


df_ = df.reset_index(drop=True)
X = df_[feats_2].to_numpy(dtype=np.float32)
y = df_[target_2].to_numpy(dtype=np.float32).flatten()
index = df_[['utc', 'ship_id', 'record_index']]
print('The number of samples : ', X.shape[0])
print('The number of features : ', X.shape[1])


# In[7]:


train_val_idx, test_idx = list(kfold_by_shipid(df=df_, n_splits=5))[-1] # test_idx is the last 20% records each ship_id
train_idx, val_idx = list(kfold_by_shipid(df=df_.iloc[train_val_idx, :], n_splits=6))[-1] # val_idx is the last 16.6% records each ship_id

# Standardize feats together
scaler = StandardScaler()
X_train_val_, X_test_ = X[train_val_idx, :], X[test_idx, :]
y_train_val_, y_test_ = y[train_val_idx], y[test_idx]
X_train_val, y_train_val, X_test, y_test = standardize_train_test(scaler, X_train_val_, y_train_val_, X_test_, y_test_)
X_train, X_val = X_train_val[train_idx, :], X_train_val[val_idx, :]
y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[8]:


batch_size = 64
epochs = 120
optimizer = optimizers.Adam(learning_rate=.001)
loss = losses.mean_absolute_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callback = [callbacks.EarlyStopping(patience=20)]


# In[9]:


model = NN(input_size=X.shape[1], output_size=1, layer_params=[
           layer_param(256, 'relu', .36), layer_param(256, 'relu', .47),
           layer_param(64, 'sigmoid', .04), layer_param(512, 'relu', .53)],
           loss=loss, optimizer=optimizer, metrics=val_metrics)

model.build()


# In[10]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callback, batch_size=batch_size, epochs=epochs)


# In[11]:


y_test_pred = model.predict(X_test, rescaler=scaler)
y_train_pred = model.predict(X_train_val, rescaler=scaler)


# In[12]:


evaluate_model(y_test=y_test_, y_test_pred=y_test_pred,
               y_train=y_train_val_, y_train_pred=y_train_pred)


# In[13]:


result = df_[['utc', 'ship_id', 'record_index', 'label', 'record_length', 'normalized_tw_speed', 'tw_speed', 'speed_at_mcr']]
result = result.assign(pred = 0, is_train_val=False, is_test=False)
result.loc[train_val_idx, 'pred'] = y_train_pred
result.loc[test_idx, 'pred'] = y_test_pred
result.loc[train_val_idx, 'is_train_val'] = True
result.loc[test_idx, 'is_test'] = True
result.loc[:, 'pred_speed'] = result['pred'] * result['speed_at_mcr']


# In[26]:


result.query('record_index == 232945').sort_values(by='tw_speed')


# In[15]:


from sklearn.metrics import mean_absolute_error


# In[16]:


mean_absolute_error(result['tw_speed'], result['pred_speed'])


# In[17]:


colors = sns.color_palette(palette='muted', n_colors=3)
labels = ['tw_speed', 'speed_train_pred', 'speed_test_pred']


def plot_pred(df, colors, labels, output_figpath):
    y = df[['utc', 'tw_speed']]
    y_train_pred = df.loc[lambda df: df['is_train_val']
                          == True, ['utc', 'pred_speed']]
    y_test_pred = df.loc[lambda df: df['is_test'] == True, ['utc', 'pred_speed']]
    if len(y_train_pred) == 0 and len(y_test_pred) == 0:
        raise ValueError

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='utc', y='tw_speed', data=y,
                 color=colors[0], label=labels[0], ax=ax)
    train_mae, test_mae = 'NA', 'NA'
    if len(y_train_pred) != 0:
        if len(y_train_pred) == len(y):
            train_mae = mean_absolute_error(
                y['tw_speed'].values, y_train_pred['pred_speed'].values)
            train_mae = np.round(train_mae, 3)
        sns.lineplot(x='utc', y='pred_speed', data=y_train_pred,
                     color=colors[1], label=labels[1], ax=ax)
    if len(y_test_pred) != 0:
        if len(y_test_pred) == len(y):
            test_mae = mean_absolute_error(
                y['tw_speed'].values, y_test_pred['pred_speed'].values)
            test_mae = np.round(test_mae, 3)
        sns.lineplot(x='utc', y='pred_speed', data=y_test_pred,
                     color=colors[2], label=labels[2], ax=ax)

    ship_id = df.query('record_index == @idx').iloc[0]['ship_id']
    label = df.query('record_index == @idx').iloc[0]['label']
    ax.set_title(
        f'record : {idx}, ship_id : {ship_id}, label : {label}\n mae@train : {train_mae}, mae@test : {test_mae}')
    fig.savefig(output_figpath +
                f'record : {idx}, ship_id : {ship_id}, label : {label}')


# In[24]:


from sklearn.metrics import mean_absolute_error

result_ = result.query('is_test == True')
record_index = result_['record_index'].unique()
testmae_by_record = result_.groupby('record_index').apply(
    lambda col: mean_absolute_error(col['tw_speed'], col['pred_speed']))


# In[44]:


testmae_by_record.sort_values().head(20)


# In[47]:


testmae_by_record.sort_values().tail()


# In[32]:


result_ = result.loc[~result['record_index'].isin([95078, 89983, 232945]), :]


# In[33]:


mean_absolute_error(result_['tw_speed'], result_['pred_speed'])


# In[ ]:


from tqdm.notebook import tqdm_notebook
for idx in tqdm_notebook(testmae_by_record.sort_values().index):
    plot_pred(df=result_.query('record_index == @idx'), colors=colors,
              labels=labels, output_figpath='fig/tw_speed/')


# In[ ]:




