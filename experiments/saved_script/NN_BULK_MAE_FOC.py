#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


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


feature_filename = 'feat_bulk.csv'
label = 4
target = ['foc_me']
all_feats = ['crnt_lg', 'wind_lg', 
             'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
             'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
             'draft_aft', 'draft_fore', 'trim', 'draft_mean',
             'rpm', 'og_speed', 
             'dwt', 'draft', 'breadth',
             'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
             'date_built_year', 'power_at_mcr',
             'speed_norm', 'rpm_norm']

feats_1 = ['crnt_lg', 'wind_lg', 
             'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
             'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
             'draft_aft', 'draft_fore', 'trim', 'draft_mean',
             'rpm', 'og_speed', ]

feats_2 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
           'rpm',
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr',]

feats_3 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
           'rpm',
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr', 'rpm_norm']


# In[4]:


df = (pd
      .read_csv(feature_path / feature_filename,
                index_col=0, parse_dates=['utc']))


# In[5]:


df = (pd
      .read_csv(feature_path / feature_filename,
                index_col=0, parse_dates=['utc'])
      .loc[lambda df: df['label'] == label, :]
     )
ship_ids = np.unique(df['ship_id'])

print('The number of ship id is : ', len(ship_ids))


# In[7]:


df.loc[:, 'rpm_3'] = df['rpm'] ** 3 
df.loc[:, 'rpm_norm_3'] = df['rpm_norm'] **3
feats_2_ = feats_3 + ['rpm_3', 'rpm_norm_3']


# ######  feats_2

# In[10]:


df_ = df.reset_index(drop=True)
X = df_[feats_2_].to_numpy(dtype=np.float32)
y = df_[target].to_numpy(dtype=np.float32).flatten()
index = df_[['utc', 'ship_id', 'record_index']]
print('The number of samples : ', X.shape[0])
print('The number of features : ', X.shape[1])


# In[11]:


train_val_idx, test_idx = list(kfold_by_shipid(df=df_, n_splits=5))[-1] # test_idx is the last 20% records each ship_id
train_idx, val_idx = list(kfold_by_shipid(df=df_.iloc[train_val_idx, :], n_splits=6))[-1] # val_idx is the last 16.6% records each ship_id

scaler = StandardScaler()
X_train_val_, X_test_ = X[train_val_idx, :], X[test_idx, :]
y_train_val_, y_test_ = y[train_val_idx], y[test_idx]
X_train_val, y_train_val, X_test, y_test = standardize_train_test(scaler, X_train_val_, y_train_val_, X_test_, y_test_)
X_train, X_val = X_train_val[train_idx, :], X_train_val[val_idx, :]
y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[12]:


batch_size = 32
epochs = 200
optimizer = optimizers.Adam(learning_rate=.001)
loss = losses.mean_absolute_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callback = [callbacks.EarlyStopping(patience=25)]


# In[13]:


model = NN(input_size=X.shape[1], output_size=1, layer_params=[
           layer_param(256, 'relu', .36), layer_param(256, 'relu', .47),
           layer_param(64, 'sigmoid', .04), layer_param(512, 'relu', .53)],
           loss=loss, optimizer=optimizer, metrics=val_metrics)

model.build()


# In[14]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callback, batch_size=batch_size, epochs=epochs)


# In[18]:


df_.iloc[test_idx, :].groupby('ship_id').apply(lambda x: len(x)).sort_values().tail(20)


# In[65]:


ship_id = 22347


# In[66]:


test_data = df_.iloc[test_idx, :].query('ship_id == @ship_id')[feats_2]
test_label = df_.iloc[test_idx, :].query('ship_id == @ship_id')[target]
test_data_ = test_data.to_numpy(dtype=np.float32)
test_label_ = test_label.to_numpy(dtype=np.float32).flatten()


# In[67]:


test_data


# In[68]:


test_data_.shape


# In[69]:


idx=1000


# In[70]:


x = test_data_[idx,:]
y = np.array(test_label_[idx])
x_y = np.concatenate((x,y), axis=None)


# In[71]:


x_y_ = np.repeat(x_y[:, np.newaxis], 50, axis=1).T
rpm_range = np.linspace(start=df_.query('ship_id == @ship_id')['rpm'].min(), stop=df_.query('ship_id == @ship_id')['rpm'].max(), num=50)


# In[72]:


x_y_[:, 14] = rpm_range


# In[73]:


X_y_rescale = scaler.transform(x_y_)


# In[74]:


X_rescale = X_y_rescale[:, :-1]


# In[75]:


y_pred = model.predict(X_rescale, rescaler=scaler)


# In[76]:


y_pred


# In[77]:


fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x=rpm_range, y=y_pred, ax=ax)


# In[78]:


df_.query('ship_id == @ship_id')['rpm'].min(), df_.query('ship_id == @ship_id')['rpm'].max(),


# In[79]:


df_.query('ship_id == @ship_id')['foc_me'].min(), df_.query('ship_id == @ship_id')['foc_me'].max()


# In[15]:


y_test_pred = model.predict(X_test, rescaler=scaler)
# fig, ax = plt.subplots(figsize=(16, 8))
# sns.scatterplot(x=y_test_, y=y_test_pred, ax=ax)
# sns.lineplot(x=y_test_, y=y_test_, ax=ax, color='r')
# _ = ax.set_title('y_test vs y_pred')


# In[16]:


y_train_pred = model.predict(X_train_val, rescaler=scaler)
# fig, ax = plt.subplots(figsize=(16, 8))
# sns.scatterplot(x=y_train_val_, y=y_train_pred, ax=ax)
# sns.lineplot(x=y_train_val_, y=y_train_val_, ax=ax, color='r')
# _ = ax.set_title('y_train vs y_train_pred')


# In[17]:


evaluate_model(y_test=y_test_, y_test_pred=y_test_pred,
               y_train=y_train_val_, y_train_pred=y_train_pred)


# In[18]:


result = df_[['utc', 'ship_id', 'record_index', 'label', 'record_length', 'foc_me']]
result = result.assign(pred = 0, is_train_val=False, is_test=False)
result.loc[train_val_idx, 'pred'] = y_train_pred
result.loc[test_idx, 'pred'] = y_test_pred
result.loc[train_val_idx, 'is_train_val'] = True
result.loc[test_idx, 'is_test'] = True


# In[19]:


result


# In[20]:


colors = sns.color_palette(palette='muted', n_colors=3)
labels = ['foc_me', 'focme_train_pred', 'focme_test_pred']


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


# In[21]:


from sklearn.metrics import mean_absolute_error

result_ = result.query('is_test == True')
record_index = result_['record_index'].unique()
testmae_by_record = result_.groupby('record_index').apply(
    lambda col: mean_absolute_error(col['foc_me'], col['pred']))


# In[24]:


testmae_by_record.sort_values().tail(20)


# In[23]:


from tqdm.notebook import tqdm_notebook
for idx in tqdm_notebook(testmae_by_record.sort_values().index):
    plot_pred(df=result_.query('record_index == @idx'), colors=colors,
              labels=labels, output_figpath='fig/feats_2_focme/')


# In[ ]:




