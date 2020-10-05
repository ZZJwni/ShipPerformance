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
label = 2
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
            'rpm', 'og_speed', 
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr',]

feats_3 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
         
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr', 'speed_norm', 'rpm_norm']

feats_4 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
           'rpm', 'og_speed', 
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr', 'speed_norm', 'rpm_norm']


# In[4]:


df = (pd
      .read_csv(feature_path / feature_filename,
                index_col=0, parse_dates=['utc'])
      .loc[lambda df: df['label'] == label, :]
     )
ship_ids = np.unique(df['ship_id'])

print('The number of ship id is : ', len(ship_ids))


# ######  feats_2

# In[5]:


df_ = df.reset_index(drop=True)
X = df_[feats_2].to_numpy(dtype=np.float32)
y = df_[target].to_numpy(dtype=np.float32).flatten()
index = df_[['utc', 'ship_id', 'record_index']]
print('The number of samples : ', X.shape[0])
print('The number of features : ', X.shape[1])


# In[20]:


df_[feats_2].columns


# In[6]:


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


batch_size = 64
epochs = 10
optimizer = optimizers.Adam(learning_rate=.001)
loss = losses.mean_absolute_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callback = [callbacks.EarlyStopping(patience=200)]


# In[13]:


model = NN(input_size=X.shape[1], output_size=1, layer_params=[
           layer_param(256, 'relu', .36), layer_param(256, 'relu', .47),
           layer_param(64, 'sigmoid', .04), layer_param(512, 'relu', .53)],
           loss=loss, optimizer=optimizer, metrics=val_metrics)

model.build()


# In[14]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callback, batch_size=batch_size, epochs=epochs)


# In[15]:


y_test_pred = model.predict(X_test, rescaler=scaler)
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=y_test_, y=y_test_pred, ax=ax)
sns.lineplot(x=y_test_, y=y_test_, ax=ax, color='r')
_ = ax.set_title('y_test vs y_pred')


# In[16]:


y_train_pred = model.predict(X_train_val, rescaler=scaler)
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=y_train_val_, y=y_train_pred, ax=ax)
sns.lineplot(x=y_train_val_, y=y_train_val_, ax=ax, color='r')
_ = ax.set_title('y_train vs y_train_pred')


# In[17]:


evaluate_model(y_test=y_test_, y_test_pred=y_test_pred,
               y_train=y_train_val_, y_train_pred=y_train_pred)


# ##### Error Analysis

# In[18]:


result = X_y.iloc[X_train_val.shape[0]:].copy()


# In[98]:


train = X_y.iloc[:X_train_val.shape[0]].copy()


# In[53]:


result.loc[:, 'pred'] = y_test_pred
result.loc[:, 'test']  = y_test_
result.loc[:, 'error'] = (y_test_ - y_test_pred) / y_test_


# In[54]:


result.head()


# In[60]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='error', data=result.query('ship_id == 12796'), ax=ax)
sns.lineplot(x=result.query('ship_id == 12796')['utc'], y=-.25, ax=ax)


# In[61]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='error', data=result.query('ship_id == 18180'), ax=ax)
sns.lineplot(x=result.query('ship_id == 12796')['utc'], y=-.25, ax=ax)


# In[76]:


cols = ['crnt_lg', 'wind_lg', 'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
       'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr', 'draft_aft', 'draft_fore',
       'trim', 'draft_mean', 'foc_me', 'rpm', 'og_speed']


# 12796

# In[106]:


data = result.query('ship_id == 12796')
train_ = train.query('ship_id == 12796')
for col in cols:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(data.query('error > -.25')
                 [col], kde=False, norm_hist=True, label='error > -.25', ax=ax, bins=20)
    sns.distplot(data.query('error <= -.25')
                 [col], kde=False, norm_hist=True, label='error <= -.25', ax=ax, bins=20)
#     sns.distplot(train_[col], kde=False, norm_hist=True, label='train', ax=ax)
    ax.set_title(col)
    fig.legend()


# In[11]:


result.loc[:, 'abs_error'] = np.abs(result['error'])


# In[25]:


df_ = result.query('ship_id == 12796').query('error> -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['wind_lg'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[21]:


df_ = result.query('ship_id == 12796').query('error<= -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['wind_lg'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[22]:


df_ = result.query('ship_id == 12796').query('error<= -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['rpm'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[26]:


df_ = result.query('ship_id == 12796').query('error> -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['rpm'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[29]:


df_ = result.query('ship_id == 12796').query('error<= -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['og_speed'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[30]:


df_ = result.query('ship_id == 12796').query('error> -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['og_speed'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[31]:


result.query('ship_id == 12796').sort_values(by='error').head(50)


# In[50]:


df_ = result.query('ship_id == 12796').loc[lambda df: (date(2018, 2, 21)<=df['utc']) & (df['utc']<=date(2018, 2, 25)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[54]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='rpm', data=df_, ax=ax, label='rpm')
# sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[55]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')
# sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[56]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='wind_lg', data=df_, ax=ax, label='wind_lg')
# sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[39]:


df_ = result.query('ship_id == 12796').loc[lambda df: (
    date(2017, 5, 7) <= df['utc']) & (df['utc'] <= date(2017, 5, 11)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[40]:


df_ = result.query('ship_id == 12796').loc[lambda df: (
    date(2018, 2, 22) <= df['utc']) & (df['utc'] <= date(2018, 2, 24)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[68]:


df_ = result.query('ship_id == 12796').loc[lambda df: (
    date(2016, 10, 27) <= df['utc']) & (df['utc'] <= date(2016, 11, 2)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[62]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='rpm', data=df_, ax=ax, label='rpm')
# sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[60]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='wind_lg', data=df_, ax=ax, label='wind_lg')
# sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[63]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[ ]:


# df_ = result.query('ship_id == 12796').loc[lambda df: (
    date(2016, 7,14) <= df['utc']) & (df['utc'] <= date(2016, 7, 18)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# 18180

# In[ ]:





# In[107]:


data = result.query('ship_id == 18180')
train_ = train.query('ship_id == 18180')
for col in cols:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(data.query('error > -.25')
                 [col], kde=False, norm_hist=True, label='error > -.25', ax=ax, bins=20)
    sns.distplot(data.query('error <= -.25')
                 [col], kde=False, norm_hist=True, label='error <= -.25', ax=ax, bins=20)
#     sns.distplot(train_[col], kde=False, norm_hist=True, label='train', ax=ax)
    ax.set_title(col)
    fig.legend()


# In[ ]:


df_ = result.query('ship_id == 18180').query('error> -.25')
fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df_['wind_lg'], data2=df_['error'], ax=ax,
            cmap="Reds", shade=True, shade_lowest=False)


# In[86]:


result.query('ship_id == 18180').sort_values(by='error').head(50)


# In[73]:


df_ = result.query('ship_id == 18180').loc[lambda df: (
    date(2016, 10, 10) <= df['utc']) & (df['utc'] <= date(2016, 10, 20)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[74]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='rpm', data=df_, ax=ax, label='rpm')


# In[75]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='wind_lg', data=df_, ax=ax, label='wind_lg')


# In[76]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[82]:


df_ = result.query('ship_id == 18180').loc[lambda df: (
    date(2017, 3, 18) <= df['utc']) & (df['utc'] <= date(2017, 3, 24)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[83]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='rpm', data=df_, ax=ax, label='rpm')


# In[84]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='wind_lg', data=df_, ax=ax, label='wind_lg')


# In[85]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[78]:


df_ = result.query('ship_id == 18180').loc[lambda df: (
    date(2018, 4, 10) <= df['utc']) & (df['utc'] <= date(2018, 4, 16)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[79]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='rpm', data=df_, ax=ax, label='rpm')


# In[80]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='og_speed', data=df_, ax=ax, label='og_speed')


# In[81]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='wind_lg', data=df_, ax=ax, label='wind_lg')


# In[87]:


df_ = result.query('ship_id == 18180').loc[lambda df: (
    date(2018, 3, 26) <= df['utc']) & (df['utc'] <= date(2018, 4, 2)), :]
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='utc', y='pred', data=df_, ax=ax, label='pred')
sns.lineplot(x='utc', y='test', data=df_, ax=ax, label='test')


# In[67]:


# result.to_csv('../output/12796_18180_mae_4layers.csv')


# In[24]:


result = pd.read_csv('../output/12796_18180_mae_4layers.csv', index_col=0, parse_dates=['utc'])


# In[36]:


# model.delete()


# In[ ]:




