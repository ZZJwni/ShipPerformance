#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[9]:


import sys
sys.path.append('..')


# In[10]:


from datetime import datetime, date


# In[11]:


from model.utils import evaluate_model
from model.utils import plot_train_val_loss, plot_pred_test
from model.utils import split_train_test, standardize_train_test
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


# In[12]:


feature_filename = 'leg_top5_feat.csv'


# In[13]:


df = pd.read_csv(feature_path / feature_filename,
                 index_col=0, parse_dates=['utc'])
ship_ids = np.unique(df.index)

print(ship_ids)


# ###### 12796 and 18180

# In[14]:


X_y = df.loc[(12796, 18180), :].reset_index().sort_values(by='utc').reset_index(drop=True)
X = X_y.drop(['foc_me', 'utc', 'ship_id'], axis=1)
y = X_y['foc_me']
index = X_y[['utc', 'ship_id']]
print('The number of features : ', X.shape[1])


# In[17]:


df_ = df.loc[(12796, 18180), :].copy().sort_values(by='utc')


# In[22]:


df_.iloc[int(len(df_)*.75): ].groupby('ship_id').apply(lambda x: len(x))


# In[20]:


scaler = StandardScaler()
X_train_val_, y_train_val_, X_test_, y_test_ = split_train_test(X, y, .25)
X_train_val, y_train_val, X_test, y_test = standardize_train_test(scaler, X_train_val_, y_train_val_, X_test_, y_test_)
X_train, y_train, X_val, y_val = split_train_test(X_train_val, y_train_val, .2, shuffle=True)

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[21]:


batch_size = 64
epochs = 3000
optimizer = optimizers.SGD(learning_rate=.001)
loss = losses.mean_absolute_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callback = [callbacks.EarlyStopping(patience=500)]


# In[22]:


model = NN(input_size=X.shape[1], output_size=1, layer_params=[
           layer_param(256, 'relu', .36), layer_param(256, 'relu', .47),
           layer_param(64, 'sigmoid', .04), layer_param(512, 'relu', .53)],
           loss=loss, optimizer=optimizer, metrics=val_metrics)

model.build()


# In[23]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callback, batch_size=batch_size, epochs=epochs)


# In[24]:


y_test_pred = model.predict(X_test, rescaler=scaler)
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=y_test_, y=y_test_pred, ax=ax)
sns.lineplot(x=y_test_, y=y_test_, ax=ax, color='r')
_ = ax.set_title('y_test vs y_pred')


# In[25]:


y_train_pred = model.predict(X_train_val, rescaler=scaler)
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=y_train_val_, y=y_train_pred, ax=ax)
sns.lineplot(x=y_train_val_, y=y_train_val_, ax=ax, color='r')
_ = ax.set_title('y_train vs y_train_pred')


# In[26]:


evaluate_model(y_test=y_test_, y_test_pred=y_test_pred,
               y_train=y_train_val_, y_train_pred=y_train_pred)


# ##### Error Analysis

# In[42]:


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




