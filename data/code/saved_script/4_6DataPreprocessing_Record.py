#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

from pathlib import Path
ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# ###### bulk carrier

# In[2]:


db_bulk_carrier = pd.read_csv(data_path / 'processed/db_bulk_acrrier.csv')

df = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv',
                 parse_dates=['utc'],
                 usecols=['course', 'crnt_u', 'crnt_v',
                          'draft_aft', 'draft_fore', 'foc_me',
                          'og_speed', 'rpm', 'sea_dir',
                          'sea_ht', 'sea_per', 'sig_ht',
                          'swl_dir', 'swl_ht',
                          'swl_per', 'wind_u', 'wind_v',
                          'client', 'utc', 'ship_id'],
                 dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                      'draft_aft', 'draft_fore', 'foc_me',
                                      'og_speed', 'rpm', 'sea_dir',
                                      'sea_ht', 'sea_per', 'sig_ht',
                                      'swl_dir', 'swl_ht',
                                      'swl_per', 'wind_u', 'wind_v'], np.float32))

df_ = df.loc[df['ship_id'].isin(db_bulk_carrier['ship_id']), :].copy()
del df
df = df_


# In[3]:


df.info()


# In[4]:


draft_columns = ['draft_aft', 'draft_fore']
performance_columns = ['foc_me', 'rpm']
speed_columns = ['og_speed', 'course']
crnt_columns = ['crnt_u', 'crnt_v']
sea_columns = ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']
swell_columns = ['swl_dir', 'swl_ht', 'swl_per']
wind_columns = ['wind_u', 'wind_v']

remaining_columns = set(df.columns) -set(draft_columns) -set(performance_columns) -set(speed_columns) - set(crnt_columns) - set(sea_columns) - set(swell_columns) - set(wind_columns)
assert remaining_columns == {'ship_id', 'utc', 'client'}


# In[5]:


# reset index
df.reset_index(inplace=True, drop=True)

# The number of total samples
print('The number of total samples : ', len(df))
# The median of number  of samples by ship_id
print('The median of number  of samples by ship_id :', 
      df.groupby('ship_id').apply(lambda s: len(s)).median())
# The min of number  of samples by ship_id
print('The min of number  of samples by ship_id :', 
      df.groupby('ship_id').apply(lambda s: len(s)).min())


# In[6]:


#1. Remove ship_id with too little records

# We want keep ship_id which contains at least about 6 months' record.
select_id = (df
             .groupby('ship_id')
             .apply(lambda s:len(s) >= 1500)
             .loc[lambda s: s == True]
             .index
             .to_list())

###### Check ship_id which will be removed
ship_id = df['ship_id'].unique()
dropped_id = set(ship_id) - set(select_id)
# According to the results we don not remove any ship_id.
print('The number of ship_id will be droppped : ', len(dropped_id))
print('The number of samples will be dropped : ', df.query('ship_id in @dropped_id').shape[0]) 


# In[7]:


#2. Remove duplicated records

# Check check if there are any duplicated records in each ship_id,
# The duplicated records is: For a `ship_id` and the same `client`, 
# if there are multiple records at the same `utc`, the records at that 
# 'utc' are duplicated records.
df.sort_values(by=['ship_id', 'client', 'utc'], inplace=True)
record = df.groupby(['ship_id', 'client', 'utc']).apply(lambda s: len(s))
print(record.value_counts())

print(f'Number of samples before : {len(df)}.')
df.drop_duplicates(subset=['ship_id', 'client', 'utc'], 
                   keep='first', inplace=True)
print(f'Number of samples after : {len(df)}.')


# In[8]:


#3. Remove missing values

# Chek missing values
cols = wind_columns + sea_columns + swell_columns + crnt_columns

missing_value_ = df[cols].apply(lambda col: np.sum(col >= np.float64(9000))) / len(df) * 100
missing_value_ = (pd
                 .DataFrame({'ratio of missing value(%)' : missing_value_})
                 .sort_values(by='ratio of missing value(%)'))
print(missing_value_)
print(f'Number of samples before : {len(df)}.')
df_isnan = (df[cols] >= np.float(9000)).sum(axis=1)
df = df.loc[df_isnan == 0, :]
df.reset_index(drop=True, inplace=True)
print('The number of samples after : ', len(df))


# In[9]:


# 4. Remove improper values

# Check degrees
deg_cols = ['course', 'sea_dir', 'swl_dir']
print(df.loc[:, deg_cols].apply(lambda col: col.max(), axis=0))
print('----------')
print(df.loc[:, deg_cols].apply(lambda col: col.min(), axis=0))
# Remove minus degrees
print(f'Number of samples before : {len(df)}.')
df = (df
      .loc[df['sea_dir'] > 0., :]
      .loc[df['swl_dir'] > 0., :]
      )
print('The number of samples after : ', len(df))


# In[10]:


# 4. Remove improper values
# Check performace columns, speed_columns and draft columns.
df.reset_index(drop=True, inplace=True)
df_ = pd.merge(left=df, right=db_bulk_carrier[[
               'ship_id', 'draft', 'speed_at_mcr', 'rpm_at_mcr', 'power_at_mcr']], on='ship_id', how='left')

print('The number of records with improper og_speed : ', len(df_.loc[(df_['og_speed'] <= 0) | (df_['og_speed'] > df_['speed_at_mcr'])]))
print('The number of records with improper rpm : ', len(df_.loc[(df_['rpm'] <= 0) | (df_['rpm'] > df_['rpm_at_mcr'])]))
print('The number of records with improper draft_aft : ', len(df_.loc[(df_['draft_aft'] < 0) | (df_['draft_aft'] >= 1.5 * df_['draft'])]))
print('The number of records with improper draft_fore : ', len(df_.loc[(df_['draft_fore'] < 0) | (df_['draft_fore'] >= 1.5 * df_['draft'])]))

# Remove improper og_speed, rpm, draft_aft and draft_fore.
print(f'Number of samples before : {len(df_)}.')
df_ = (df_
       .query('0<og_speed<speed_at_mcr')
       .query('0<rpm<rpm_at_mcr')
       .loc[lambda df: df['draft_aft']< 1.5 * df['draft'], :]
       .loc[lambda df: df['draft_fore']< 1.5 * df['draft'], :]
       .loc[lambda df: df['foc_me'] > 0]
      )
print('The number of samples after : ', len(df_))
del df
df = df_


# In[11]:


#5. Remove ship_id with too little records again

# We want keep ship_id which contains at least about 6 months' record.
select_id = (df
             .groupby('ship_id')
             .apply(lambda s:len(s) >= 1500)
             .loc[lambda s: s == True]
             .index
             .to_list())

###### Check ship_id which will be removed
ship_id = df['ship_id'].unique()
dropped_id = set(ship_id) - set(select_id)
# According to the results we don not remove any ship_id.
print('The number of ship_id will be droppped : ', len(dropped_id))
print('The number of samples will be dropped : ', df.query('ship_id in @dropped_id').shape[0])

df.query('ship_id in @select_id', inplace=True)


# In[12]:


# 5. Remove outliers of og_speed, rpm and foc_me
df.reset_index(drop=True, inplace=True)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
sns.distplot(a=df.loc[:, 'og_speed'], ax=axes[0])
sns.distplot(a=df.loc[:, 'rpm'], ax=axes[1])
sns.distplot(a=df.loc[:, 'foc_me'], ax=axes[2])


# In[13]:


from pyod.models.knn import KNN
from pyod.models.pca import PCA

def KNN_outlier(array):
    
    clf = KNN(n_neighbors = min(100, int(array.shape[0] * .1)), contamination=.1)
    clf.fit(array)
    return clf.decision_scores_, clf.labels_

def PCA_outlier(array):
    
    clf = PCA(n_components=2, contamination=.1, )
    clf.fit(array)
    return clf.decision_scores_, clf.labels_


# In[14]:


df.loc[:, 'norm_rpm'] = df['rpm'] / df['rpm_at_mcr']
df.loc[:, 'norm_foc'] = df['foc_me'] / (df['power_at_mcr'] * 0.745699872)
df.loc[:, 'norm_rpm**3'] = df['norm_rpm'] ** 3
# df.loc[:, 'rpm**3'] = df['rpm'] ** 3


# In[15]:


df.set_index('ship_id', inplace=True)
df.loc[:, 'outlier_score'] = 0.
df.loc[:, 'outlier_label'] = 0.
for idx in np.unique(df.index):
#     print(len(df.loc[idx, ['norm_rpm', 'norm_foc']].values))
    scores, labels =         KNN_outlier(df.loc[idx, ['norm_rpm**3', 'norm_foc']].values)
    df.loc[idx, 'outlier_score'] = scores
    df.loc[idx, 'outlier_label'] = labels
df.reset_index(drop=False, inplace=True)


# In[16]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df['outlier_score'], ax=ax, kde=False)

print('The max of outlier score : ', df['outlier_score'].max())
print('The min of outlier score : ', df['outlier_score'].min())

# fig, ax = plt.subplots(figsize=(8, 8))
# sns.distplot(a=df.loc[df['outlier_score']> 1, 'outlier_score'], ax=ax, kde=False)

print(df['outlier_score'].value_counts())


# In[17]:


fig, axes = plt.subplots(1, 3, figsize=(24, 8))
sns.distplot(a=df.loc[lambda df: df['outlier_label'] == 0., 'og_speed'], ax=axes[0], kde=True)
sns.distplot(a=df.loc[lambda df: df['outlier_label'] == 0., 'rpm'], ax=axes[1], kde=True)
sns.distplot(a=df.loc[lambda df: df['outlier_label'] == 0., 'foc_me'], ax=axes[2], kde=True)


# In[18]:


fig, axes = plt.subplots(1, 3, figsize=(24, 8))
sns.distplot(a=df.loc[lambda df: df['outlier_label'] == 1., 'og_speed'], ax=axes[0], kde=True)
sns.distplot(a=df.loc[lambda df: df['outlier_label'] == 1., 'rpm'], ax=axes[1], kde=True)
sns.distplot(a=df.loc[lambda df: df['outlier_label'] == 1., 'foc_me'], ax=axes[2], kde=True)


# In[19]:


print(df['outlier_label'].value_counts() / len(df))


# In[20]:


print('The number of samples before : ', len(df))
df = (df
      .query('outlier_label == 0.')
      .drop(['norm_rpm', 'norm_foc', 'norm_rpm**3', 'outlier_score', 'outlier_label'], axis=1))
print('The number of samples after : ', len(df))


# In[21]:


df.to_csv(data_path / 'processed/df_bulk_remove_outliers.csv')


# ###### BULK CARRIER continous records

# In[22]:


print('The number of records : ', len(df))


# In[23]:


df.reset_index(drop=False, inplace=True)
if  'index' in df.columns:
    df.drop('index', axis=1, inplace=True)


# In[24]:


df_utc = df[['ship_id', 'utc', 'client']]

#向前差分
s1 = df_utc.groupby(['ship_id', 'client'])['utc'].apply(lambda s: s.diff())
s1 = [s.total_seconds() / 3600 for s in s1]
#向后差分
s2 = df_utc.groupby(['ship_id', 'client'])['utc'].apply(lambda s: s.diff(-1))
s2 = [np.abs(s.total_seconds()) / 3600 for s in s2]

df_utc.loc[:, 's1'] = s1
df_utc.loc[:, 's2'] = s2

df_utc = df_utc.fillna(100)


# In[25]:


# fig, ax = plt.subplots(figsize=(8, 8))
# df_utc['s1'].value_counts().iloc[:20].plot(kind='bar', ax=ax)

# fig, ax = plt.subplots(figsize=(8, 8))
# df_utc['s2'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# In[26]:


# Classify each data points
df_utc.loc[:, 'isolated'] = (df_utc['s1'] > 6) & (df_utc['s2'] > 6)
df_utc.loc[:, 'start'] = (df_utc['s1'] > 6) & (df_utc['s2'] <= 6)
df_utc.loc[:, 'end'] = (df_utc['s1'] <= 6) & (df_utc['s2'] > 6)
df_utc.loc[:, 'inner'] = (df_utc['s1'] <= 6) & (df_utc['s2'] <= 6)

print('The ratio of inner : ', len(df_utc.query('inner == True')) / len(df_utc))
print('The ratio of isolated : ', len(df_utc.query('isolated == True')) / len(df_utc))


# In[ ]:





# In[31]:


df = pd.merge(left=df, right=df_utc.drop(['ship_id', 'utc', 'client'], axis=1), 
              left_index=True, right_index=True)

df.query('isolated == False', inplace=True)
df.reset_index(drop=True, inplace=True)


# In[32]:


# Calculate the length of continous records.
start_idx = df.query('start == True').index
end_idx = df.query('end == True').index

record_length = np.zeros(max(end_idx)+1)
record_index = np.zeros(max(end_idx)+1, dtype=np.int64)
for idx, (i, j) in enumerate(zip(start_idx, end_idx)):
    record_length[i:j+1] = j - i + 1
    record_index[i:j+1] = idx + 1

df.loc[:, 'record_length'] = record_length
df.loc[:, 'record_index'] = record_index


# In[30]:


# length = end_idx - start_idx
# fig, ax = plt.subplots(figsize=(8, 8))
# sns.distplot(a=length,
#              kde=False, norm_hist=False, bins=100, label='length')
# _ = ax.set_title('histogram of length of continus record')
# fig.legend()


# In[37]:


# We need to remove too short continous records.
print('The number of records before : ', len(df))
df_ = df.query('record_length >= 56')
print('The number of records after : ', len(df_))
print('The ratio of dropped records : ', len(df_) / len(df))
del df
df = df_


# In[38]:


# In each continous record, we delete records with too small foc_me

# diff = df.groupby('record_index')['foc_me'].apply(
#     lambda s: s.diff(-1) / (s + 1))
foc_too_small = df.groupby('record_index')['foc_me'].apply(
    lambda s: s < .15 * np.max(s))

# df.loc[:, 'foc_diff'] = np.abs(diff.fillna(0))
df.loc[:, 'foc_too_small'] = foc_too_small

print(df['foc_too_small'].value_counts())

df_ = df.query('foc_too_small == True')
print('The ratio of inner : ', len(df_.query('inner == True')) / len(df_))
print('The ratio of start : ', len(df_.query('start == True')) / len(df_))
print('The ratio of end : ', len(df_.query('end == True')) / len(df_))

dropped_index = df_['record_index'].unique()
print(len(df.loc[df['record_index'].isin(dropped_index), :]) / len(df))

print('The number of records before : ', len(df))
df = df.loc[~df['record_index'].isin(dropped_index), :]
print('The number of records after : ', len(df))


# In[ ]:


from tqdm import tqdm_notebook
record_index = df['record_index'].unique()
for idx in tqdm_notebook(record_index):
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='utc', y='foc_me', data=df.loc[df['record_index'] == idx, :])
    fig.savefig(data_path / f'processed/fig_2/{idx}')


# In[ ]:


df.to_csv(data_path / 'processed/df_bulk_remove_outliers_and_.csv')

