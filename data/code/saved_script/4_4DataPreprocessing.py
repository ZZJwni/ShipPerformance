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


# In[2]:


df = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
                 index_col=0, parse_dates=['utc'], 
                 dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                      'draft_aft', 'draft_fore', 'foc_me',
                                      'lat', 'lon', 'og_speed', 'pri_dir',
                                      'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                      'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                      'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                      'swl_per', 'wind_u', 'wind_v'], np.float32))

ship_ids = df['ship_id'].unique()


# In[12]:


ship_db = pd.read_excel(io=data_path / 'raw/shipdb.xlsx', sheet_name=3, 
                        usecols=['wni_ship_num', 'ship_type', 'date_built_year', 
                                 'length', 'breadth','depth', 'draft', 'dwt',
                                 'disp', 'gt', 'teu', 'power_at_mcr',
                                 'speed_at_mcr', 'rpm_at_mcr'])

# ship_db_2 = pd.read_excel(io=data_path / 'raw/AIS_shipdblist_shipdb.xlsx', sheet_name=0, 
#                           usecols=['wni_ship_num', 'ship_type', 
#                                    'length', 'breadth','depth', 'draft', 'dwt',
#                                    'gt', 'teu', 
#                                    'speed_at_mcr', 'rpm_at_mcr'])

# ship_db_2 = ship_db_2.merge(right=ship_db_1[['wni_ship_num', 'date_built_year', 'disp', 'power_at_mcr']], on='wni_ship_num', how='left', copy=False)
# ship_db_1 = ship_db_1.loc[ship_db_1['wni_ship_num'].isin(set(ship_ids) - set(ship_db_2['wni_ship_num'])), :]


# ##### Check the data

# ###### A first look of data

# In[3]:


df.head()


# In[8]:


df.info()


# ###### Check columns

# In[4]:


wave_columns = ['pri_ht', 'pri_dir', 'pri_per',
                'sec_ht', 'sec_dir', 'sec_per']
draft_columns = ['draft_aft', 'draft_fore', 'deadweight']
foc_columns = ['foc_me', 'rpm']
speed_columns = ['og_speed', 'course']
crnt_columns = ['crnt_u', 'crnt_v']
sea_columns = ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']
swell_columns = ['swl_dir', 'swl_ht', 'swl_per']
wind_columns = ['wind_u', 'wind_v']
position_columns = ['lat', 'lon']

remaining_columns = set(df.columns) - set(wave_columns) -set(draft_columns) -set(foc_columns) -set(speed_columns) - set(crnt_columns) - set(sea_columns) - set(swell_columns) - set(wind_columns) - set(position_columns)
assert remaining_columns == {'ship_id', 'utc', 'client'}


# In[5]:


df.reset_index(inplace=True, drop=True)


# ###### Check number of samples per ship_id

# In[11]:


# The number of total samples
print('The number of total samples : ', len(df))
# The average of samples by ship_id
print('The averge of samples per ship :', 
      df.groupby('ship_id').apply(lambda s: len(s)).mean())
# print('The 10% percentile of samples per ship :', 
#       np.percentile(df.groupby('ship_id').apply(lambda s: len(s)), .1))


# ###### Figure 1 : The distribuction of n_samples by ship_id

# In[12]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# #### Data Cleaning

# ##### Remove ship_id which contains too little samples

# In[6]:


ship_id = list(df['ship_id'].unique())

# We want keep ship_id which contains at least about 6 months' record.
select_id = (df
             .groupby('ship_id')
             .apply(lambda s:len(s) >= 1500)
             .loc[lambda s: s == True]
             .index
             .to_list())

###### Check ship_id which will be removed

dropped_id = set(ship_id) - set(select_id)
print('The number of ship_id will be droppped : ', len(dropped_id))
print('The number of samples will be dropped : ', df.query('ship_id in @dropped_id').shape[0])


# ###### Drop the above samples

# In[7]:


df.query('ship_id in @select_id', inplace=True)


# ##### Check client 

# In[19]:


client_per_ship = df.groupby('ship_id')['client'].apply(lambda s: len(np.unique(s)))
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=client_per_ship, kde=False, norm_hist=False)
_ = ax.set_title('distribution of number of client per ship_id')


# ###### check if there are any duplicated samples per ship_id per client

# In[8]:


df.sort_values(by=['ship_id', 'client', 'utc'], inplace=True)
record = df.groupby(['ship_id', 'client', 'utc']).apply(lambda s: len(s))

print(record.value_counts())


# ###### Figure 2  : The distribution of number of record per ship per client per utc

# In[16]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=record, kde=False, norm_hist=False)
_ = ax.set_title('distribution of number of record per ship per client per utc')


# ###### drop duplicates

# In[9]:


print(f'Number of samples before : {len(df)}.')

df.drop_duplicates(subset=['ship_id', 'client', 'utc'], 
                   keep='first', inplace=True)

print(f'Number of samples after : {len(df)}.')


# ##### Check missing values

# In[10]:


cols = wind_columns + sea_columns + swell_columns + crnt_columns + wave_columns

missing_value = df[cols].apply(lambda col: np.sum(col == np.float64(9999))) / len(df) * 100
missing_value = (pd
                 .DataFrame({'ratio of missing value(%)' : missing_value})
                 .sort_values(by='ratio of missing value(%)'))


# In[11]:


print(missing_value)


# ######  Missing value ratio

# In[28]:


fig, ax = plt.subplots(figsize=(8, 8))
missing_value.plot(kind='bar', ax=ax)


# ###### drop missing values

# In[12]:


df_isnan = (df[cols] == np.float(9999)).sum(axis=1)
df.loc[:, 'is_nan'] = df_isnan

df.query('is_nan == 0', inplace=True)
df.drop('is_nan', axis=1, inplace=True)

print('The number of samples : ', len(df))
df.reset_index(drop=True, inplace=True)


# ###### Figure 3: The distribution of number of samples per ship_id

# In[17]:


df_raw = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))

fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
sns.distplot(a=df.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id (after remove duplicated samples and missing values)')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# #### Outliers

# ##### modify degrees

# In[13]:


# change dir from [0, 2*pi] to [0, pi]
deg_cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']

print(df.loc[:, deg_cols].apply(lambda col: col.max(), axis=0))
print('----------')
print(df.loc[:, deg_cols].apply(lambda col: col.min(), axis=0))


# ###### Remove minus degress

# In[14]:


deg_cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']
selected_idx = ((df[deg_cols] >= 0).sum(axis=1) == 5)

print('The ratio of minus degress : ',  selected_idx.value_counts()[False]/ len(selected_idx) * 100)


# In[15]:


df.drop(selected_idx.loc[lambda s: s==False].index, axis=0, inplace=True)

# df = (df
#             .loc[selected_idx.values, :]
#             .assign(course=lambda df : df['course'] / 360 * 2 * np.pi,
#                     pri_dir=lambda df : df['pri_dir'] / 360 * 2 * np.pi,
#                     sec_dir=lambda df : df['sec_dir'] / 360 * 2 * np.pi,
#                     sea_dir=lambda df : df['sea_dir'] / 360 * 2 * np.pi,
#                     swl_dir=lambda df : df['swl_dir'] / 360 * 2 * np.pi,))

print('The number of samples : ', len(df))
df.reset_index(drop=True, inplace=True)


# In[16]:


df.to_csv(data_path / 'processed/new_1/df_without_na.csv')


# ##### Check outliers in rpm, foc_me and og_speed

# In[17]:


from pyod.models.knn import KNN 

def outlier_score(array):
    
    clf = KNN(n_neighbors = min(100, int(array.shape[0] * .1)), contamination=.06)
    clf.fit(array)
    return clf.decision_scores_


# In[18]:


df.set_index('ship_id', inplace=True)


# In[21]:


df.loc[:, 'outlier_score'] = 0.
for idx in np.unique(df.index):
    df.loc[idx, 'outlier_score'] =         outlier_score(df.loc[idx, ['og_speed', 'rpm', 'foc_me']].values)


# In[22]:


print(df['outlier_score'].value_counts())


# In[20]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df['outlier_score'], ax=ax, kde=True)


# In[23]:


print('The ratio of speed outliers : ', np.sum(df['outlier_score'] >=1) / len(df))
print('The ratio of speed outliers : ', np.sum(df['outlier_score'] >=3) / len(df))


# In[22]:


fig, axes = plt.subplots(1, 3, figsize=(24, 8))
sns.distplot(a=df.loc[lambda df: df['outlier_score'] <3, 'og_speed'], ax=axes[0], kde=True)
sns.distplot(a=df.loc[lambda df: df['outlier_score'] <3, 'rpm'], ax=axes[1], kde=True)
sns.distplot(a=df.loc[lambda df: df['outlier_score'] <3, 'foc_me'], ax=axes[2], kde=True)


# In[25]:


df.reset_index(drop=False, inplace=True)


# In[26]:


df = df.merge(right=ship_db[['wni_ship_num', 'speed_at_mcr', 'rpm_at_mcr']], 
              how='left', left_on='ship_id', right_on='wni_ship_num', copy=False)


# ###### drop outliers

# In[28]:


print('The number of samples : ', len(df))

df = (df
      .loc[lambda df: df['outlier_score'] < 3., :]
      .loc[lambda df: df['og_speed'] >= .15 * df['speed_at_mcr'] , :]
      .loc[lambda df: df['rpm'] >= .15 * df['rpm_at_mcr'] , :]
      .loc[lambda df: df['foc_me'] > 0., :])


print('The number of samples : ', len(df))


# In[28]:


fig, axes = plt.subplots(1, 3, figsize=(24, 8))
sns.distplot(a=df['og_speed'], ax=axes[0], kde=True)
sns.distplot(a=df['rpm'], ax=axes[1], kde=True)
sns.distplot(a=df['foc_me'], ax=axes[2], kde=True)


# In[29]:


for col in ['index', 'speed_at_mcr', 'rpm_at_mcr']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


# In[ ]:


df.to_csv(data_path / 'processed/new_1/df.csv')


# In[3]:


df = pd.read_csv(filepath_or_buffer=data_path / 'processed/new_1/df.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))


# In[32]:


df_raw = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))

fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
sns.distplot(a=df.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id (after remove duplicated samples, missing values and outliers)')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# #### Continous records

# In[5]:


df.reset_index(drop=False, inplace=True)
if  'index' in df.columns:
    df.drop('index', axis=1, inplace=True)


# In[6]:


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


# ###### Figure 4 : utc前后差分的分布图

# In[37]:


fig, ax = plt.subplots(figsize=(8, 8))
df_utc['s1'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# In[38]:


fig, ax = plt.subplots(figsize=(8, 8))
df_utc['s2'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# ###### 给每个时间点分类

# In[7]:


df_utc.loc[:, 'isolated'] = (df_utc['s1'] > 6) & (df_utc['s2'] > 6)
df_utc.loc[:, 'start'] = (df_utc['s1'] > 6) & (df_utc['s2'] <= 6)
df_utc.loc[:, 'end'] = (df_utc['s1'] <= 6) & (df_utc['s2'] > 6)
df_utc.loc[:, 'inner'] = (df_utc['s1'] <= 6) & (df_utc['s2'] <= 6)


# In[8]:


# point_stats = df_utc.groupby('ship_id')[['alone', 'start', 'end', 'inner']].apply(lambda x: np.sum(x))


# In[9]:


df = df.merge(right=df_utc.drop(['ship_id', 'utc', 'client'], axis=1), 
              left_index=True, right_index=True, copy=False)

df.query('isolated == False', inplace=True)
df.reset_index(drop=True, inplace=True)


# In[10]:


start_idx = df.query('start == True').index
end_idx = df.query('end == True').index

record_length = np.zeros(max(end_idx)+1)
record_index = np.zeros(max(end_idx)+1, dtype=np.int64)
for idx, (i, j) in enumerate(zip(start_idx, end_idx)):
    record_length[i:j+1] = j - i + 1
    record_index[i:j+1] = idx + 1

df.loc[:, 'record_length'] = record_length
df.loc[:, 'record_index'] = record_index


# ###### Figure 5: The distribution of length

# In[43]:


length = end_idx - start_idx
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=length,
             kde=False, norm_hist=False, bins=100, label='length')
_ = ax.set_title('histogram of length of continus record')
fig.legend()


# ######  去掉变化太大的foc_me

# In[11]:


df.query('record_length >= 56', inplace=True)

diff = df.groupby('record_index')['foc_me'].apply(
    lambda s: s.diff(-1) / (s + 1))
is2small = df.groupby('record_index')['foc_me'].apply(
    lambda s: s < .15 * np.max(s))

df.loc[:, 'foc_diff'] = np.abs(diff.fillna(0))
df.loc[:, 'is_too_small'] = is2small


# In[12]:


df[['record_length', 'record_index']].sort_values(by='record_length', ascending=False)


# In[47]:


df['is_too_small'].value_counts()


# In[39]:


df['foc_diff'].sort_values(ascending=False)


# In[49]:


df.loc[lambda df: df['is_too_small'] == False, ['foc_diff',
                                                'record_index']].sort_values(by='foc_diff', ascending=False)


# In[52]:


fig, ax = plt.subplots(figsize=(8, 8))

sns.distplot(a=df.loc[lambda df: (df['is_too_small'] == False) & (df['foc_diff'] > 0.), 'foc_diff'],
             ax=ax)


# In[13]:


print('The ratio of dropped record : ',
      len(df.loc[lambda df: (df['is_too_small'] == True) | (df['foc_diff'] > 2), 'record_index'].unique())\
      / len(df['record_index'].unique()))


# In[17]:


print('The ratio of dropped ship_id : ',
      len(df.loc[lambda df: (df['is_too_small'] == True) | (df['foc_diff'] > 2), 'ship_id'].unique())\
      / len(df['ship_id'].unique()))


# In[22]:


df.loc[:, 'is_foc_ok'] = True
dropped_index = df.loc[lambda df: (df['is_too_small'] == True) | (df['foc_diff'] > 2), 'record_index'].unique()

df.loc[lambda df: df['record_index'].isin(dropped_index), 'is_foc_ok'] = False


# In[23]:


df.to_csv(data_path / 'processed/new_1/df_no_outliers.csv')


# In[72]:


from tqdm import tqdm_notebook
for idx in tqdm_notebook(dropped_index):
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='utc', y='foc_me', data=df.loc[df['record_index'] == idx, :])
    fig.savefig(data_path / f'processed/fig/{idx}')


# In[62]:


df = pd.read_csv(filepath_or_buffer=data_path / 'processed/new_1/df_no_outliers.csv', 
                       index_col=0, 
                       parse_dates=['utc'], 
                       dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                            'draft_aft', 'draft_fore', 'foc_me',
                                            'lat', 'lon', 'og_speed', 'pri_dir',
                                            'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                            'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                            'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',]))
     

df_raw = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))

fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
sns.distplot(a=df_mod_6.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id (after remove duplicated samples, missing values, outliers)')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# In[24]:


print('The ratio of dropped row : ', len(df.loc[lambda df: df['is_foc_ok'] == False, :]) / len(df))


# In[25]:


df_ = (df.loc[lambda df: df['is_foc_ok']==True, ['record_length', 'record_index']]
       .sort_values(by='record_length', ascending=False)
       .drop_duplicates(subset=['record_index']))


# In[26]:


print(df_.head())


# In[30]:


fig, axes = plt.subplots(5, 1, figsize=(16, 8*5))
for i, idx in enumerate(df_.head()['record_index']):
    sns.lineplot(x='utc', y='foc_me', data=df.loc[df['record_index'] == idx, :], ax=axes[i])


# In[29]:


print(df_.iloc[6995:7000])


# In[32]:


fig, axes = plt.subplots(5, 1, figsize=(16, 8*5))
for i, idx in enumerate(df_.iloc[6995:7000]['record_index']):
    sns.lineplot(x='utc', y='foc_me', data=df.loc[df['record_index'] == idx, :], ax=axes[i])


# In[27]:


print(df_.tail())


# In[33]:


fig, axes = plt.subplots(5, 1, figsize=(16, 8*5))
for i, idx in enumerate(df_.tail()['record_index']):
    sns.lineplot(x='utc', y='foc_me', data=df.loc[df['record_index'] == idx, :], ax=axes[i])


# #### ship db

# ###### 问题: 怎么划分ship?
# 1. 按照建造时间
# 2. 按照尺寸
# 3. 按照类型
# 4. 按照引擎性能

# In[3]:


df = pd.read_csv(filepath_or_buffer=data_path / 'processed/new_1/df_no_outliers.csv', 
                       index_col=0, 
                       parse_dates=['utc'], 
                       dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                            'draft_aft', 'draft_fore', 'foc_me',
                                            'lat', 'lon', 'og_speed', 'pri_dir',
                                            'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                            'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                            'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',]))
     


# In[4]:


ship_ids = df.loc[lambda df: df['is_foc_ok'] == True, 'ship_id'].unique()
ship_db = ship_db.rename(columns={'wni_ship_num' : 'ship_id'})
ship_db = ship_db.loc[lambda df: df['ship_id'].isin(ship_ids), :]
ship_db = ship_db.set_index('ship_id')


# In[ ]:


ship_db


# In[6]:


ship_db.drop(['ship_type', 'date_built_year'], axis=1).apply(lambda col: np.sum(np.isnan(col))) / len(ship_db)


# In[5]:


ship_db.loc[:, 'n_samples'] = df.loc[lambda df: df['is_foc_ok'] == True, :].groupby('ship_id').apply(lambda s: len(s))


# In[10]:


# ship_db.loc[ship_db.drop(['dwt', 'disp', 'gt', 'teu',], axis=1).isna().sum(axis=1) != 0, :]


# In[6]:


print(ship_db['ship_type'].value_counts())


# ###### BULK CARRIER

# In[7]:


X = (ship_db
     .loc[lambda df: df['ship_type'] == 'BULK CARRIER', :]
     .drop(['n_samples', 'ship_type', 'teu', 'disp'], axis=1)
     .dropna()
     .assign(date_built_year=lambda df: df['date_built_year'].astype(int) - 2000)
     .astype(np.float32))


# In[8]:


X.drop(['date_built_year'], axis=1).apply(lambda col: np.sum(np.isnan(col))) / len(ship_db)


# In[9]:


from sklearn.cluster import DBSCAN, KMeans
clustering = KMeans(n_clusters=5)


# In[10]:


clustering.fit(X)


# In[11]:


pd.Series(clustering.labels_).value_counts()


# In[12]:


X.loc[:, 'label'] = clustering.labels_


# In[134]:


# g = sns.PairGrid(data=X, height=4, aspect=1, hue='label')
# g.map_diag(plt.hist)
# g.map_offdiag(plt.scatter)


# In[ ]:


from itertools import combinations
cols = ['length', 'breadth', 'depth', 'draft', 'gt', 'dwt', 'rpm_at_mcr',
        'power_at_mcr']
iters = list(combinations(cols, 2))
fig, axes = plt.subplots(len(iters),1,  figsize=(8, len(iters) * 8))
cmap = sns.color_palette(palette='muted', n_colors=5)
for i, (x, y) in enumerate(iters):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=x, y=y, hue='label', data=X, ax=ax, palette=cmap)
    fig.savefig(data_path / f'processed/fig_1/{x}-{y}')        


# In[13]:


from itertools import combinations
cols = ['length', 'breadth', 'depth', 'draft', 'gt', 'dwt', 'rpm_at_mcr',
        'power_at_mcr']
iters = list(combinations(cols, 2))
fig, axes = plt.subplots(len(iters),1,  figsize=(8, len(iters) * 8))
cmap = sns.color_palette(palette='muted', n_colors=5)
for i, (x, y) in enumerate(iters):
    sns.scatterplot(x=x, y=y, hue='label', data=X, ax=axes[i], palette=cmap)
        


# In[137]:


X


# In[139]:


df.columns


# In[14]:


df_cols = ['ship_id', 'client', 'course', 'crnt_u', 'crnt_v', 
           'draft_aft', 'draft_fore', 'foc_me', 'lat', 'lon', 'og_speed',
           'pri_dir', 'pri_ht', 'pri_per', 'rpm', 'sea_dir', 'sea_ht', 'sea_per',
           'sec_dir', 'sec_ht', 'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
           'swl_per', 'utc', 'wind_u', 'wind_v','record_length',
           'record_index']

df_ = df.loc[lambda df: df['is_foc_ok'] == True, df_cols]


# In[15]:


X_ = X.reset_index()


# In[16]:


data = pd.merge(left=df_, right=X_, on='ship_id', how='inner')


# In[17]:


len(data) / len(df_)


# In[18]:


len(data)


# In[19]:


len(df_)


# In[21]:


data.to_csv(data_path / 'processed/record_bulk.csv')


# In[20]:


data.groupby('label').apply(lambda s: len(s))


# In[ ]:




