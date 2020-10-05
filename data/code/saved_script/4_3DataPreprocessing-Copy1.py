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


ship_db = pd.read_excel(data_path / 'raw/shipdb.xlsx', sheet_name=3)

ship_db = ship_db[['wni_ship_num', 'ship_type', 'date_built_year', 'length', 'breadth',
       'depth', 'draft', 'dwt', 'disp', 'gt', 'teu', 'power_at_mcr',
       'speed_at_mcr', 'rpm_at_mcr']]


# In[2]:


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


# ##### Check the data

# ###### A first look of data

# In[3]:


df_raw.head()


# In[4]:


df_raw.info()


# ###### check columns

# In[2]:


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


# In[7]:


remaining_columns = set(df_raw.columns) - set(wave_columns) -set(draft_columns) -set(foc_columns) -set(speed_columns) - set(crnt_columns) - set(sea_columns) - set(swell_columns) - set(wind_columns) - set(position_columns)
assert remaining_columns == {'ship_id', 'utc', 'client'}


# In[9]:


# reset index
df_raw.reset_index(inplace=True, drop=True)


# ###### check distribution of number of samples

# In[17]:


# The number of total samples
print('The number of total samples : ', len(df_raw))
# The average of samples by ship_id
print('The averge of samples per ship :', 
      df_raw.groupby('ship_id').apply(lambda s: len(s)).mean())
# print('The 10% percentile of samples per ship :', 
#       np.percentile(df_raw.groupby('ship_id').apply(lambda s: len(s)), .1))


# ###### Figure 1 : The distribuction of n_samples by ship_id

# In[13]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# #### Data Cleaning

# ##### Remove ship_id with too little samples: df_mod_1

# In[41]:


ship_id = list(df_raw['ship_id'].unique())


# In[52]:


# We want keep ship_id which contains at least about 6 months' record.
select_id = (df_raw
             .groupby('ship_id')
             .apply(lambda s:len(s) >= 1500)
             .loc[lambda s: s == True]
             .index
             .to_list())


# ###### Check ship_id which will be removed

# In[53]:


dropped_id = set(ship_id) - set(select_id)
print('The number of ship_id will be droppped : ', len(dropped_id))
print('The number of samples will be dropped : ', df_raw.query('ship_id in @dropped_id').shape[0])


# ###### Drop the above samples

# In[54]:


df_mod_1 = df_raw.query('ship_id in @select_id')


# ###### Save df_mod_1

# In[56]:


# Save df_mod_1
df_mod_1.to_csv(data_path / 'processed/new/df_mod_1.csv')


# In[3]:


# df_mod_1 = pd.read_csv(data_path / 'processed/new/df_mod_1.csv', 
#                        index_col=0, parse_dates=['utc'], 
#                        dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
#                                             'draft_aft', 'draft_fore', 'foc_me',
#                                             'lat', 'lon', 'og_speed', 'pri_dir',
#                                             'pri_ht', 'pri_per', 'rpm', 'sea_dir',
#                                             'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
#                                             'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
#                                             'swl_per', 'wind_u', 'wind_v'], np.float32))


# ##### Check client 

# In[9]:


client_per_ship = df_mod_1.groupby('ship_id')['client'].apply(lambda s: len(np.unique(s)))
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=client_per_ship, kde=False, norm_hist=False)
_ = ax.set_title('distribution of number of client per ship_id')


# ###### check if there are any duplicated samples per ship_id per client

# In[12]:


df_mod_1 = df_mod_1.sort_values(by=['ship_id', 'client', 'utc'])


# In[13]:


record = df_mod_1.groupby(['ship_id', 'client', 'utc']).apply(lambda s: len(s))


# ###### Figure 2  : The distribution of number of record per ship per client per utc

# In[16]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=record, kde=False, norm_hist=False)
_ = ax.set_title('distribution of number of record per ship per client per utc')


# ###### drop duplicates

# In[18]:


df_mod_2 = df_mod_1.drop_duplicates(
    subset=['ship_id', 'client', 'utc'], keep='first', inplace=False)


# In[21]:


print(
    f'Number of samples before : {len(df_mod_1)} ; Number of samples after : {len(df_mod_2)}.')
print(f'Ratio of  dropped samples : {1 - len(df_mod_2) / len(df_mod_1)}')


# ###### Save df_mod_2

# In[22]:


df_mod_2.to_csv(data_path / 'processed/new/df_mod_2.csv')


# In[23]:


del df_mod_1


# ##### Check missing values

# In[24]:


cols = wind_columns + sea_columns + swell_columns + crnt_columns + wave_columns


# In[26]:


missing_value = df_mod_2[cols].apply(lambda col: np.sum(col == np.float64(9999))) / len(df_mod_2) * 100
missing_value = (pd
                 .DataFrame({'ratio of missing value(%)' : missing_value})
                 .sort_values(by='ratio of missing value(%)'))


# ######  Missing value ratio

# In[28]:


fig, ax = plt.subplots(figsize=(8, 8))
missing_value.plot(kind='bar', ax=ax)


# ###### drop missing values

# In[31]:


df_isnan = (df_mod_2[cols] == np.float(9999)).sum(axis=1)


# In[34]:


df_mod_3 = df_mod_2.loc[df_isnan == 0, :]


# ###### Save df_mod_3

# In[36]:


df_mod_3.to_csv(data_path / 'processed/new/df_mod_3.csv')


# In[5]:


# del df_mod_2


# ###### Figure 3: The distribution of number of samples per ship_id

# In[7]:


# df_mod_3 = pd.read_csv(filepath_or_buffer=data_path / 'processed/new/df_mod_3.csv', 
#                        index_col=0, 
#                        parse_dates=['utc'], 
#                        dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
#                                             'draft_aft', 'draft_fore', 'foc_me',
#                                             'lat', 'lon', 'og_speed', 'pri_dir',
#                                             'pri_ht', 'pri_per', 'rpm', 'sea_dir',
#                                             'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
#                                             'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
#                                             'swl_per', 'wind_u', 'wind_v'], np.float32))

# df_raw = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
#                      index_col=0, 
#                      parse_dates=['utc'], 
#                      dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
#                                           'draft_aft', 'draft_fore', 'foc_me',
#                                           'lat', 'lon', 'og_speed', 'pri_dir',
#                                           'pri_ht', 'pri_per', 'rpm', 'sea_dir',
#                                           'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
#                                           'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
#                                           'swl_per', 'wind_u', 'wind_v'], np.float32))

fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
sns.distplot(a=df_mod_3.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id (after remove duplicated samples and missing values)')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# In[9]:


# del df_raw


# #### Outliers

# ##### modify degrees

# In[4]:


# change dir from [0, 2*pi] to [0, pi]
deg_cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']

print(df_mod_3.loc[:, deg_cols].apply(lambda col: col.max(), axis=0))
print('----------')
print(df_mod_3.loc[:, deg_cols].apply(lambda col: col.min(), axis=0))


# ###### Remove minus degress and convert deg to rad

# In[3]:


df_mod_3 = pd.read_csv(filepath_or_buffer=data_path / 'processed/new/df_mod_3.csv', 
                       index_col=0, 
                       parse_dates=['utc'], 
                       dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                            'draft_aft', 'draft_fore', 'foc_me',
                                            'lat', 'lon', 'og_speed', 'pri_dir',
                                            'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                            'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                            'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                            'swl_per', 'wind_u', 'wind_v'], np.float32))


# In[4]:


df_mod_3.reset_index(inplace=True)


# In[5]:


deg_cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']
selected_idx = ((df_mod_3[deg_cols] >= 0).sum(axis=1) == 5)

print('The ratio of minus degress : ',  selected_idx.value_counts()[False]/ len(selected_idx) * 100)


# In[6]:


df_mod_3 = (df_mod_3
            .loc[selected_idx.values, :]
            .assign(course=lambda df : df['course'] / 360 * 2 * np.pi,
                    pri_dir=lambda df : df['pri_dir'] / 360 * 2 * np.pi,
                    sec_dir=lambda df : df['sec_dir'] / 360 * 2 * np.pi,
                    sea_dir=lambda df : df['sea_dir'] / 360 * 2 * np.pi,
                    swl_dir=lambda df : df['swl_dir'] / 360 * 2 * np.pi,))


# ##### Check outliers in rpm, foc_me and og_speed

# In[7]:


from pyod.models.knn import KNN 

def outlier_scores(array):
    
    clf = KNN(n_neighbors = min(100, int(array.shape[0] * .1)), contamination=.06)
    clf.fit(array)
    return clf.decision_scores_

def is_outlier(array):
    
    clf = KNN(n_neighbors = min(100, int(array.shape[0] * .1)), contamination=.06)
    clf.fit(array)
    return clf.labels_


# In[8]:


df_mod_4 = pd.merge(left=df_mod_3, right=ship_db[['wni_ship_num','speed_at_mcr', 'rpm_at_mcr']],
                    left_on='ship_id', right_on='wni_ship_num',
                    how='left'
                   )


# In[9]:


df_mod_4.drop('wni_ship_num', axis=1, inplace=True)


# In[20]:


df_mod_4.set_index('ship_id', inplace=True)


# In[21]:


ship_id = np.unique(df_mod_4.index)


# In[22]:


result = dict()
for idx in ship_id:
    result[idx] =     outlier_scores(df_mod_4.loc[idx, ['og_speed', 'rpm', 'foc_me']].values)


# In[26]:


df_mod_4.loc[:, 'outlier_score'] = 0.


# In[27]:


for key in result:
    df_mod_4.loc[key, 'outlier_score'] = result[key]


# In[28]:


df_mod_4['outlier_score'].value_counts()


# In[30]:


print('The ratio of  outliers : ', np.sum(df_mod_4['outlier_score'] >=3) / len(df_mod_4))


# ###### og_speed

# In[10]:


df_mod_4.loc[:, 'speed_outlier_score'] = df_mod_4.groupby(
    'ship_id')['og_speed'].transform(lambda s: outlier_scores(s.values.reshape((-1, 1))))


# In[11]:


print(df_mod_4['speed_outlier_score'].value_counts())


# In[12]:


print('The ratio of speed outliers : ', np.sum(df_mod_4['speed_outlier_score'] >=1) / len(df_mod_4))


# In[31]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_4.loc[lambda df: df['outlier_score'] <3, 'og_speed'], ax=ax, kde=True)


# ###### rpm

# In[14]:


df_mod_4.loc[:, 'rpm_outlier_score'] = df_mod_4.groupby(
    'ship_id')['rpm'].transform(lambda s: outlier_scores(s.values.reshape((-1, 1))))


# In[15]:


print(df_mod_4['rpm_outlier_score'].value_counts())


# In[16]:


print('The ratio of rpm outliers : ', np.sum(df_mod_4['rpm_outlier_score'] >=1) / len(df_mod_4))


# In[17]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_4.loc[lambda df: df['rpm_outlier_score'] <1, 'rpm'], ax=ax, kde=True)


# ###### foc_me

# In[18]:


df_mod_4.loc[:, 'foc_outlier_score'] = df_mod_4.groupby(
    'ship_id')['foc_me'].transform(lambda s: outlier_scores(s.values.reshape((-1, 1))))


# In[19]:


print(df_mod_4['foc_outlier_score'].value_counts())


# In[20]:


print('The foc of foc outliers : ', np.sum(df_mod_4['foc_outlier_score'] >=1) / len(df_mod_4))


# In[21]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_4.loc[lambda df: df['foc_outlier_score'] <1, 'foc_me'], ax=ax, kde=True)


# ###### Remove outliers and save as df_mod_5 and Handle small values in og_speed, rpm, foc_me

# In[35]:


df_mod_5 = (df_mod_4
            .query('speed_outlier_score < 1')
            .query('rpm_outlier_score < 1')
            .query('foc_outlier_score < 1')
           )

df_mod_5 = (df_mod_5
            .loc[lambda df: df['og_speed'] >= .15 * df['speed_at_mcr'] , :]
            .loc[lambda df: df['rpm'] >= .15 * df['rpm_at_mcr'] , :]
            .loc[lambda df: df['foc_me'] > 0., :])


# In[36]:


print('The ratio of outliers : ', 1 - len(df_mod_5) / len(df_mod_4))


# In[37]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_5['og_speed'], ax=ax, kde=True)


# In[38]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_5['rpm'], ax=ax, kde=True)


# In[39]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_5['foc_me'], ax=ax, kde=True)


# In[41]:


for col in ['index', 'speed_outlier_score', 
            'rpm_outlier_score','foc_outlier_score']:
    if col in df_mod_5.columns:
        df_mod_5.drop(col, axis=1, inplace=True)
df_mod_5.to_csv(data_path / 'processed/new/df_mod_5.csv')


# In[43]:


# df_mod_5 = pd.read_csv(filepath_or_buffer=data_path / 'processed/new/df_mod_5.csv', 
#                        index_col=0, 
#                        parse_dates=['utc'], 
#                        dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
#                                             'draft_aft', 'draft_fore', 'foc_me',
#                                             'lat', 'lon', 'og_speed', 'pri_dir',
#                                             'pri_ht', 'pri_per', 'rpm', 'sea_dir',
#                                             'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
#                                             'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
#                                             'swl_per', 'wind_u', 'wind_v'], np.float32))

# df_raw = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
#                      index_col=0, 
#                      parse_dates=['utc'], 
#                      dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
#                                           'draft_aft', 'draft_fore', 'foc_me',
#                                           'lat', 'lon', 'og_speed', 'pri_dir',
#                                           'pri_ht', 'pri_per', 'rpm', 'sea_dir',
#                                           'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
#                                           'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
#                                           'swl_per', 'wind_u', 'wind_v'], np.float32))

fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id')
sns.distplot(a=df_mod_5.groupby('ship_id').apply(lambda s: len(s)),
             kde=False, norm_hist=False, bins=100, label='n_samples by ship_id (after remove duplicated samples, missing values and outliers)')
_ = ax.set_title('histogram of number of samples')
fig.legend()


# In[45]:


# del df_raw


# In[3]:


df_mod_5 = pd.read_csv(filepath_or_buffer=data_path / 'processed/new/df_mod_5.csv', 
                       index_col=0, 
                       parse_dates=['utc'], 
                       dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                            'draft_aft', 'draft_fore', 'foc_me',
                                            'lat', 'lon', 'og_speed', 'pri_dir',
                                            'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                            'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                            'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                            'swl_per', 'wind_u', 'wind_v'], np.float32))


# #### Continous records

# In[4]:


df_mod_5.reset_index(drop=False, inplace=True)
if  'index' in df_mod_5.columns:
    df_mod_5.drop('index', axis=1, inplace=True)


# In[5]:


df_utc = df_mod_5[['ship_id', 'utc', 'client']]

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

# In[50]:


fig, ax = plt.subplots(figsize=(8, 8))
df_utc['s1'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# In[51]:


fig, ax = plt.subplots(figsize=(8, 8))
df_utc['s2'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# ###### 给每个时间点分类

# In[6]:


df_utc.loc[:, 'alone'] = (df_utc['s1'] > 6) & (df_utc['s2'] > 6)
df_utc.loc[:, 'start'] = (df_utc['s1'] > 6) & (df_utc['s2'] <= 6)
df_utc.loc[:, 'end'] = (df_utc['s1'] <= 6) & (df_utc['s2'] > 6)
df_utc.loc[:, 'inner'] = (df_utc['s1'] <= 6) & (df_utc['s2'] <= 6)


# In[9]:


# point_stats = df_utc.groupby('ship_id')[['alone', 'start', 'end', 'inner']].apply(lambda x: np.sum(x))


# In[10]:


df_mod_6 = pd.merge(left=df_mod_5, right=df_utc.drop(['ship_id', 'utc', 'client'], axis=1), 
                    left_index=True, right_index=True)

df_mod_6 = df_mod_6.query('alone == False')
df_mod_6.reset_index(drop=True, inplace=True)


# In[11]:


start_idx = df_mod_6.query('start == True').index
end_idx = df_mod_6.query('end == True').index

record_length = np.zeros(max(end_idx)+1)
record_index = np.zeros(max(end_idx)+1, dtype=np.int64)
for idx, (i, j) in enumerate(zip(start_idx, end_idx)):
    record_length[i:j+1] = j - i + 1
    record_index[i:j+1] = idx + 1

df_mod_6.loc[:, 'record_length'] = record_length
df_mod_6.loc[:, 'record_index'] = record_index


# ###### Figure 5: The distribution of length

# In[59]:


length_unique = end_idx - start_idx
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=length_unique,
             kde=False, norm_hist=False, bins=100, label='length')
_ = ax.set_title('histogram of length of continus record')
fig.legend()


# ######  去掉变化太大的foc_me

# In[15]:


df_mod_6_ = df_mod_6.query('record_length >= 56')


# In[ ]:


df_mod_6_.groupby('record_index')


# ###### Save df_6

# In[61]:


df_mod_6.to_csv(data_path / 'processed/new/df_mod_6.csv')


# In[62]:


# df_mod_6 = pd.read_csv(filepath_or_buffer=data_path / 'processed/new/df_mod_6.csv', 
#                        index_col=0, 
#                        parse_dates=['utc'], 
#                        dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
#                                             'draft_aft', 'draft_fore', 'foc_me',
#                                             'lat', 'lon', 'og_speed', 'pri_dir',
#                                             'pri_ht', 'pri_per', 'rpm', 'sea_dir',
#                                             'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
#                                             'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
#                                             'swl_per', 'wind_u', 'wind_v'], np.float32))

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


# 

# #### ship db

# ###### 问题: 怎么划分ship?
# 1. 按照建造时间
# 2. 按照尺寸
# 3. 按照类型
# 4. 按照引擎性能

# In[63]:


ship_db

