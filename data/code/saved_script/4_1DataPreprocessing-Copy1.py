#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas                  as     pd
import numpy                   as     np

# Visualization
import seaborn                 as     sns
import matplotlib.pyplot       as     plt
sns.set_style("whitegrid")

from pathlib import Path
ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# In[2]:


df_raw = pd.read_csv(data_path / 'processed/leg_top5.csv', index_col=0, parse_dates=['utc'])


# ###### leg.json

# In[3]:


df_raw.head()


# In[4]:


dropped_columns = ['client', 'point_type', 'loading_condition',
                   'deadweight', 'power', 'sst', 'foc_total', 'load']
wave_columns = ['pri_ht', 'pri_dir', 'pri_per',
                'sec_ht', 'sec_dir', 'sec_per']
draft_columns = ['draft_aft', 'draft_fore']
foc_columns = ['foc_me', 'rpm']
speed_columns = ['og_speed', 'course', 'tw_speed']
crnt_columns = ['crnt_u', 'crnt_v']
sea_columns = ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']
swell_columns = ['swl_dir', 'swl_ht', 'swl_per']
wind_columns = ['wind_u', 'wind_v']
position_columns = ['lat', 'lon']


# In[5]:


set(df_raw.columns) - set(dropped_columns) - set(wave_columns) -set(draft_columns) -set(foc_columns) -set(speed_columns) - set(crnt_columns) - set(sea_columns) - set(swell_columns) - set(wind_columns) - set(position_columns)


# In[6]:


df_raw[dropped_columns]


# In[7]:


df_raw['foc_total'].value_counts()


# In[8]:


df_mod = (df_raw
          .drop(dropped_columns, axis=1)
          .drop_duplicates(subset=['ship_id', 'utc'], keep='first')
          .set_index(['ship_id'])
          .sort_values(by=['ship_id', 'utc'])
          )
ship_ids = np.unique(df_mod.index)


# #### Time Series Visualization and Data Cleaning

# In[9]:


def line_plot(cols, data=df_mod):
    colors = sns.color_palette()[:len(cols)]
    fig, axes = plt.subplots(len(ship_ids), len(
        cols), figsize=(15*len(cols), 5*len(ship_ids)))
    for ship_id, axes_ in zip(ship_ids, axes):
        for col, ax, color in zip(cols, axes_, colors):
            sns.lineplot(x='utc', y=col, ax=ax,
                         data=data.loc[ship_id], color=color)
            ax.set_title(ship_id)


# ##### ship speed
# 1. tw_speed 需要计算
# 2. og_spped 异常值处理(速度为零的值)

# In[9]:


line_plot(speed_columns)


# ##### shaft performance

# 1. foc_me有异常值(消耗量为零)以及缺失值
# 2. rpm有异常值(转速为零)

# In[13]:


line_plot(foc_columns)


# ##### draft

# In[14]:


line_plot(draft_columns)


# ##### wind conditions

# wind_u和wind_v有缺失值

# In[15]:


line_plot(wind_columns)


# ##### sea conditions

# 缺失值

# In[16]:


line_plot(sea_columns)


# ##### swell conditions

# 缺失值

# In[18]:


line_plot(swell_columns)


# ##### crnt conditions

# 缺失值

# In[19]:


cols = crnt_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### wave conditions

# 缺失值

# In[20]:


line_plot(wave_columns)


# #### Data Cleaning

# In[10]:


df_mod['wind_u'].value_counts()


# In[11]:


cols = wind_columns + sea_columns + swell_columns + crnt_columns + wave_columns


# ##### 统计缺失值

# In[12]:


df_tmp = pd.DataFrame(df_mod[cols].apply(lambda col: np.sum(col == np.float64(9999))) / len(df_mod) * 100)
df_tmp.columns=['欠損値比率(%)']


# In[13]:


df_tmp


# ##### 移除缺失值

# In[14]:


def _check_nan(arr, nan=np.float64(9999)):
    for a in arr:
        if a == nan:
            return True
    else:
        return False


selected_idx = ~df_mod[cols].apply(lambda row: _check_nan(row), axis=1)

# Nearly 15% rows will be dropped
print(sum(selected_idx) / len(selected_idx))

df_mod_1 = df_mod.loc[selected_idx, :]


# In[15]:


for ship_id in ship_ids:
    print('ship id : ', ship_id)
    print(len(df_mod_1.loc[ship_id, :]) / len(df_mod.loc[ship_id, :]))


# ##### modify degrees

# In[16]:


# change dir from [0, 2*pi] to [0, pi]
cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']

print(df_mod_1.loc[:, cols].apply(lambda col: col.max(), axis=0))


# 我们将在5_FeatureEngineering中处理degree的问题

# In[17]:


# def modify_degrees(arr):
#     return np.where(arr>180, 360-arr, arr)

# df_mod_2 = df_mod_1.assign(**{
#     col: modify_degrees(df_mod_1[col].values) for col in cols})

# print(df_mod_2.loc[:, cols].apply(lambda col: col.max(), axis=0))

df_mod_2 = df_mod_1.copy()


# In[18]:


df_mod_2['sea_dir'].value_counts()


# ##### check outliers

# In[19]:


# check outliers

cols = ['og_speed', 'foc_me', 'rpm']
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(
    cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.distplot(a=df_mod_2.loc[ship_id, col], ax=ax, color=color)
        ax.set_title(ship_id)


# In[20]:


df_mod_3 = df_mod_2.copy()
df_mod_3 = df_mod_3.loc[(df_mod_3
                         .groupby('ship_id')['og_speed']
                         .apply(lambda s: (s >= s.quantile(.03)))
                         ), :]
df_mod_3 = df_mod_3.loc[(df_mod_3
                         .groupby('ship_id')['foc_me']
                         .apply(lambda s: (s >= s.quantile(.03)) & (s <= s.quantile(.97)))
                         ), :]
df_mod_3 = df_mod_3.loc[(df_mod_3
                         .groupby('ship_id')['rpm']
                         .apply(lambda s: (s >= s.quantile(.02)))
                         ), :]


# In[21]:


print(len(df_mod_3) / len(df_mod_2))
print(len(df_mod_3) / len(df_mod))


# In[33]:


cols = ['og_speed', 'foc_me', 'rpm']
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(
    cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.distplot(a=df_mod_3.loc[ship_id, col], ax=ax, color=color)
        ax.set_title(ship_id)


# #### Visualization with cleaned data

# In[34]:


line_plot(speed_columns, df_mod_3)


# In[35]:


line_plot(foc_columns, df_mod_3)


# In[36]:


line_plot(draft_columns, df_mod_3)


# In[37]:


line_plot(wind_columns, df_mod_3)


# In[38]:


line_plot(sea_columns, df_mod_3)


# In[39]:


line_plot(swell_columns, df_mod_3)


# In[40]:


line_plot(crnt_columns, df_mod_3)


# In[93]:


cols = wave_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[22]:


df_mod_3.to_csv(data_path / 'leg_top5_mod.csv')

