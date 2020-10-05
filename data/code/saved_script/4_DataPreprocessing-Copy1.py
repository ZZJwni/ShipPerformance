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


# In[5]:


df_raw = pd.read_csv(data_path / 'processed/leg_top5.csv', index_col=0, parse_dates=['utc'])


# In[6]:


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


# In[ ]:


df_mod = (df_raw
          .drop(dropped_columns, axis=1)
          .drop_duplicates(subset=['ship_id', 'utc'], keep='first')
          .set_index(['ship_id'])
          .sort_values(by=['ship_id', 'utc'])
          )
ship_ids = np.unique(df_mod.index)


# #### Time Series Visualization

# ##### ship speed
# 1. og_speed 有的值太小
# 2. course变化幅度(?)
# 3. tw_speed 需要计算

# In[11]:


cols = speed_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### shaft performance

# 1. foc_me 有太大的值和太小的值
# 2. rpm 有太小的值

# In[12]:


cols = foc_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### draft

# In[13]:


cols = draft_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### wind conditions

# wind_u和wind_v有太大的值

# In[14]:


cols = wind_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### sea conditions

# 异常值

# In[16]:


cols = sea_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### swell conditions

# 异常值

# In[5]:


cols = swell_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### crnt conditions

# 异常值

# In[6]:


cols = crnt_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# ##### wave conditions

# 异常值

# In[7]:


cols = wave_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod.loc[ship_id], color=color)
        ax.set_title(ship_id)


# #### Data Cleaning

# ##### drop nan

# In[51]:


cols = wind_columns + sea_columns + swell_columns + crnt_columns + wave_columns


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


# ##### modify degrees

# In[95]:


# # change dir from [0, 2*pi] to [0, pi]
# cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']

# print(df_mod_1.loc[:, cols].apply(lambda col: col.max(), axis=0))


# def modify_degrees(arr):
#     return np.where(arr>180, 360-arr, arr)

# df_mod_2 = df_mod_1.assign(**{
#     col: modify_degrees(df_mod_1[col].values) for col in cols})

# print(df_mod_2.loc[:, cols].apply(lambda col: col.max(), axis=0))

df_mod_2 = df_mod_1.copy()


# ##### check outliers

# In[84]:


# check outliers

cols = ['og_speed', 'foc_me', 'rpm']
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(
    cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.distplot(a=df_mod_2.loc[ship_id, col], ax=ax, color=color)
        ax.set_title(ship_id)


# In[96]:


df_mod_3 = df_mod_2.copy()
df_mod_3 = df_mod_3.loc[(df_mod_3
                         .groupby('ship_id')['og_speed']
                         .apply(lambda s: (s >= s.quantile(.02)) & (s <= s.quantile(.98)))
                         ), :]
df_mod_3 = df_mod_3.loc[(df_mod_3
                         .groupby('ship_id')['foc_me']
                         .apply(lambda s: (s >= s.quantile(.02)) & (s <= s.quantile(.98)))
                         ), :]
df_mod_3 = df_mod_3.loc[(df_mod_3
                         .groupby('ship_id')['rpm']
                         .apply(lambda s: (s >= s.quantile(.02)) & (s <= s.quantile(.98)))
                         ), :]

print(len(df_mod_3) / len(df_mod_2))


# In[85]:


cols = ['og_speed', 'foc_me', 'rpm']
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(
    cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.distplot(a=df_mod_3.loc[ship_id, col], ax=ax, color=color)
        ax.set_title(ship_id)


# #### Visualization with cleaned data

# In[86]:


cols = speed_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[87]:


cols = foc_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[88]:


cols = draft_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[89]:


cols = wind_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[90]:


cols = sea_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[91]:


cols = swell_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[92]:


cols = crnt_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[93]:


cols = wave_columns
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids), len(cols), figsize=(15*len(cols), 5*len(ship_ids)))
for ship_id, axes_ in zip(ship_ids, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.lineplot(x='utc', y=col, ax=ax, data=df_mod_3.loc[ship_id], color=color)
        ax.set_title(ship_id)


# In[97]:


df_mod_3.to_csv(data_path / 'leg_top5_mod.csv')


# In[ ]:




