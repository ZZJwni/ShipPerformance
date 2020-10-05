#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data wrangling
import pandas as pd
import numpy as  np

from pathlib import Path

ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# In[2]:


# Import dataset into Pandas DataFrame
df_raw = pd.read_hdf(data_path / 'processed/leg.h5')

# Check column names and data types
df_raw.info()

# draft_raw = pd.read_csv(data_path / 'processed/draftlist.csv', index_col=0)


# In[3]:


# dropped_columns = ['client', 'point_type', 'loading_condition',
#                    'deadweight', 'power', 'sst', 'foc_total']

dropped_columns = ['point_type', 'loading_condition',
                   'power', 'sst', 'foc_total']


# In[4]:


df_raw.drop(dropped_columns, inplace=True, axis=1)


# In[5]:


df_raw.dtypes


# In[7]:


df_raw.reset_index(inplace=True, drop=True)


# ##### 处理类型为object的列

# ###### load

# In[11]:


df_raw.loc[:, 'load'].head()


# In[13]:


for a in df_raw.loc[:, 'load'].unique():
    print(a)


# OK...... Let's drop load this time...

# In[8]:


df_raw.drop('load', axis=1, inplace=True)


# ###### og_speed

# In[24]:


for a in df_raw.loc[:, 'og_speed'].unique():
    try:
        np.float(a)
    except ValueError:
        print(a)


# In[9]:


dropped_idx = []
for idx, og_speed in zip(df_raw.index, df_raw.loc[:, 'og_speed']):
    try:
        np.float32(og_speed)
    except ValueError:
        print(og_speed)
        dropped_idx.append(idx)


# Drop improper og_speed!!!

# In[10]:


df_raw.drop(dropped_idx, axis=0, inplace=True)
df_raw.reset_index(inplace=True, drop=True)


# ###### rpm

# In[36]:


for a in df_raw.loc[:, 'rpm'].unique():
    try:
        np.float(a)
    except ValueError:
        print(a)


# In[11]:


dropped_idx = []
for idx, og_speed in zip(df_raw.index, df_raw.loc[:, 'rpm']):
    try:
        np.float32(og_speed)
    except ValueError:
        print(og_speed)
        dropped_idx.append(idx)


# Drop improper rpm!!!

# In[12]:


df_raw.drop(dropped_idx, axis=0, inplace=True)
df_raw.reset_index(inplace=True, drop=True)


# ###### tw_speed

# In[12]:


df_raw.loc[:, 'tw_speed'].unique()


# drop tw_speed!

# In[13]:


df_raw.drop('tw_speed', axis=1, inplace=True)


# Now we convert og_speed and rpm to np.float32

# In[14]:


df_raw.loc[:, 'og_speed'] = df_raw['og_speed'].astype(np.float32)
df_raw.loc[:, 'rpm'] = df_raw['rpm'].astype(np.float32)


# In[ ]:


df_raw.to_csv(data_path / 'processed/leg.csv')


# In[ ]:




