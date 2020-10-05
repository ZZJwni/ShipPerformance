#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data wrangling
import pandas                  as     pd
import numpy                   as     np
import missingno               as     msno

from pathlib import Path

ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# In[2]:


# Import dataset into Pandas DataFrame
df_raw = pd.read_hdf(data_path / '/processed/leg.h5')

# Check column names and data types
df_raw.info()

draft_raw = pd.read_csv(data_path / 'processed/draftlist.csv', index_col=0)


# In[3]:


# Missing records
df_raw.apply(lambda col: col.isna().sum(), axis=0)


# In[4]:


df_raw.groupby('ship_id').apply(lambda df: len(df)).sort_values(ascending=False)


# In[6]:


# Select 5 ship_id which have longest record
df_s = df_raw.loc[df_raw.ship_id.isin([17165, 18180, 12796, 15391, 24881]), :]


# In[8]:


df_s.to_csv(data_path / 'processed/leg_top5.csv')


# In[ ]:




