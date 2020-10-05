#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


df_raw = (pd
          .read_csv(data_path / 'processed/leg_top5.csv', index_col=0, parse_dates=['utc'])
          .set_index(['ship_id', 'utc'])
          .sort_values(by=['ship_id', 'utc'])
         )

ship_id = df_raw.index.levels[0]


# In[5]:


draftlist = (pd
             .read_csv(data_path / 'processed/draftlist.csv', index_col=0, parse_dates=['dep_utc', 'arr_utc'])
             .drop_duplicates(subset=['ship_id', 'dep_utc', 'arr_utc'])
             .set_index(['ship_id', 'dep_utc', 'arr_utc'])
             .sort_values(by=['ship_id', 'dep_utc', 'arr_utc'])
             .assign(rout_id= lambda df: np.arange(1, len(df)+1))
            )

rout_id = pd.melt(draftlist['rout_id'].reset_index(), 
                  value_vars=['dep_utc', 'arr_utc'], 
                  id_vars=['ship_id', 'rout_id'],
                  value_name='utc')


# In[8]:


df_tmp = rout_id.set_index(['ship_id', 'utc']).                 sort_values(by=['ship_id', 'utc']).                 loc[ship_id, :]
print(np.unique((df_tmp['variable'].values[::2] == 'dep_utc')))
print(np.unique((df_tmp['variable'].values[1::2] == 'arr_utc')))


# In[9]:


df_tmp.loc[:, ]


# ###### label the routes

# In[12]:


df_raw


# In[44]:


draftlist_ = draftlist.reset_index()[['ship_id', 'dep_utc', 'arr_utc', 'rout_id']].sort_values(by='ship_id')
draftlist_ = draftlist_.loc[draftlist_['ship_id'].isin(ship_id), :]
# draftlist_.loc[:, 'ship_id'] = draftlist_['ship_id'].astype('str')


# In[63]:


draftlist_.loc[draftlist_['ship_id'] == 12796, :].sort_values('dep_utc')


# In[29]:


df_raw.loc[:, 'rout_id'] = 0


# In[58]:


from tqdm import tqdm_notebook
for _, row in draftlist_.iterrows():
    id_ = row['ship_id']
    dep_utc = row['dep_utc']
    arr_utc = row['arr_utc']
    rout_id = row['rout_id']
    df_raw.loc[(id_, dep_utc) : (id_, arr_utc), 'rout_id'] = rout_id


# In[59]:


df_raw['rout_id'].value_counts()


# In[60]:


df_raw


# In[ ]:




