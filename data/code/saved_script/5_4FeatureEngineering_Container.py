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


df = pd.read_csv(data_path / 'processed/record_container.csv', index_col=0, parse_dates=['utc'])
db_bulk_carrier = pd.read_csv(data_path / 'processed/db_container_ship.csv', index_col=0)
df = pd.merge(left=df, right=db_bulk_carrier[['ship_id', 'power_at_mcr', 
                                              'speed_at_mcr', 'rpm_at_mcr', 'draft']],
              on='ship_id', how='left')


# In[3]:


df.columns


# In[4]:


df.info()


# #### Feature Engineering

# In[5]:


# transfer deg -> rad
df_mod = df.assign(
    course=df['course'] / 360 * 2 * np.pi,
    sea_dir=df['sea_dir'] / 360 * 2 * np.pi,
    swl_dir=df['swl_dir'] / 360 * 2 * np.pi,
)


# In[6]:


# transfer hp -> kw
df_mod = df_mod.assign(power_at_mcr = df_mod['power_at_mcr'] * 0.745699872 )


# In[7]:


def lg_from_uv(u: np.array, v: np.array, course: np.array) -> np.array:
    """Compute longitudinal components from u, v components
    """
    return u * np.sin(course) + v * np.cos(course)

def tr_from_uv(u: np.array, v: np.array, course: np.array) -> np.array:
    """Compute longitudinal components from u, v components
    """
    return u * np.cos(course)  - v * np.sin(course)


def lg_from_magnitude(m: np.array, m_dir: np.array, course: np.array) -> np.array:
    """Compute longitudinal components from a magnitude and its dir.
    """
    return m * np.cos(m_dir - course)


def tr_from_magnitude(m: np.array, m_dir: np.array, course: np.array) -> np.array:
    """Compute transverse components from a magnitude and its dir.
    """
    return m * np.sin(m_dir - course)


# In[8]:


#  12 weather features
df_mod = df_mod.assign(
    crnt_L=3600 / 1852 *
    lg_from_uv(df_mod['crnt_u'].values,
               df_mod['crnt_v'].values, df_mod['course'].values),
    crnt_T=3600 / 1852 *
    tr_from_uv(df_mod['crnt_u'].values,
               df_mod['crnt_v'].values, df_mod['course'].values),
    wind_L=3600 / 1852 *
    lg_from_uv(df_mod['wind_u'].values,
               df_mod['wind_v'].values, df_mod['course'].values),
    wind_T=3600 / 1852 *
    tr_from_uv(df_mod['wind_u'].values,
               df_mod['wind_v'].values, df_mod['course'].values),
    sea_ht_L=lg_from_magnitude(
        df_mod['sea_ht'].values, df_mod['sea_dir'].values, df_mod['course'].values),
    sea_ht_T=tr_from_magnitude(
        df_mod['sea_ht'].values, df_mod['sea_dir'].values, df_mod['course'].values),
    sea_per_L=lg_from_magnitude(
        df_mod['sea_per'].values, df_mod['sea_dir'].values, df_mod['course'].values),
    sea_per_T=tr_from_magnitude(
        df_mod['sea_per'].values, df_mod['sea_dir'].values, df_mod['course'].values),
    swl_ht_L=lg_from_magnitude(
        df_mod['swl_ht'].values, df_mod['swl_dir'].values, df_mod['course'].values),
    swl_ht_T=tr_from_magnitude(
        df_mod['swl_ht'].values, df_mod['swl_dir'].values, df_mod['course'].values),
    swl_per_L=lg_from_magnitude(
        df_mod['swl_per'].values, df_mod['swl_dir'].values, df_mod['course'].values),
    swl_per_T=tr_from_magnitude(
        df_mod['swl_per'].values, df_mod['swl_dir'].values, df_mod['course'].values),
)


# In[9]:


df_mod[['ship_id', 'utc', 'og_speed', 'crnt_L', 'record_index']]


# In[10]:


# 3 performance features, `power` and `load` are lost.
df_mod = df_mod.assign(normalized_rpm=df_mod['rpm'] / df_mod['rpm_at_mcr'],
#                        normalaized_power = df_mod['power'] / df_mod['power_at_mcr']
                       )


# In[11]:


# 5 ship features, `load_condition`, `dwt(voyage)` and `gm`  are lost.
df_mod = df_mod.assign(draft_diff_norm=(df_mod['draft_fore']-df_mod['draft_aft']) / df_mod['draft'],
                       draft_sum_norm=(df_mod['draft_fore']+df_mod['draft_aft']) / df_mod['draft'],
#                        gm_breadth = df_mod['gm'] / df_mod['breadth']

                       )


# In[12]:


# 3 targets
df_mod = df_mod.assign(tw_speed = df_mod['og_speed'] - df_mod['crnt_L'],
                       normalized_tw_speed = (df_mod['og_speed'] - df_mod['crnt_L']) / df_mod['speed_at_mcr'],
                       normalized_foc_me = df_mod['foc_me'] / df_mod['power_at_mcr'])


# In[13]:


df_mod.columns


# In[14]:


features = [
    'draft_aft', 'draft_fore', 
    'sig_ht', 
    'crnt_L', 'crnt_T','wind_L', 'wind_T', 
    'sea_ht_L', 'sea_ht_T', 'sea_per_L', 'sea_per_T',
    'swl_ht_L', 'swl_ht_T', 'swl_per_L', 'swl_per_T',
    'normalized_rpm', 'draft_sum_norm', 'draft_diff_norm']

targets = ['tw_speed', 'normalized_tw_speed',
           'normalized_foc_me', 'foc_me', 'og_speed']

variables = ['ship_id', 'client', 'record_length', 'record_index', 'utc', 'rpm']


# In[15]:


df_mod[features + targets + variables].to_csv(data_path / 'processed/features_container.csv')


# In[ ]:




