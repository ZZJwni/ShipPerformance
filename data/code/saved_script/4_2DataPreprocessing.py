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


# ###### leg.json

# In[3]:


df_raw.head()


# In[4]:


wave_columns = ['pri_ht', 'pri_dir', 'pri_per',
                'sec_ht', 'sec_dir', 'sec_per']
draft_columns = ['draft_aft', 'draft_fore']
foc_columns = ['foc_me', 'rpm']
speed_columns = ['og_speed', 'course']
crnt_columns = ['crnt_u', 'crnt_v']
sea_columns = ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']
swell_columns = ['swl_dir', 'swl_ht', 'swl_per']
wind_columns = ['wind_u', 'wind_v']
position_columns = ['lat', 'lon']

remaining_columns = set(df_raw.columns) - set(wave_columns) -set(draft_columns) -set(foc_columns) -set(speed_columns) - set(crnt_columns) - set(sea_columns) - set(swell_columns) - set(wind_columns) - set(position_columns)
assert remaining_columns == {'ship_id', 'utc'}


# In[5]:


df_raw.reset_index(inplace=True, drop=True)


# In[5]:


print('The number of data points : ', len(df_raw))


# In[3]:


df_raw_length = pd.DataFrame({'number of samples': df_raw.groupby('ship_id').apply(lambda s: len(s))})


# In[8]:


df_raw_length.mean()


# ###### 图1: 样本个数的分布图

# In[10]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_raw_length['number of samples'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of samples')


# In[47]:


df_raw_length.sort_values(by='number of samples')


# In[49]:


df_mod = df_raw.drop_duplicates(
    subset=['ship_id', 'utc'], keep='first', inplace=False)


# In[50]:


df_mod_length = pd.DataFrame({'number of samples': df_mod.groupby('ship_id').apply(lambda s: len(s))})


# In[51]:


df_mod_length.loc[df_mod_length['number of samples'].isna() == True, :]


# In[71]:


df_dup_ratio = (df_raw_length - df_mod_length) / df_raw_length * 100
df_dup_ratio.columns = ['ratio of duplicated samples']


# In[72]:


df_dup_ratio.loc[df_dup_ratio['ratio of duplicated samples'].isna() == True, :]


# In[73]:


df_dup_ratio.loc[:, 'number of samples'] = df_raw_length['number of samples']


# In[74]:


df_dup_ratio = df_dup_ratio.sort_values(by='ratio of duplicated samples', ascending=False)


# In[75]:


df_dup_ratio


# ###### 图2: 重复数据比列在前1000的样本长度分布

# In[76]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_dup_ratio.iloc[:1000, :]['number of samples'], kde=False, norm_hist=False)
_ = ax.set_title('histogram of number of samples with top 1000 duplicated ratio')


# ###### 图3: 含有重复数据的比例的分布图

# In[67]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_dup_ratio.loc[df_dup_ratio['ratio of duplicated samples']
                                > 0., 'ratio of duplicated samples'], kde=False, norm_hist=False)
_ = ax.set_title('histogram of ratio of duplicated samples(%)')


# In[20]:


print('The number of data points : ', len(df_mod))


# ###### Check duplicated rows

# In[21]:


df_dup = df_raw.loc[set(df_raw.index) - set(df_mod.index),
                    :].set_index(['ship_id', 'utc']).sort_values(by=['ship_id', 'utc'])


# In[23]:


df_dup.to_csv(data_path / 'processed/duplicated.csv')


# In[24]:


del df_mod
del df_dup


# In[11]:


df_mod = (df_raw
          .drop_duplicates(subset=['ship_id', 'utc'], keep='first')
          .set_index(['ship_id'])
          .sort_values(by=['ship_id', 'utc'])
          )
ship_ids = np.unique(df_mod.index)


# In[7]:


print('The number of data points : ', len(df_mod))


# ###### 图4: 移除重复值后的分布

# In[80]:


df_mod_length = pd.DataFrame({'number of samples': df_mod.groupby('ship_id').apply(lambda s: len(s))})
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_length['number of samples'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of samples(after drop duplicated)')


# #### Data Cleaning

# In[8]:


cols = wind_columns + sea_columns + swell_columns + crnt_columns + wave_columns


# ##### 统计缺失值

# In[89]:


df_tmp = pd.DataFrame(df_mod[cols].apply(lambda col: np.sum(col == np.float64(9999))) / len(df_mod) * 100)
df_tmp.columns=['ratio of missing value(%)']
df_tmp = df_tmp.sort_values(by='ratio of missing value(%)')


# ######  图5: 各列的缺失值比例

# In[90]:


fig, ax = plt.subplots(figsize=(8, 8))
df_tmp.plot(kind='bar', ax=ax)


# ##### 移除缺失值

# In[10]:


# def _check_nan(arr, nan=np.float64(9999)):
#     for a in arr:
#         if a == nan:
#             return True
#     else:
#         return False


# selected_idx = ~df_mod[cols].apply(lambda row: _check_nan(row), axis=1)


# print('丢弃的数据比例 : ', 1 - sum(selected_idx) / len(selected_idx))

# df_mod_1 = df_mod.loc[selected_idx, :]


# In[9]:


df_t = (df_mod[cols] == np.float(9999))

df_t = df_t.sum(axis=1)

df_mod_1 = df_mod.loc[df_t == 0, :]


# In[10]:


1- len(df_mod_1) / len(df_mod)


# In[11]:


df_mod_1_l = pd.DataFrame({'number of samples': df_mod_1.groupby('ship_id').apply(lambda s: len(s))})


# In[19]:


df_mod_l = pd.DataFrame({'number of samples': df_mod.groupby('ship_id').apply(lambda s: len(s))})


# In[32]:


df_ratio = (1- df_mod_1_l / df_mod_l) * 100
df_ratio.columns = ['ratio of missing values']
df_ratio.loc[:, 'number of samples'] = df_mod_l['number of samples']
df_ratio = df_ratio.sort_values(by='ratio of missing values', ascending=False)


# In[41]:


df_ratio.iloc[:500]


# ###### 图6: 缺失值比率的分布图

# In[44]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_ratio['ratio of missing values'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of ratio of missing values(%) (after drop duplicated)')


# ###### 图7: 缺失值比率前500的samples分布情况

# In[45]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_ratio.iloc[:500]['number of samples'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of samples with top 500 ratio of missing values(%) (after drop duplicated)')


# ###### 图7-1： 移除了缺失值后的分布图

# In[13]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_1_l['number of samples'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of samples(after drop duplicated and missing values)')


# In[32]:


# df_mod_1 = df_mod.loc[selected_idx, :]


# In[34]:


for ship_id in ship_ids:
    print('ship id : ', ship_id)
    print('current length : ', len(df_mod_1.loc[ship_id, :]))
    print(len(df_mod_1.loc[ship_id, :]) / len(df_mod.loc[ship_id, :]))


# ##### modify degrees

# In[11]:


# change dir from [0, 2*pi] to [0, pi]
cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']

print(df_mod_1.loc[:, cols].apply(lambda col: col.max(), axis=0))
print('----------')
print(df_mod_1.loc[:, cols].apply(lambda col: col.min(), axis=0))


# 1. Degree 不能有负数, 我们删除负数的角度
# 2. 将角度转化为弧度

# In[16]:


df_mod_1.reset_index(inplace=True)


# In[19]:


cols = ['course', 'pri_dir', 'sea_dir', 'sec_dir', 'swl_dir']
selected_idx = ((df_mod_1[cols] >= 0).sum(axis=1) == 5)

print('丢弃的数据比例 : ',  selected_idx.value_counts()[False]/ len(selected_idx))


# In[20]:


df_mod_1 = df_mod_1.loc[selected_idx.values, :]


# In[21]:


# def modify_degrees(arr):
#     return np.where(arr>180, 360-arr, arr)

# df_mod_2 = df_mod_1.assign(**{
#     col: modify_degrees(df_mod_1[col].values) for col in cols})

# print(df_mod_2.loc[:, cols].apply(lambda col: col.max(), axis=0))

df_mod_2 = df_mod_1.assign(
    course=df_mod_1['course'] / 360 * 2 * np.pi,
    pri_dir=df_mod_1['pri_dir'] / 360 * 2 * np.pi,
    sec_dir=df_mod_1['sec_dir'] / 360 * 2 * np.pi,
    sea_dir=df_mod_1['sea_dir'] / 360 * 2 * np.pi,
    swl_dir=df_mod_1['swl_dir'] / 360 * 2 * np.pi,
)


# In[22]:


for col in cols:
    df_mod_2.loc[:, col] = df_mod_2[col].astype(np.float32)


# In[23]:


df_mod_2.info()


# In[24]:


del df_mod
del df_mod_1


# In[62]:


df_mod_2.to_csv(data_path / data_path / 'processed/df_mod_2.csv')


# In[7]:


df_mod_2 = pd.read_csv(filepath_or_buffer=data_path / 'processed/df_mod_2.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))


# ##### check outliers

# In[8]:


df_mod_2.set_index('ship_id', inplace=True)


# In[9]:


df_mod_2.head()


# In[12]:


df_mod_2['og_speed'].plot(kind='box')


# In[25]:


df_mod_2['og_speed'].quantile(.97)


# In[27]:


df_mod_2['og_speed'].quantile(.03)


# In[28]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_2.query('8.4<=og_speed<=20.36')
             ['og_speed'], ax=ax, kde=True)


# In[ ]:


### fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_2['foc_me'], ax=ax, kde=True)


# In[71]:


ship_ids_ = ship_ids[:10]

# check outliers
cols = ['og_speed', 'foc_me', 'rpm']
colors = sns.color_palette()[:len(cols)]
fig, axes = plt.subplots(len(ship_ids_), len(
    cols), figsize=(15*len(cols), 5*len(ship_ids_)))
for ship_id, axes_ in zip(ship_ids_, axes):
    for col, ax, color in zip(cols, axes_, colors):
        sns.distplot(a=df_mod_2.loc[ship_id, col], ax=ax, color=color)
        ax.set_title(ship_id)


# In[26]:


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


# In[27]:


df_mod_3.to_csv(data_path / 'processed/df_mod_3.csv')


# In[ ]:





# In[28]:


del df_mod_2
del df_raw


# In[25]:


df_mod_3_l


# In[4]:


df_mod_3 = pd.read_csv(filepath_or_buffer=data_path / 'processed/df_mod_3.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))


# In[7]:


df_mod_3_l = pd.DataFrame({'number of samples' : df_mod_3.groupby('ship_id').apply(lambda s: len(s))})


# In[12]:


df_ratio = (df_raw_length - df_mod_3_l) / df_raw_length
df_ratio.columns = ['ratio of dropped samples']


# In[13]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_ratio['ratio of dropped samples'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of ratio of dropped samples')


# ###### 图8: 移除异常值, 缺失值, 重复值后的数据分布

# In[29]:


df_mod_3_l = pd.DataFrame({'number of samples' : df_mod_3.groupby('ship_id').apply(lambda s: len(s))})
# df_mod_3_l.columns = ['number of samples']
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_3_l['number of samples'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of samples(after remove duplicated, missing values and outliers)')


# In[4]:


df_mod_3 = pd.read_csv(filepath_or_buffer=data_path / 'processed/df_mod_3.csv', 
                     index_col=0, 
                     parse_dates=['utc'], 
                     dtype=dict.fromkeys(['course', 'crnt_u', 'crnt_v',
                                          'draft_aft', 'draft_fore', 'foc_me',
                                          'lat', 'lon', 'og_speed', 'pri_dir',
                                          'pri_ht', 'pri_per', 'rpm', 'sea_dir',
                                          'sea_ht', 'sea_per', 'sec_dir', 'sec_ht',
                                          'sec_per', 'sig_ht', 'swl_dir', 'swl_ht',
                                          'swl_per', 'wind_u', 'wind_v'], np.float32))


# In[8]:


df_mod_3_l = pd.DataFrame({'number of samples' : df_mod_3.groupby('ship_id').apply(lambda s: len(s))})
# df_mod_3_l.columns = ['number of samples']
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_mod_3_l['number of samples'], kde=False, norm_hist=False, bins=100, label='after data cleaning')
sns.distplot(a=df_raw_length['number of samples'], kde=False, norm_hist=False, bins=100, label = 'before data cleaning')
fig.legend()


# ##### 统计连续记录的比例

# In[30]:


df_mod_3.reset_index(drop=False, inplace=True)


# In[31]:


df_utc = df_mod_3[['ship_id', 'utc']]


# In[32]:


#向前差分
s1 = df_utc.groupby('ship_id')['utc'].apply(lambda s: s.diff())
s1 = [s.total_seconds() / 3600 for s in s1]
#向后差分
s2 = df_utc.groupby('ship_id')['utc'].apply(lambda s: s.diff(-1))
s2 = [np.abs(s.total_seconds()) / 3600 for s in s2]


# In[33]:


df_utc.loc[:, 's1'] = s1
df_utc.loc[:, 's2'] = s2


# In[34]:


df_utc_ = df_utc.fillna(100)
df_utc = df_utc.fillna(0)


# ###### 图9: utc前后差分的分布图

# In[35]:


fig, ax = plt.subplots(figsize=(8, 8))
df_utc['s1'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# In[36]:


fig, ax = plt.subplots(figsize=(8, 8))
df_utc['s2'].value_counts().iloc[:20].plot(kind='bar', ax=ax)


# ###### 给每个时间点分类

# In[37]:


df_utc_.loc[:, 'alone'] = (df_utc_['s1'] > 6) & (df_utc_['s2'] > 6)
df_utc_.loc[:, 'start'] = (df_utc_['s1'] > 6) & (df_utc_['s2'] <= 6)
df_utc_.loc[:, 'end'] = (df_utc_['s1'] <= 6) & (df_utc_['s2'] > 6)
df_utc_.loc[:, 'inner'] = (df_utc_['s1'] <= 6) & (df_utc_['s2'] <= 6)


# In[42]:


point_stats = df_utc_.groupby('ship_id')[['alone', 'start', 'end', 'inner']].apply(lambda x: np.sum(x))


# In[43]:


point_stats


# ###### 图10： 各个点的分布情况

# In[44]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=point_stats['alone'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of isolated point')


# In[45]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=point_stats['start'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of start point')


# In[46]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=point_stats['end'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of end point')


# In[47]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=point_stats['inner'], kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of inner point')


# In[60]:


start_idx = df_utc_.query('start == True').index
end_idx = df_utc_.query('end == True').index


# In[61]:


start_idx


# In[64]:


df_utc_start = df_utc_.query('start == True')


# In[75]:


df_utc_start.loc[:, 'l'] = end_idx - start_idx + 1


# In[78]:


df_utc_start.groupby('ship_id').max()


# In[79]:


df_utc_start.groupby('ship_id').min()


# ###### 统计长度大于64的样本数量

# In[81]:


df_utc_start.groupby('ship_id')['l'].apply(lambda x: np.sum(x>=64))


# In[82]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=df_utc_start.groupby('ship_id')['l'].apply(
    lambda x: np.sum(x >= 64)), kde=False, norm_hist=False, bins=100)
_ = ax.set_title('histogram of number of continus records longer than 64')


# In[83]:


df_utc_start


# In[86]:


df_utc_end = df_utc_.query('end == True')
df_utc_end.loc[:, 'l'] = end_idx - start_idx + 1


# In[87]:


df_utc_start_ = df_utc_start.query('l>=64')
df_utc_end_ = df_utc_end.query('l>=64')


# In[88]:


df_utc_end_


# In[89]:


df_utc_start_ 


# In[92]:


selected_idx = pd.DataFrame({'start':df_utc_start_.index, 
                             'end' : df_utc_end_.index,
                             'ship_id': df_utc_start_['ship_id']})


# In[96]:


selected_idx.to_csv(data_path / 'processed/selected_idx_df_mod_3.csv')


# In[97]:


df_utc_.to_csv(data_path / 'processed/df_utc_.csv')


# In[100]:


df_utc_start_['ship_id'].max()


# ##### ship db

# In[101]:


ship_db = pd.read_excel(data_path / 'raw/shipdb.xlsx', sheet_name=3)


# In[103]:


ship_db = ship_db.set_index('wni_ship_num')


# In[107]:


ship_ids = df_utc_start_['ship_id'].unique()


# In[111]:


ship_db_ = ship_db.loc[ship_ids, :]


# ###### 问题: 怎么划分ship?
# 1. 按照建造时间
# 2. 按照尺寸
# 3. 按照类型
# 4. 按照引擎性能

# In[114]:


ship_db_


# In[115]:


len(ship_ids)


# ###### 图11： 船只类型的直方图

# In[118]:


ship_db_.groupby('ship_type').apply(lambda x: len(x)).plot(kind='bar')


# In[121]:


a = 'BULK CARRIER'
bc = ship_db_.query('ship_type == @a')


# In[123]:


bc.head()


# In[129]:


fig,ax = plt.subplots(figsize=(8, 8))
bc['date_built_year'].fillna('unknown').value_counts().plot(kind='bar')


# In[137]:


# 最大值 1000
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=bc['rpm_at_mcr'], kde=False, norm_hist=False)


# In[138]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(data=bc['speed_at_mcr'], )


# In[144]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.kdeplot(data=bc.query('rpm_at_mcr < 140').query('13<=speed_at_mcr<=15')['speed_at_mcr'],
            data2=bc.query('rpm_at_mcr < 140').query('13<=speed_at_mcr<=15')['rpm_at_mcr'],
            cmap="Reds", shade=True, shade_lowest=False, ax=ax)


# In[146]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=bc['depth'].dropna(), kde=False, norm_hist=False)


# In[147]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(a=bc['draft'].dropna(), kde=False, norm_hist=False)


# ###### 问题: 在起点和终点处需要过滤掉foc_me突变的点

# In[ ]:


# fig, ax = plt.subplots(figsize=(16, 8))
x = [1,2,3,4,5,6,7,8,9,  10]
y = [10, 10, 40, 42, 43, 45, 48, 45, 12, 10]
sns.lineplot(x=x, y=y, ax=ax)


# In[ ]:




