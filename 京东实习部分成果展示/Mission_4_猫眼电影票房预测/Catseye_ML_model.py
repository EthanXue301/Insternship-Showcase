# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin



file_path1 = "/Users/xuegeng/Documents/JD_Quant/Mission_4_Catseye/box_chg.xlsx"
file_path2 = "/Users/xuegeng/Documents/JD_Quant/Mission_4_Catseye/show_rate.xlsx"
file_path3 = "/Users/xuegeng/Documents/JD_Quant/Mission_4_Catseye/all_box_office.xlsx"
dfo = pd.read_excel(file_path3)[['releaseInfo','first_day']]
df = pd.read_excel(file_path1).dropna()
#df = df.merge(dfo,how="outer",on='releaseInfo').dropna()
df2 = df[[u'rate', u'foreign', u'time_type',u'box_chg#1']].copy()
df3 = df[[u'rate', u'foreign', u'time_type',u'box_chg#1',u'box_chg#2']].copy()
df4 = df[[u'rate', u'foreign', u'time_type',u'box_chg#1',u'box_chg#2',u'box_chg#3']].copy()
df5 = df[[u'rate', u'foreign', u'time_type',u'box_chg#1',u'box_chg#2',u'box_chg#3',u'box_chg#4']].copy()
dff = pd.read_excel(file_path2)[['releaseInfo',u'showrate#1', u'showrate#2', u'showrate#3',u'showrate#4', u'showrate#5']]
dff = dff.merge(df,on = 'releaseInfo',how  = 'left').replace(u'--',np.nan).dropna()
dff1 = dff[[u'rate', u'foreign', u'time_type', u'showrate#1']].copy()
dff2 = dff[[u'rate', u'foreign', u'time_type',u'box_chg#1', u'showrate#1']].copy()
dff3 = dff[[u'rate', u'foreign', u'time_type',u'box_chg#1',u'box_chg#2', u'showrate#1']].copy()
dff4 = dff[[u'rate', u'foreign', u'time_type',u'box_chg#1',u'box_chg#2',u'box_chg#3', u'showrate#1' ]].copy()
dff5 = dff[[u'rate', u'foreign', u'time_type',u'box_chg#1',u'box_chg#2',u'box_chg#3', u'box_chg#4', u'showrate#1']].copy()
   

dfs = [df2,df3,df4,df5,dff1,dff2,dff3,dff4,dff5]

def tell_the_high_corr_pairs(df1):
    col_names = df1.corr().columns.values
    
    for col, row in (df1.corr().abs() > 0.5).iteritems():
        print(col, col_names[row.values])
        

def df_to_array(df):
    itl = []
    for i in df.T.iteritems():
        s = list(i)
        itl.append(list(s[1]))
    return np.array(itl)


def array_to_label(X_array,n_clusters=3):
    batch_size = 45
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                        n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(X_array)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    mbk_means_labels = pairwise_distances_argmin(X_array, mbk_means_cluster_centers)
    return mbk_means_labels


def dfs_syn(dfs):
    
    tf_list = []
    for dff in dfs:
        xa = df_to_array(dff)
        lbs = array_to_label(xa)
        tf_list.append(lbs)
    return tf_list





mn = list(df[u'releaseInfo'])
dd = dfs_syn(dfs)#[:4])
cluster_df = pd.DataFrame(dd,columns=mn).T

#cluster_df  = cluster_df.to_excel("/Users/xuegeng/xinxi2.xlsx")
























