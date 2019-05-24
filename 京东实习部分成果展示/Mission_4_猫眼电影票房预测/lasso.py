# -*- coding: utf-8 -*-
#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

df = pd.read_excel("/Users/xuegeng/Documents/JD_Quant/Mission_4_Catseye/all_box_office.xlsx")
col = [  u'releaseInfo',          u'rate',     u'rate_type',       u'foreign',
         u'displayTime',     u'time_type', u'showDayOfWeek',     u'first_day',
                u'上映2天',          u'上映3天',          u'上映4天',          u'上映5天',
                u'上映6天',          u'上映7天',          u'上映8天',          u'上映9天',
               u'上映10天',         u'上映11天',         u'上映12天',         u'上映13天',
               u'上映14天',         u'上映15天',         u'上映16天',         u'上映17天',
               u'上映18天',         u'上映19天',         u'上映20天',         u'上映21天',
               u'上映22天',         u'上映23天',         u'上映24天',         u'上映25天',
               u'上映26天',         u'上映27天',         u'上映28天',         u'上映29天',
               u'上映30天',   u'Unnamed: 37']

df = df.fillna(method='pad',axis=1)

from sklearn.linear_model import Lasso
def lasso_regression(data, predictors,y, alpha = 1e-15):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data[y])
    y_pred = lassoreg.predict(data[predictors])
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data[y])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret



#Initialize predictors to all 15 powers of x
predictors=[u'first_day',
                u'上映2天',          u'上映3天',          u'上映4天',          u'上映5天',
                u'上映6天',          u'上映7天',          u'上映8天',          u'上映9天',
               u'上映10天',         u'上映11天',         u'上映12天',         u'上映13天',
               u'上映14天',         u'上映15天',         u'上映16天',         u'上映17天',
               u'上映18天',         u'上映19天',         u'上映20天']
print len(lasso_regression(df, predictors, predictors[-1]))



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



































