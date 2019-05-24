# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tushare as ts
import os,time
from datetime import datetime,timedelta
trading_days  = map(pd.to_datetime,ts.get_hist_data('hs300').index)
trading_days  = list(reversed(trading_days))

sec_map_path = "/Users/xuegeng/Documents/JD_Quant/Mission_1_Shareholder_Overweigh/stock_sec_map.xlsx"
dfcs = pd.read_excel(sec_map_path)

buffer_path = "/Users/xuegeng/Documents/JD_Quant/Mission_6_脱水研报跟踪/buffer.xlsx"
dfbf = pd.read_excel(buffer_path)

wd_dry_path = "/Users/xuegeng/Documents/JD_Quant/Mission_6_脱水研报跟踪/trade_blotter_wind.xlsx"
dfwd = pd.read_excel(wd_dry_path)



def name_to_code(name):
    """
    将股票名字转化为股票代码
    """
    try:
        name = name.replace(' ','')
        code = dfcs[dfcs.code_name == name].code.values[0]
    except (IndexError,AttributeError):
        code = "Aschloch!"
    return code    
    
def indate_to_outdate(date1):
    """
    将某个日期推后5个交易日
    """
    try:
        indtid = trading_days.index(date1)
        outdtid = indtid + 5
        if outdtid < 0:
            return np.nan
        outdt = trading_days[outdtid]
    except (IndexError,ValueError):
        outdt = np.nan
    return outdt

def get_open_price(code,date_):
    """
    用股票代码和日期,得到开盘价
    """
    if isinstance(date_,pd._libs.tslib.NaTType):
        return np.nan
    try:
        path = "/Users/xuegeng/Documents/JD_Quant/Mission_3_variance_ratio_test/AM/%s.xls"%code
        data = pd.read_excel(path)
        opens = data[data[u"日期"] == date_.strftime("%Y-%m-%d")][u"开盘价(元)"].values[-1]
    except IOError:
        opens = np.nan
    except IndexError:  
        try:  
            code = code.split('.')[0]
            dt = date_.strftime("%Y-%m-%d")
            opens = ts.get_hist_data(code,start=dt,end=dt).iloc[0].open
        except IndexError:
            opens = np.nan
    return opens

def apply_get_inprice(row):
    code = row[u'code']
    date_ = row[u'in_date']
    return get_open_price(code,date_)
    
def apply_get_outprice(row):
    code = row[u'code']
    date_ = row[u'out_date']
    return get_open_price(code,date_)

def sum_cap(cap_dic):
    return sum(map(lambda x:cap_dic[x]['inCap'],range(5)))
    


def cal_volume(dfjq):
    """
    计算手数
    """
    dfjq = dfjq[~dfjq.duplicated([u'in_date', u'code'])].copy()
    cap_dic  = dict(zip(range(5),[{"isNew":True,"inCap":0,"outCap":0}]*5))
    cap_dic  = {0: {'inCap': 0, 'outCap': 0, 'isNew': True}, 1: {'inCap': 0, 'outCap': 0, 'isNew': True}, 2: {'inCap': 0, 'outCap': 0, 'isNew': True}, 3: {'inCap': 0, 'outCap': 0, 'isNew': True}, 4: {'inCap': 0, 'outCap': 0, 'isNew': True}}
    indates = np.unique(dfjq[u"in_date"])
    indates = np.sort(indates)
    cap_list = []
    for idt in indates:
        print "[+] Processing .." + str(idt)
        idt_str = pd.to_datetime(idt).strftime("%Y-%m-%d")
        trades = dfjq[dfjq[u"in_date"] ==idt_str]
        quanti = len(trades)
        weekday = pd.to_datetime(idt).weekday()
        cap_dic[weekday]["inCap"] = cap_dic[weekday]["outCap"]
        
        #修改初始资金
        if cap_dic[weekday]["isNew"]:
            cap_dic[weekday]["isNew"] = False
            cap_dic[weekday]["inCap"] = quanti*100000
            cap_dic[weekday]["outCap"] = quanti*100000
        
        
        for i in range(quanti):
            data = trades.iloc[i]
            inprice = data[u"in_price"]
            outprice = data[u"out_price"]
            ava_cap = cap_dic[weekday]["inCap"]/(quanti-i)
            
            volume = int(ava_cap/(100*inprice))*100
            
            idx  = data['idx']
            dfjq = dfjq.set_value(idx,'volume',volume)
            
            
            data[u'volume'] = volume
            cap_dic[weekday]["inCap"] -= volume*inprice
            cap_dic[weekday]["outCap"] += volume*(outprice-inprice)
        
         
        
        cap = sum_cap(cap_dic)
        cap_list.append((idt,cap))
            
    return dfjq,cap_list       



"""流程"""

#更新完buffer先执行这一段

#dfbf['code']  = dfbf['name'].apply(name_to_code)
#dfbf['in_price'] = dfbf.apply(apply_get_inprice,axis=1)
#dfbf['out_date']  = dfbf['in_date'].apply(indate_to_outdate)
#dfbf['out_price'] = dfbf.apply(apply_get_outprice,axis=1)
#dfbf.to_excel("/Users/xuegeng/Documents/JD_Quant/Mission_6_脱水研报跟踪/buffer.xlsx")

#将buffer粘贴到trade_blotter_wind
#把trade_blotter_wind的index和idx长度拉对，执行下面一段

dfwd,cap_list = cal_volume(dfwd)

ls = []
for row in dfwd.iterrows():
   row = row[1]
   idt = row['in_date']
   odt = row['out_date']
   ls.append((row['code'],row['volume'],idt,row['in_price'],u"证券",u"买入"))
   ls.append((row['code'],row['volume'],odt,row['out_price'],u"证券",u"卖出"))
#执行下面代码用于得到万德持仓
df = pd.DataFrame(ls,columns = [u"证券代码",u"买卖数量",u"买卖日期",u"买卖价格",u"证券类型",u"买卖方向"])
df = df.sort_values(u"买卖日期")
print len(df)
df2 = df.dropna()
print len(df2)
df2 = df2[~df2.duplicated([u"买卖日期", u"证券代码",u"买卖方向"])].copy()
print len(df2)
df2.to_excel("/Users/xuegeng/Documents/JD_Quant/Mission_6_脱水研报跟踪/FWD.xlsx")


##必须粘贴到wind自己的模板上，那个xls的文件
