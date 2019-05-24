# -*- coding: utf-8 -*-
import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
sys.setrecursionlimit(10000)
# define the data/predictors as the pre-set feature names  
os.chdir(os.path.dirname(os.path.realpath(__file__)))
df = pd.read_excel(os.getcwd() + os.sep + "hs300Components160101.xlsx")

r = 0.04
dtt = 1/255.0

code_dict = {}
date_dict = {}

SHORT_POINT = 1.25
COVER_POINT = 0.75
SELL_POINT = -0.5
BUY_POINT = -1.25

SUPER_SHORT = u"超卖区域"
SHORTING = u"卖方力量"
XUWU = u"中间区域"
BUYING = u"买方力量"
SUPER_BUY = u"超买区域"

SHORT = "SHORT"
COVER = "COVER"
SELL = "SELL"
BUY = "BUY"

DLONG = "DLONG"
DSHORT = "DSHORT"
DNOPE = "DNOPE"


class indiv:
    #日期，收盘价，当日股息，s-score，沪深300的收盘价，λ
    dt = None
    code = ''
    close = 0.0
    pre_close = 0.0
    #rt = 0.0                #相较于昨日涨幅
    ss = 0.0
    pre_ss = 0.0       
    hs300 = 0.0
    pre_hs300 = 0.0
    
    pre_area = ''
    area = ''
    transaction = ''
    direction = ''
    

    def __init__(self,dt,code,close,pre_close,ss,pre_ss,hs300,pre_hs300):
        self.dt = dt
        self.code = code
        self.close = close
        self.ss = ss
        self.hs300 = hs300
        self.pre_close = pre_close
        self.pre_ss = pre_ss
        self.pre_hs300 = pre_hs300
        self.rt = close/pre_close - 1
        self.long_rt = 0.0
        self.short_rt = 0.0
        
        
        #判断今天所在区域
        if ss >= SHORT_POINT:
            self.area = SUPER_SHORT
        elif ss < SHORT_POINT and ss >COVER_POINT:
            self.area = SHORTING
        elif ss <= COVER_POINT and ss >= SELL_POINT:
            self.area = XUWU
        elif ss < SELL_POINT and ss >BUY_POINT:
            self.area = BUYING
        elif ss <= BUY_POINT:
            self.area = SUPER_BUY
        else:
            self.area = "NONE"
            
        #判断昨天所在区域
        if pre_ss >= SHORT_POINT:
            self.pre_area = SUPER_SHORT
        elif pre_ss < SHORT_POINT and pre_ss >COVER_POINT:
            self.pre_area = SHORTING
        elif pre_ss <= COVER_POINT and pre_ss >= SELL_POINT:
            self.pre_area = XUWU
        elif pre_ss < SELL_POINT and pre_ss >BUY_POINT:
            self.pre_area = BUYING
        elif pre_ss <= BUY_POINT:
            self.pre_area = SUPER_BUY
        else:
            self.pre_area = "NONE"
        
        self.change_transaction_direction()
    
    def __repr__(self):
        return str(self.dt) + "_" + self.code
    
    def change_transaction_direction(self):
        
        #做空
        if self.pre_area == SHORTING and self.area == SUPER_SHORT:
            self.transaction = SHORT
        #平空
        if self.pre_area == SHORTING and self.area == XUWU:
            self.transaction = COVER
        #做多
        if self.pre_area == BUYING and self.area == SUPER_BUY:
            self.transaction = BUY
        #平多
        if self.pre_area == BUYING and self.area == XUWU:
            self.transaction = SELL



  
def return_signal(i,code):
    """i是第几天，code是哪只票，返回信号和分数"""
    test_df = df[[code,'000300.SH']][i:i+60]
    test_df = (test_df/test_df.iloc[0]).apply(np.log)
    test_df['idd'] = np.array(range(60))/252.0
    X = test_df[['000300.SH','idd']]
    y = test_df[code]
    model = sm.OLS(y, X).fit()
    beta = model.params[0]
    alpha = model.params[1]
    inverse = np.std(test_df[code])
    gap = test_df.iloc[-1][code]-(test_df.iloc[-1]['000300.SH']*beta + test_df.iloc[-1]['idd']*alpha)
    sscore = gap/inverse
    
    #print code,i,sscore
    
    if sscore >= SHORT_POINT:
        signal = SUPER_SHORT
    elif sscore < SHORT_POINT and sscore >COVER_POINT:
        signal = SHORTING
    elif sscore <= COVER_POINT and sscore >= SELL_POINT:
        signal = XUWU
    elif sscore < SELL_POINT and sscore >BUY_POINT:
        signal = BUYING
    elif sscore <= BUY_POINT:
        signal = SUPER_BUY
    else:
        signal = "NONE"

    return signal,sscore





def insert_obj_dict(code):
    
    """
    用于修改对象的参数值
    """
    
    global code_dict,date_dict
    
    objs = []
    prev_direc = DNOPE
    
    pre_signal,pre_sscore = return_signal(0,code)           #前一天策略评分
    pre_close_price = df.iloc[0][code]                 #当日收盘价
    pre_hs300 = df.iloc[0]['000300.SH']                #当日沪深300
    
    
    for i in range(0,len(df)-59):
        
        signal,sscore = return_signal(i,code)           #策略评分
        date = df.Date[i+59]                       #日期
        close_price = df.iloc[i][code]             #当日收盘价
        hs300 = df.iloc[i]['000300.SH']            #当日沪深300
        obj = indiv(date,code,close_price,pre_close_price,sscore,pre_sscore,hs300,pre_hs300) #构建模型
        
        #修改持仓方向,根据持仓方向来计算套利回报
        obj.direction = prev_direc
        if obj.transaction in [COVER,SELL]:
            prev_direc = DNOPE
        elif obj.transaction == BUY:
            prev_direc = DLONG
        elif obj.transaction == SHORT:
            prev_direc = DSHORT
        
        #计算套利回报，以及其中的多头回报和空头回报
        if obj.direction == DLONG:
            obj.rt = (obj.close/obj.pre_close - 1)+(obj.pre_hs300/obj.hs300 - 1)
            obj.long_rt = (obj.close/obj.pre_close - 1)
            obj.short_rt = (obj.pre_hs300/obj.hs300 - 1)
        elif obj.direction == DSHORT:
            obj.rt = (obj.pre_close/obj.close - 1)+(obj.hs300/obj.pre_hs300 - 1)
            obj.long_rt = (obj.hs300/obj.pre_hs300 - 1)
            obj.short_rt = (obj.pre_close/obj.close - 1)
        
       
            
        #只有有持仓那天才加入date_dict
        print obj.__repr__() + " " + str(obj.rt) + " " + obj.direction
        if obj.direction != DNOPE:
            if date_dict.has_key(obj.dt):
                date_dict[obj.dt].append(obj)
            else:
                date_dict[obj.dt] = [obj]
        
        pre_signal,pre_sscore = signal,sscore       #修改pre变量
        pre_close_price = close_price
        pre_hs300 = hs300
        
        objs.append(obj)

    code_dict[code] = objs


#将对象构造好，塞进code为key的字典和date为key的字典
l = list(df.columns)
l.remove('Date')
l.remove('000300.SH')
map(insert_obj_dict,l)

#将timestamp和1/255相互转换
ts_list = date_dict.keys()
ts_list.sort()
dtt_list = np.array(range(1,len(ts_list)+1))*dtt
ts_to_dt_dict = dict(zip(ts_list,dtt_list))
dt_to_ts_dict = dict(zip(dtt_list,ts_list))


def to_dt(dt):
    if isinstance(dt, float):
        return dt
    else:
        return ts_to_dt_dict[dt]
def to_ts(ts):
    if isinstance(ts, pd.Timestamp):
        return ts
    else:
        return dt_to_ts_dict[ts]


#计算pnl的函数
#def qit(dt):
#    dt = to_ts(dt)
#    lmd = 1.0/len(date_dict[dt])
#    dt = to_dt(dt)
#    return E(dt)*lmd
#
#def sum_abs_q(dt):
#    dt = to_ts(dt)
#    n = len(date_dict[dt])
#    dt = to_dt(dt)
#    return np.abs(qit(dt)-qit(dt-dtt))*n*5e-4
#
#def sum_qrt(dt):
#    dt = to_ts(dt)
#    obj_list = date_dict[dt]
#    dt = to_dt(dt)
#    return qit(dt)*sum(map(lambda x:x.rt,obj_list))
#
#def sum_qt(dt):
#    dt = to_ts(dt)
#    obj_list = date_dict[dt]
#    n = len(obj_list)
#    dt = to_dt(dt)
#    return qit(dt)*n
#    
#e_result_dict = {}
#def E(t):
#    global e_result_dict
#    t = to_dt(t)
#    
#    #试探该结果是否已经计算出来，如果计算过就不用重复计算了
#    if e_result_dict.has_key(t):
#        return e_result_dict[t]
#    
#    if t == dtt:
#        result = 4e6
#    else:
#        i1 = E(t - dtt)
#        i2 = E(t - dtt)*r*dtt
#        i3 = sum_qrt(t - dtt)
#        i4 = -(sum_qt(t - dtt)*r * dtt)
#        #i5 = - sum_abs_q(t)
#        
#        result =  i1 + i2 + i3 + i4 
#    e_result_dict[t] = result
#    return result
#上面几个函数都没用了，递归会超出python的限制



def get_turnover():
    """
    得到换手率
    """
    date_turnover_dict = {}
    date_list = np.sort(date_dict.keys())
    for i in range(len(date_list)-1):
        day1 = set(map(lambda x:x.code,date_dict[date_list[i]]))
        day2 = set(map(lambda x:x.code,date_dict[date_list[i+1]]))
        N1 = float(len(day1))
        N2 = float(len(day2))
        n1 = len(day1-day2)
        n2 = len(day2-day1)
        n3 = len(day1&day2)
        turnover = n1/N1+n2/N2+np.abs(n3/N1-n3/N2)
        #print "{:.2%}".format(turnover)
        date_turnover_dict[date_list[i]] = turnover
    return date_turnover_dict

#换手率和日期的字典
date_turnover_dict = get_turnover()
mean_turnover = np.mean(date_turnover_dict.values())





def from_sr_to_greeks(sr,title = "",mean_turnover = mean_turnover):
    """
    计算希腊值
    """
    
    N = 252.0
    ##年化收益
    days = len(sr)
    r = sr.values[-1]/sr.values[0]-1#9月26日到12月21
    ar = (1+r)**(N/days)-1    
    
    #标准差
    varss = np.std((sr/sr.shift(1)-1).dropna())
    
    #夏普率
    sharpe = (ar-0.04)/(varss*(N**0.5))
    
    mc = sr.values
    #最大回撤
    setback = sr - sr.cummax()
    max_fall = setback.min()
    #最大回撤日,
    max_fall_day = setback[setback==max_fall].index[0].strftime("%Y-%m-%d")
    
    #胜率
    win_rate = len(sr[sr>sr.shift(1)])/float(len(sr))
    
    #信息比率
    avg_residual = np.mean(np.abs(mc-mc.mean()))
    std_residual = np.std(np.abs(mc-mc.mean()))
    info_rate = avg_residual/std_residual
    
    #收益波动率
    n = len(sr)
    srp = (sr/sr.shift(1)-1).dropna()
    volatitlity  =  ((float(N)/n-1)*((srp-srp.mean())**2))**0.5
    volatitlity  =  np.std(srp)
    
    import prettytable as pt

    ## 按行添加数据
    tb = pt.PrettyTable()
    tb.field_names = ["参数", "参数值"]
    tb.add_row(["名称",title])
    tb.add_row(["年化收益率","{:.2%}".format(ar)])
    tb.add_row(["夏普率","{:.3}".format(sharpe)])
    tb.add_row(["最大回撤",int(np.abs(max_fall))])
    tb.add_row(["最大回撤日", max_fall_day])
    tb.add_row(["胜率", "{:.2%}".format(win_rate)])
    tb.add_row(["信息比率", "{:.2%}".format(info_rate)])
    tb.add_row(["平均日换手率", "{:.2%}".format(mean_turnover)])
    print(tb)
    
    #return ar,varss,sharpe,max_fall,max_fall_day,win_rate,info_rate,volatitlity
    
def yield_capital(n, e = 1e6):
    """
    用于得到收益曲线
    ==========================
    使用while循环，参数进行直接修改
    不会超出递归限制，因为这根本没有递归
    这是迭代，迭代和递归都是循环的一种
    """
    global date_turnover_dict
    dt = dtt
    l = dt_to_ts_dict.keys()
    l.sort()
    i = 0
    positions =[]
    while True:                     
        if i == n - 1:
            return e#,positions
        #多空的指数需要抵消，多头多于空头一种情况，其他一种情况
        long_num = map(lambda x:x.direction,date_dict[to_ts(dt)]).count(DLONG)
        short_num = map(lambda x:x.direction,date_dict[to_ts(dt)]).count(DSHORT)
        if long_num>short_num:
            total_num = long_num
        else:
            total_num = short_num
        positions.append(float(total_num)/len(date_dict[to_ts(dt)]))
        #这是满仓的情况
        #i,e  = i+1, e \
        #            +e*1.0/total_num*sum(map(lambda x:x.rt,date_dict[to_ts(dt)]))\
        #            -e*3e-4*date_turnover_dict[to_ts(dt)]
        #            #+ e *r*dtt - e*1.0/len(date_dict[to_ts(dt)])*len(date_dict[to_ts(dt)])*r * dtt 
        #没有抵消的，非满仓
        #i,e  = i+1, e \
        #            +e*1.0/len(date_dict[to_ts(dt)])*sum(map(lambda x:x.short_rt,date_dict[to_ts(dt)]))\
        #            -e*3e-4*date_turnover_dict[to_ts(dt)]
        #            #+ e *r*dtt - e*1.0/len(date_dict[to_ts(dt)])*len(date_dict[to_ts(dt)])*r * dtt 
        #只做多股票，满仓
        #只保留多头
        stock_objs = filter(lambda x:x.direction == DLONG,date_dict[to_ts(dt)])
        if len(stock_objs) == 0:
            i,e  = i+1, e
        else:
            i,e  = i+1, e \
                    +e*1.0/len(date_dict[to_ts(dt)])*sum(map(lambda x:x.long_rt,stock_objs))\
                    -e*3e-4*date_turnover_dict[to_ts(dt)]
                    #+ e *r*dtt - e*1.0/len(date_dict[to_ts(dt)])*len(date_dict[to_ts(dt)])*r * dtt 
        dt = l[i]

        

title = "Position Full Long Only(Index Included)"
sr = pd.Series(map(lambda x:yield_capital(x),(range(1,len(date_dict)+1))),index = np.sort(date_dict.keys()))
from_sr_to_greeks(sr,title)




#展示pnl
sr.plot(grid=True,title=title)
plt.show()


#看某只股票的sscore
#sdf = []
#dff = pd.DataFrame(sdf,columns=['d','ss','p','hs'])
#dff['p']  = (dff['p'] - np.mean(dff['p']))/np.std(dff['p'])
#dff['hs']  = (dff['hs'] - np.mean(dff['hs']))/np.std(dff['hs'])
#dff['sp'] = SHORT_POINT
#dff['cp'] = COVER_POINT
#dff['bp'] = BUY_POINT
#dff['sep'] = SELL_POINT
#dff.plot('d',y = ['ss','p','sp','cp','bp','sep','hs'])

