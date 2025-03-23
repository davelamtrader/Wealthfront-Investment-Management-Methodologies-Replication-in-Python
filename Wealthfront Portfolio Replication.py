import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt

def get_return(ticker, start=None, end=None):
    """
    Retrieves prices for ticker from Uqer and computes returns
    based on adjusted closing prices.

    Parameters
    ----------
    ticker : str
        ticker, e.g. "000001.XSHE","000300.ZICN"
    start : '20140101'
        Start date of time period to retrieve
    end : '20160701'
        End date of time period to retrieve

    Returns
    -------
    pandas.DataFrame
        Returns of ticker in requested period.
    """
    if start is None:
        start = '20070101'
    if end is None:
        end = datetime.date.today().strftime('%Y%m%d')
    # 000001.ZICN 上證綜指；399001.ZICN 深證成指；399005.ZICN	 中小板指；399006.ZICN 創業板指；
    # 000016.ZICN 上證50；000300.ZICN 滬深300；399905.ZICN 中證500
    # 000905.ZICN 中證500;SPX.ZIUS 標普500;000012.ZICN 上證國債;000013.ZICN 上證企業債
    if ticker in ['000001.ZICN','399001.ZICN','399005.ZICN','399006.ZICN','000016.ZICN','000300.ZICN','399905.ZICN','000905.ZICN','SPX.ZIUS','000012.ZICN','000013.ZICN']: 
        px = DataAPI.MktIdxdGet(indexID=ticker,beginDate=start,endDate=end,field=u"tradeDate,closeIndex",pandas="1")
        px = px.rename(columns={'tradeDate':'Date','closeIndex':'Close'})
    else:
        px = DataAPI.MktEqudAdjGet(secID=ticker,beginDate=start,endDate=end,field=u"tradeDate,closePrice",pandas="1") 
        px = px.rename(columns={'tradeDate':'Date','closePrice':'Close'})
    px['Date'] =  pd.to_datetime(px['Date']) #因為DataAPI所有日期都是以字串形式存儲的，因此此處需要轉換
    px = px.set_index("Date")
    px['return'] = px['Close'].pct_change().fillna(0.)
    
    return px['return']


startdate = '20120101'
enddate = '20150101'
secIDs = ['000300.ZICN','000905.ZICN','399006.ZICN','SPX.ZIUS','000012.ZICN','000013.ZICN','050003.OFCN']   # 七類資產的secID
rtns = DataFrame()
for i in range(len(secIDs)-1):
    cp = DataAPI.JY.MktIdxdJYGet(indexID=secIDs[i],startDate=startdate,endDate=enddate,field=u"secShortName,tradeDate,closeIndex",pandas="1")
    cp.sort(columns = 'tradeDate', inplace = True)
    cp.columns = ['secShortName','tradeDate','return']
    cp['return'][1:] = 1.0 * cp['return'][1:].values / cp['return'][:-1].values - 1  
    cp['return'][:1] = 0
    rtns = pd.concat([rtns,cp],axis = 0)  #  dataframe連接操作
cp = DataAPI.JY.FundNavJYGet(secID=secIDs[len(secIDs)-1],beginDate=startdate,endDate=enddate,field=u"secShortName,endDate,dailyProfit",pandas="1")
cp.columns = ['secShortName','tradeDate','return']
cp['return'] = cp['return'].values / 10000
rtns = pd.concat([rtns,cp],axis = 0)
rtn_table = pd.crosstab(rtns['tradeDate'],rtns['secShortName'], values = rtns['return'], aggfunc = sum)  #  一維表變為二維表
rtn_table = rtn_table[[6,2,3,5,1,0,4]]
rtn_table.fillna(0, inplace = True)  #  將NaN置換為0
rtn_table.head(20)

#先隨便計算一下指標，年化收益率，年化標準差
rtn_table.mean() * 250
rtn_table.std() * np.sqrt(250)

#接下來計算我們關心的相關係數矩陣
rtn_table.corr()

# 接下來，就來對比繪製efficient frontier，從實際中直觀感知資產多元化帶來的風險分散效果
##   構建兩個組合作為對比，組合一僅包含滬深300、中證500、創業板、國債、貨幣，組合二則包含了組合一、標普500、企業債

##   繪製effiecient frontier用到了凸優化包cvxopt，關於cvxopt的用法詳細介紹，參見。。。。。

##   在構建efficient frontier中，預期收益採取市場中性原則，用過去三年的平均收益

from cvxopt import matrix, solvers

portfolio1 = [0,1,2,4,6]
portfolio2 = range(7)
cov_mat = rtn_table.cov() * 250   # 協方差矩陣
exp_rtn = rtn_table.mean() * 250   # 標的預期收益

def cal_efficient_frontier(portfolio): 
    #簡單的容錯處理
    if len(portfolio) <= 2 or len(portfolio) > 7:
        raise Exception('portfolio必須為長度大於2小於7的list！') 
    # 數據準備
    cov_mat1 = cov_mat.iloc[portfolio][portfolio]
    exp_rtn1 = exp_rtn.iloc[portfolio]
    max_rtn = max(exp_rtn1)
    min_rtn = min(exp_rtn1)
    risks = [] 
    returns = []
    # 均勻選取20個點來作圖
    for level_rtn in np.linspace(min_rtn, max_rtn, 20):   
        sec_num = len(portfolio)
        P = 2*matrix(cov_mat1.values)
        q = matrix(np.zeros(sec_num))
        G = matrix(np.diag(-1 * np.ones(sec_num)))
        h = matrix(0.0, (sec_num,1))
        A = matrix(np.matrix([np.ones(sec_num),exp_rtn1.values]))
        b = matrix([1.0,level_rtn])
        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q, G, h, A, b)
        risks.append(sol['primal objective'])
        returns.append(level_rtn)
    return np.sqrt(risks), returns

#  計算畫圖資料
risk1, return1 = cal_efficient_frontier(portfolio1)
risk2, return2 = cal_efficient_frontier(portfolio2)

# 在上述準備好資料之後，接下來就構建組合一(滬深300、中證500、創業板、國債、貨幣)和組合二(組合一 + 標普500、企業債)的efficient frontier
fig = plt.figure(figsize = (14,8))
ax1 = fig.add_subplot(111)
ax1.plot(risk1,return1)
ax1.plot(risk2,return2)
ax1.set_title('Efficient  Frontier', fontsize = 14)
ax1.set_xlabel('Standard  Deviation', fontsize = 12)
ax1.set_ylabel('Expected  Return', fontsize = 12)
ax1.tick_params(labelsize = 12)
ax1.legend(['portfolio1','portfolio2'], loc = 'best', fontsize = 14)

# 接下來，給定預期收益，得到最優權重
##  如上分析，在得到最優的efficient frontier之後（本例中為組合二），便可以在資產池中進行資產配置
##  假定某投資者的風險厭惡係數為3（係數越大，表明越厭惡風險，投資更保守），那麼就可以借鑒均方差優化來計算自由的資產配置權重

risk_aversion = 3
P = risk_aversion * matrix(cov_mat.values)
q = -1 * matrix(exp_rtn.values)
G = matrix(np.vstack((np.diag(np.ones(len(exp_rtn))),np.diag(-np.ones(len(exp_rtn))))))
h = matrix(np.array([np.ones(len(exp_rtn)),np.zeros(len(exp_rtn))]).reshape(len(exp_rtn)*2,1))
A = matrix(np.ones(len(exp_rtn)),(1,len(exp_rtn)))
b = matrix([1.0])
solvers.options['show_progress'] = False
sol = solvers.qp(P,q, G, h, A, b)
DataFrame(index=exp_rtn.index,data = np.round(sol['x'],2), columns = ['weight'])  # 權重精確到小數點後兩位
