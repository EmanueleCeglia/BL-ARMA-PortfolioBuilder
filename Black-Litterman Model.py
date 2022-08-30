import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import requests
import json
import warnings
import random
from statsmodels.tools.sm_exceptions import ConvergenceWarning

#libreria per effettuare l'ADF test, se il p-value del test è minore di 0.05 noi rigettiamo l'ipotesi nulla H0: la serie è non stazionaria
from statsmodels.tsa.stattools import adfuller

#libreria per osservare se c'è autocorrelazione
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#libreria per automatizzare l'adf test e trovare il numero d (differenziazione) di arima
from pmdarima.arima.utils import ndiffs





#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#


#------------------------------------SEZIONE ARIMA --------------------------------------------------------------------------------------------------------#
def ARIMA_PREDICTOR(ticker,start,end,end2):
# =============================================================================
#     start = date(2010,1,1)
#     end = date(2021,6,30)
#     
# =============================================================================
    ## INIZIO GABOLA PER PREZZI TRIMESTRALI
    df = web.get_data_yahoo(ticker,start,end)
    df = df['Close']
    df_q = df.groupby(pd.Grouper(freq='Q')).mean()
    
    values = np.array([])
    index = np.array([])
    
    pos = 0
    for i in df.index:
        for dates in df_q.index:
            if i == dates:
                values = np.append(values, df.values[pos])
                index = np.append(index, i)
        pos += 1
        
    dataframe = pd.DataFrame(data=values, index=index, columns=(['Close']))
    
    missed_dates = []
    for i in df_q.index:
        if i in dataframe.index:
            True
        else:
            missed_dates.append(i)
    
    found_dates = []
    
    for x in range(1,10):
        for i in missed_dates:
            date = i - timedelta(days=x)
            if date in df.index:
                found_dates.append(date)
                missed_dates.remove(i)
    
    found_dates.sort()
            
    lista_date = list(df.index)
    lista_valori = []
    for i in found_dates:
        lista_valori.append(df[lista_date.index(i)])
    
    
    dataframe_missed_values = pd.DataFrame(data=lista_valori, index=found_dates, columns=(['Close']))
    
    frames = [dataframe, dataframe_missed_values]
    
    final_df = pd.concat(frames)
    dataset_adj = final_df.sort_index()
    dataset_adj = pd.Series(data=dataset_adj['Close'], index=dataset_adj.index)
    ##FINE GABOLA PER PREZZI TRIMESTRALI 
    
    for i in range(len(df_q)):
        df_q[i] = dataset_adj[i]   
    
    #NB nel for sopra df_q da semplice serie trimestrale diventa serie trimestrale con valori giusti (non media dei valori del trimestre)

    #dataset_adj = df_q
    dataset_rtn = np.log(df_q/df_q.shift(1)).dropna()
    dataset_rtn.name = ticker+" CC Return"
    
    dataset_rtn.index = pd.DatetimeIndex(dataset_rtn.index.values,
                               freq=dataset_rtn.index.inferred_freq)
    
    
    #MODIFICARE ADJ CON RTN
    diff = dataset_rtn
    
    #con questa funzione posso automatizzare la ricerca del parametro d di arima
    d = ndiffs(diff, test='adf')
# =============================================================================
#     for i in range(4):
#         result = adfuller(diff)
#         if result[1] > 0.05:
#             diff = diff.diff().dropna()
#             d = i + 1
# =============================================================================
    
   
    train_size = int(len(dataset_rtn) - 2)
    train, test = dataset_rtn[0:train_size], dataset_rtn[train_size:]
    #test_adj = dataset_adj[train_size:]
    test = test.tail(2)
    
    #fitRet = sm.tsa.seasonal_decompose(train, model='additive')
    
    
    ar = [1,2,3,4,6,8,10]
    ma = [1,2,3,4,6,8,10]
    RMSE = []
    AIC = []
    params = []
    
    #GRID SEARCH
    for i in ar:
        for y in ma:
            mod = sm.tsa.statespace.SARIMAX(train,				
                                    order=(i, d, y),			
                                    enforce_stationarity=False,		
                                    enforce_invertibility=False)
            results = mod.fit(method='lbfgs', maxiter=400)
            pred_uc = results.get_forecast(steps=2)
            #Creo variabile che contiene la media di upper and lower bound
            predictions = pred_uc.predicted_mean
            predictions = predictions.fillna(0)
            #Salvo nel vettore AIC la AIC del ciclo in corso
            AIC.append(results.aic)
            #Stessa cosa con RMSE
            rmse = np.sqrt(mean_squared_error(test, predictions))
            RMSE.append(rmse)
            #Salvo i parametri usati
            params.append((i,y))
            
    
    #BEST MODEL
    print("BEST MODEL USED:")
    print("RMSE:", min(RMSE))
    print("AIC:", AIC[RMSE.index(min(RMSE))])
    print("AR:", params[RMSE.index(min(RMSE))][0], "D:", d, "MA:", params[RMSE.index(min(RMSE))][1])
    print(f'ADF Statistic: {adfuller(diff)[0]}')
    print(f'p-value: {adfuller(diff)[1]}')
    #FIRST PREDICTION
    mod = sm.tsa.statespace.SARIMAX(train,				
                                    order=(params[RMSE.index(min(RMSE))][0], d, params[RMSE.index(min(RMSE))][1]),			
                                    enforce_stationarity=False,		
                                    enforce_invertibility=False)
    results = mod.fit(method='lbfgs', maxiter=400)
    pred_uc = results.get_forecast(steps=2)
    predictions = pred_uc.predicted_mean
    
    
    #SECOND PREDICTION
    mod2 = sm.tsa.statespace.SARIMAX(dataset_rtn,				
                                    order=(params[RMSE.index(min(RMSE))][0], d, params[RMSE.index(min(RMSE))][1]),			
                                    enforce_stationarity=False,		
                                    enforce_invertibility=False)
    results2 = mod2.fit(method='lbfgs', maxiter=400)
    pred_uc2 = results2.get_forecast(steps=2)
    predictions2 = pred_uc2.predicted_mean
    
    start2 = end
    end2 = end2
    
    df2 = web.get_data_yahoo(ticker,start2, end2)
    df2_ret = ((df2['Close'].tail(1).values / df2['Close'].head(1).values)-1)*100
    
    
    print("------")
    print("Expected Return:", (np.exp(predictions.mean()*2)-1)*100,"%")
    print("Real Return:", (np.exp(test.mean()*2)-1)*100,"%")
    
    with open(ticker+'.txt', 'w') as f:
        f.write(f'RMSE: {min(RMSE)} AIC: {AIC[RMSE.index(min(RMSE))]} AR:  {params[RMSE.index(min(RMSE))][0]} D: {d} MA: {params[RMSE.index(min(RMSE))][1]} ADF Statistic: {adfuller(diff)[0]} p-value: {adfuller(diff)[1]}')                                                        
        f.write('/n')
        f.write(f"Expected Return: {(np.exp(predictions.mean()*2)-1)*100}% Real Return: {(np.exp(test.mean()*2)-1)*100}%")
        f.write('/n')
        f.write(f'Expected Return 2: {(np.exp(predictions2.mean()*2)-1)*100}% Real Return: {df2_ret}% ')
        f.close()
    
    #pred_ci = pred_uc.conf_int() # default 95% confidence interval
    #upper = pred_ci.iloc[:,1]
    #lower = pred_ci.iloc[:,0]
    
# =============================================================================
#     plt.plot(train)
#     plt.plot(test)
#     plt.plot(predictions)
#     
#     #------------------------------------FINE SEZIONE ARIMA--------------------------------------------------------------------------------------------------------#
#     
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
#     ax1.plot(diff)
#     ax1.set_title(f'Diff {d}')
#     plot_acf(diff, ax=ax2);
# =============================================================================
    return np.round((np.exp(test.mean()*2)-1), 2), np.round((np.exp(predictions.mean()*2)-1), 2)

start = date(2010,1,1)
end = date(2020,12,31)
end2 = date(2021,7,1)

lista = ['ADBE','NVDA','NFLX','PFE']


# =============================================================================
# for ticker in lista:
#     ARIMA_PREDICTOR(ticker, start, end, end2)
# 
# =============================================================================


#----------------------------------- SEZIONE BLACK LITTERMAN ---------------------------------------------------------------------------------------------------------#

def log_return_ts(ticker):
    stock = web.DataReader(ticker, data_source='yahoo', start=dt(2019, 7, 1), end=dt(2020, 12, 31))
    stock = stock['Adj Close']
    stock = np.log(stock/stock.shift(1))
    stock = stock.dropna()
    stock = stock.to_frame()
    return stock


def market_capitalization(stocks_name):
    market_cap = np.array([])
    for i in stocks_name:
        print(i)
        data = requests.get('https://financialmodelingprep.com/api/v3/key-metrics/'+i+'?period=quarter&limit=130&apikey=6302451f7ef70482bb09b64dd252839f')
        data = data.json()
        market_cap = np.append(market_cap, data[2]['marketCap'])
    
    tot = sum(market_cap)
    for i in range(len(market_cap)):
        market_cap[i] = market_cap[i]/tot
        
    return np.round(np.matrix(market_cap), 4)

def implied_return(risk_coeff, cov_matrix, w):
    returns = risk_coeff*(cov_matrix * w.T)
    returns = returns/2
    return returns


def cov_matrix(stocks_name, stock_list):
    matrix = pd.concat(stock_list, axis=1)
    cov_matrix = matrix.cov()
    cov_matrix.columns = stocks_name
    cov_matrix.index = stocks_name
    return np.matrix(cov_matrix)


def P(lista_views):
    matrix = np.matrix([0 for i in range(len(lista_views)**2)]).reshape(len(lista_views),len(lista_views))
    for i in range(len(lista_views)):
        matrix[i,i] = lista_views[i]
    
    lista = []
    for i in range(len(lista_views)):
        if(np.sum(np.array(matrix[i,:]))==0):
            lista.append(i)
    
    matrix = np.delete(matrix, lista, 0)
    
    return matrix

#METODO PACCO PER STIMARE LA MATRICE OMEGA 
def omega(P, cov_matrix, tao):
    matrix = np.diag(np.diag(tao * P @ cov_matrix @ P.T))
    return matrix

def risk_aversion_coeff_lambda():
    SP500 = web.DataReader('^GSPC', data_source='yahoo', start=dt(2019,1,1), end=dt(2020, 12, 31))
    SP500 = SP500['Adj Close']
    SP500 = (SP500/SP500.shift(1))-1
    SP500 = SP500.dropna()
    SP500 = SP500.to_frame()
    coeff = (SP500.mean())/(SP500.var())
    return float(coeff)


def black_litterman_return(tao, cov_matrix, P, omega, pi, Q):
    
    bl1 = np.linalg.inv( np.linalg.inv(tao*cov_matrix) + ((P.T) @ np.linalg.inv(omega) @ P)  )
    bl2 = (np.linalg.inv(tao*cov_matrix) @ pi) + (P.T @ np.linalg.inv(omega) @ Q )
    return bl1*bl2


def black_litterman_weights(ER, cov_matrix, lambdA):
    inv_cov = np.linalg.inv(lambdA*cov_matrix)
    weights = inv_cov @ ER
    weights = weights/sum(np.array(weights))
    return np.round(weights, 4)


def idzorek_method(view_confidences, cov_matrix, pi, Q, P, tau, risk_aversion=1):

    view_omegas = []
    for view_idx in range(len(Q)):
        conf = view_confidences[view_idx]
        if conf < 0 or conf > 1:
            raise ValueError("View confidences must be between 0 and 1")

        # Special handler to avoid dividing by zero.
        # If zero conf, return very big number as uncertainty
        if conf == 0:
            view_omegas.append(1e6)
            continue
        
        P_view = P[view_idx].reshape(1, -1)
        alpha = (1 - conf) / conf  
        omega = tau * alpha * P_view @ cov_matrix @ P_view.T  
        view_omegas.append(omega.item())

    return np.diag(view_omegas)

#-----------------------------------------FINE SEZIONE BLACK LITTERMAN---------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#


ADBE = log_return_ts('ADBE')
NVDA = log_return_ts('NVDA')
PFE = log_return_ts('PFE')

V = log_return_ts('V')
JPM = log_return_ts('JPM')
JNJ = log_return_ts('JNJ')

stocks_name = ['ADBE','NVDA','PFE','V','JPM','JNJ']
stock_list = [ADBE,NVDA,PFE,V,JPM,JNJ]


##   -----   SEZIONE VIEWS   -----   ##

#VIEWS 
Q = np.matrix(np.array([0.17, 0.35, 0.08])).T

#MATRICE CON POSIZIONE DEGLI ASSET CHE HANNO LA VIEW
lista_views = [1,1,1,0,0,0]  #Con 1 indica che la stock in quella posizione ha la view con 0 non ha la view
P = P(lista_views)


##----- LAMBDA, MATRICE COVARIANZE, TAU -----##

#COEFFICIENTE DI AVVERSIONE AL RISCHIO CON SP500
lambdA = risk_aversion_coeff_lambda()

#MATRICE COVARIANZE
cov_matrix = cov_matrix(stocks_name, stock_list)

#TAU
tao = 0.1  #between 0 and 1 ANCHE SE LO CAMBIO IL MODELLO RIMANE UGUALE GUGU


##----- PESI E RITORNI IN EQUILIBRIO DI MKT -----#

#PESI IN EQUILIBRIO DI MERCATO
w = market_capitalization(stocks_name)

#IMPLIED RETURN IN EQUILIBRIO DI MERCATO
pi = implied_return(lambdA, cov_matrix, w)*100



def simple_return_ts2(ticker):
    df = web.get_data_yahoo(ticker, date(2021,1,1), date(2021,7,1))
    df = df['Close']
    ret = (df/df.shift(1)).dropna()
    return ret

adobe_ts = simple_return_ts2('ADBE')
nvidia_ts = simple_return_ts2('NVDA')
pfizer_ts = simple_return_ts2('PFE')

visa_ts = simple_return_ts2('V')
jpmorgan_ts = simple_return_ts2('JPM')
jnj_ts = simple_return_ts2('JNJ')

sp500_ts = simple_return_ts2('^GSPC')


a = np.linspace(0.1, 0.9, 9)
b = np.linspace(0.1, 0.9, 9)
c = np.linspace(0.1, 0.9, 9)

comb = []

for i in a:
    for x in b:
        for y in c:
            comb.append((np.round(i, 1), np.round(x, 1), np.round(y, 1)))

#
comb = [1]
#

sharpe_bl = []
sharpe_ts = []
pesi = []
for i in comb:
    
#LIVELLO DI CONFIDENZA CHE HO NELLE VIEWS
    #liv_conf = [i[0], i[1], i[2]]
    liv_conf = [0.1, 0.3, 0.8]
    #print(liv_conf)
##----- OMEGA STIMATA COL METODO DI IDROZEK -----##

#OMEGA STIMATA COL METODO DI IDROZEK
    omega = idzorek_method(liv_conf, cov_matrix, pi, Q, P, tao)
    omega = np.matrix(omega)
    #omega = omega(P, cov_matrix, tao)
    
    ##----- PESI E RITORNI CONSIDERANDO LE VIEWS E I LIVELLI DI CONFIDENZA IN ESSE -----#
    
    #POSTERIOR RETURN
    ER = black_litterman_return(tao, cov_matrix, P, omega, pi, Q)
    
    
    #POSTERIOR WEIGHTS
    weights = black_litterman_weights(ER, cov_matrix, lambdA)
    pesi.append(weights)
    
    #PORTOFLIO PERFOMANCE ------------------------------------------------------------------------------------------------------------------------------
    
    weights_mv = [0.0434, 0.2704, 0.0080, 0.0446, 0.6202, 0.0133]
    
    cap = 10000
    cap_bench = 10000    
    wallet = np.array([])
    benchmark = np.array([])
    
    for i in range(len(adobe_ts)):
        ret = ((adobe_ts[i]*weights[0]) + (nvidia_ts[i]*weights[1]) + (pfizer_ts[i]*weights[2]) + (visa_ts[i]*weights[3]) + (jpmorgan_ts[i]*weights[4]) + (jnj_ts[i]*weights[5]))
        cap = cap*ret
        wallet = np.append(wallet, cap)
        
        ret2 = ((adobe_ts[i]*weights_mv[0]) + (nvidia_ts[i]*weights_mv[1]) + (pfizer_ts[i]*weights_mv[2]) + (visa_ts[i]*weights_mv[3]) + (jpmorgan_ts[i]*weights_mv[4]) + (jnj_ts[i]*weights_mv[5]))
        cap_bench = cap_bench*ret2
        benchmark = np.append(benchmark, cap_bench)
    
    wallet_df = pd.DataFrame(wallet, index = sp500_ts.index)
    benchmark_df = pd.DataFrame(benchmark, index = sp500_ts.index)
    wallet_log_ret = np.log(wallet_df/wallet_df.shift(1)).dropna()
    benchmark_log_ret = np.log(benchmark_df/benchmark_df.shift(1)).dropna()
    
    plt.title('Black-Litterman vs Markowitz')
    plt.plot(wallet_log_ret, label='Black Litterman Wallet')
    plt.plot(benchmark_log_ret, label='Markowitz Wallet')
    plt.xlabel('Time from 01-01-2021 to 01-07-2021')
    plt.ylabel('Investment value')
    plt.legend()
    
    #print(cap)
            
    wallet_ts = pd.Series(wallet, index=sp500_ts.index)
    wallet_log_ret = np.log(wallet_ts/wallet_ts.shift(1)).dropna()
    
    exp_ret = (ER[0]*weights[0]) + (ER[1]*weights[1]) + (ER[2]*weights[2]) + (ER[3]*weights[3]) + (ER[4]*weights[4]) + (ER[5]*weights[5])
    
    sharpe_ratio_bl = exp_ret/np.std(wallet_log_ret)

    sharpe_ratio_ts = ((np.exp(wallet_log_ret.mean())-1)*126)/np.std(wallet_log_ret)

    sharpe_bl.append(sharpe_ratio_bl)
    sharpe_ts.append(sharpe_ratio_ts)







