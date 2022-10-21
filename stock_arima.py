import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from datetime import datetime
#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

stock = pd.read_csv('train.csv',  index_col=0)
stock2 = pd.read_csv('test.csv',  index_col=0)
df_stock = stock
df_stock2 = stock2

train_data, test_data = df_stock, df_stock2
training_data = train_data['Close'].values
test_data = test_data['Open'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = sm.tsa.arima.ARIMA(history, order=(4,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_data, model_predictions)
rmse = np.sqrt(MSE_error)

#checking accuracy
print('Testing Mean Squared Error is {}'.format(MSE_error))
print('Testing Root Mean Squared Error is {}'.format(rmse))
