"""Module that preprocesses the data needed for the machine learning model.

Downloads and preprocesses stock data obtained from Public Google BigQuery tables.
"""
import numpy as np
import pandas as pd
import baostock as bs

def load_data(start_date,end_date):
  """Load stock market data (close values for each day) for given tickers.

  Args:
    tickers (list): list of tickers

  Returns:
    pandas.dataframe: dataframe with close values of tickers

  """
  print(start_date)
  print(end_date)
  lg = bs.login()  

  ##读入上证综指的历史数据
  szzs_history = bs.query_history_k_data("000001.SH", "date,close",
                               start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")  

  ##读入深证指数的历史数据
  szzz_history = bs.query_history_k_data("sz.399106", "date,close",
                               start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")  

  szzs_data = []
  while (szzs_history.error_code == '0') & szzs_history.next():  # 逐条获取数据并合并
      szzs_data.append(szzs_history.get_row_data())
  szzs_df = pd.DataFrame(szzs_data,columns=szzs_history.fields)
  szzs_df = szzs_df.rename(columns={"close":"szzs_close"})
  szzs_df['date'] = pd.to_datetime(szzs_df['date'])
  # szzs_df = szzs_df.set_index('date')  

  szzz_data = []
  while (szzz_history.error_code == '0') & szzz_history.next():  # 逐条获取数据并合并
      szzz_data.append(szzz_history.get_row_data())
  szzz_df = pd.DataFrame(szzz_data, columns=szzz_history.fields)
  szzz_df = szzz_df.rename(columns={"close":"szzz_close"})
  szzz_df['date'] = pd.to_datetime(szzz_df['date'])
  closing_data = pd.merge(szzs_df,szzz_df,on='date', how='outer')
  closing_data = closing_data.set_index('date')
  closing_data=closing_data.astype(float)
  print(closing_data)
  return closing_data


def preprocess_data(closing_data):
  """Preprocesses data into time series.

  Args:
    closing_data (pandas.dataframe):  dataframe with close values of tickers

  Returns:
    pandas.dataframe: dataframe with time series

  """
  print(closing_data)
  # transform into log return
  log_return_data = pd.DataFrame()
  log_return_data['szzz_log_return'] = np.log(closing_data['szzz_close']/closing_data['szzz_close'].shift())
  log_return_data['szzs_log_return'] = np.log(closing_data['szzs_close']/closing_data['szzs_close'].shift())

  log_return_data['szzs_log_return_positive'] = 0
  log_return_data.loc[log_return_data['szzs_log_return'] >= 0, 'szzs_log_return_positive'] = 1
  log_return_data['szzs_log_return_negative'] = 0
  log_return_data.loc[log_return_data['szzs_log_return'] < 0, 'szzs_log_return_negative'] = 1

  # create dataframe
  training_test_data = pd.DataFrame(
    columns=[
      'szzs_log_return_positive', 'szzs_log_return_negative',
      'szzs_log_return_1', 'szzs_log_return_2', 'szzs_log_return_3',
      'szzz_log_return_1', 'szzz_log_return_2', 'szzz_log_return_3'])  

  for i in range(7, len(log_return_data)):
    szzs_log_return_positive = log_return_data['szzs_log_return_positive'].iloc[i]
    szzs_log_return_negative = log_return_data['szzs_log_return_negative'].iloc[i]
    szzs_log_return_1 = log_return_data['szzs_log_return'].iloc[i-1]
    szzs_log_return_2 = log_return_data['szzs_log_return'].iloc[i-2]
    szzs_log_return_3 = log_return_data['szzs_log_return'].iloc[i-3]
    szzz_log_return_1 = log_return_data['szzz_log_return'].iloc[i]
    szzz_log_return_2 = log_return_data['szzz_log_return'].iloc[i-1]
    szzz_log_return_3 = log_return_data['szzz_log_return'].iloc[i-2]
    training_test_data = training_test_data._append(
      {'szzs_log_return_positive':szzs_log_return_positive,
      'szzs_log_return_negative':szzs_log_return_negative,
      'szzs_log_return_1':szzs_log_return_1,
      'szzs_log_return_2':szzs_log_return_2,
      'szzs_log_return_3':szzs_log_return_3,
      'szzz_log_return_1':szzz_log_return_1,
      'szzz_log_return_2':szzz_log_return_2,
      'szzz_log_return_3':szzz_log_return_3},
      ignore_index=True)

  return training_test_data


def train_test_split(training_test_data, train_test_ratio=0.8):
  """Splits the data into a training and test set according to the provided ratio.

  Args:
    training_test_data (pandas.dataframe): dict with time series
    train_test_ratio (float): ratio of train test split

  Returns:
    tensors: predictors and classes tensors for training and respectively test set

  """
  predictors_tf = training_test_data[training_test_data.columns[2:]]
  classes_tf = training_test_data[training_test_data.columns[:2]]

  training_set_size = int(len(training_test_data) * train_test_ratio)

  train_test_dict = {'training_predictors_tf': predictors_tf[:training_set_size],
                     'training_classes_tf': classes_tf[:training_set_size],
                     'test_predictors_tf': predictors_tf[training_set_size:],
                     'test_classes_tf': classes_tf[training_set_size:]}

  return train_test_dict
