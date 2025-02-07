# includes
import pandas as pd
import numpy as np
# from scipy.stats import chi2_contingency
# from google.colab import drive
# drive.mount('/content/drive')
#
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras import metrics
# from metrics import  Precision, Recall
import joblib
# import tensorflow as tf
from tensorflow.keras.models import load_model
#scale values
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# p1 = tf.keras.metrics.Precision 
# R1 = tf.keras.metrics.Recall
# load_model = tf.keras.models.load_model 
#data procssing tools
def f1_score(y_true, y_pred):
    # Ensure y_true and y_pred are of type float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # Convert predictions to binary values (0 or 1)
    y_pred = K.round(y_pred)

    # Calculate True Positives, False Positives, False Negatives
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    # Precision and Recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

###############################

# X_scaled = pd.read_csv('Scaled_data.csv',index_col=0)
model = load_model('model_epoch_21.keras', custom_objects={'f1_score': f1_score})
###
# Load encoders and scalers
encoders = joblib.load('encoders.pkl')  # Load encoders dictionary
scalers = joblib.load('scalers.pkl')      # Load scaler object
##

###################################################
#Data preration
def modeling_1(data_model):
  # X_2 = X_scaled.drop(columns=['target'])
  Colmns1 = ['USER_ID', 'TYPE', 'STATE', 'AMOUNT_GBP', 'CURRENCY', 'COUNTRY', 'C_HOUR',
 'C_MINUTE', 'A_DAY', 'A_HOUR', 'A_MINUTE', 'B_YEAR', 'B_MONTH', 'B_DAY',
 'C_unix_time', 'O_unix_time', 'A_unix_time', 'B_unix_time', 'min',
 'since_last_tras', 'F_unix_time', 'avg_transaction_amount',
 'transaction_amount_std', 'Dif1', 'Dif2']
  data_model=data_model[Colmns1]#X_2.columns.values]
  y_user = model.predict(data_model.drop(columns=['USER_ID']),verbose=0)
  # print(data_model.iloc[0])
  # print(X3.head())
  # y_user = model.predict(X3.drop(columns=['USER_ID']))
  # print(f'prediction = {y_user}')
  return y_user
##########################################################
#data ready to model
def get_data_ready_for_model(dataset_1):
  ne_df = dataset_1.copy()
  #recareting user
  #type reverseing
  printed = False
  try:
    coi = ['TYPE','STATE','COUNTRY','CURRENCY']
    for col in coi:
      if col in encoders:
        ne_df[col] = encoders[col].transform(ne_df[col])
        ne_df[col] = scalers[col].transform(ne_df[[col]])
        if printed: print(f'{col}: become {ne_df[col].values}')

    #convert date to ...
    ne_df['C_DATETIME'] = pd.to_datetime(ne_df['CREATED_DATE'])
    ne_df['C_HOUR'] = ne_df['C_DATETIME'].dt.hour
    ne_df['C_MINUTE'] = ne_df['C_DATETIME'].dt.minute

    ne_df['O_DATETIME1'] = (pd.to_datetime(ne_df['CREATED_DATE'])-pd.to_datetime(ne_df['OPENING_DATE']))
    ne_df['A_DAY'] = ne_df['O_DATETIME1'].dt.days
    ne_df['A_HOUR'] = ne_df['O_DATETIME1'].dt.seconds // 3600 %24
    ne_df['A_MINUTE'] = (ne_df['O_DATETIME1'].dt.seconds //60 )%60

    ne_df['B_DATETIME'] = pd.to_datetime(ne_df['BIRTH_DATE'])
    ne_df['B_YEAR'] = ne_df['B_DATETIME'].dt.year
    ne_df['B_MONTH'] = ne_df['B_DATETIME'].dt.month
    ne_df['B_DAY'] = ne_df['B_DATETIME'].dt.day
    # print(f'Birth {ne_df["B_YEAR"].iloc[0]} , {ne_df["B_MONTH"].iloc[0]} , {ne_df["B_DAY"].iloc[0]} ')
    #since we are giving time in second and not milisecond
    #we devide it by 10 ^ 6 not ^9
    ne_df['C_unix_time'] = (pd.to_datetime(ne_df['CREATED_DATE'])).astype('int64') / 10**6  # Convert to seconds
    ne_df['O_unix_time'] = (pd.to_datetime(ne_df['OPENING_DATE'])).astype('int64') / 10**6  # Convert to seconds

    ne_df['A_unix_time'] = ne_df['C_unix_time'] - ne_df['O_unix_time']
    if ne_df['A_unix_time'].iloc[0] < 0:
      print(f"time between tranction and opening is invalid, assigning dfeult value")
      ne_df['A_unix_time'] = 1
    ne_df['B_unix_time'] = (pd.to_datetime(ne_df['BIRTH_DATE'])).astype('int64') / 10**6  # Convert to seconds

    ne_df['F_unix_time'] = ne_df['C_unix_time'] - ne_df['min']
    if ne_df['F_unix_time'].iloc[0] < 0:
      print(f"time between tranction {ne_df['C_unix_time'].values} and first {ne_df['min'].values} is invalid")
      ne_df['F_unix_time'] = 1

  except Exception as e:
    print(f"Error 1: {e}")
  try:

    ne_df['Dif1'] = np.where(ne_df['AMOUNT_GBP'] > ne_df['avg_transaction_amount'],
        ne_df['AMOUNT_GBP'] - ne_df['avg_transaction_amount'],
        ne_df['avg_transaction_amount'] - ne_df['AMOUNT_GBP'])

    ne_df['Dif2'] = np.where(ne_df['AMOUNT_GBP'] > ne_df['avg_transaction_amount'],
                      ne_df['Dif1'] - 2 * ne_df['transaction_amount_std'],
                      ne_df['Dif1'] - 2 * ne_df['transaction_amount_std'])


#applainy apporplarate standrd scaler
    coi = ['AMOUNT_GBP','avg_transaction_amount', 'transaction_amount_std', 'Dif1', 'Dif2',
    'C_unix_time', 'O_unix_time', 'A_unix_time', 'B_unix_time','since_last_tras',
       'F_unix_time','min','B_YEAR','B_MONTH','B_DAY', 'A_DAY', 'A_HOUR',
         'A_MINUTE','C_HOUR', 'C_MINUTE']

    for col in coi:
      if col in scalers:
        ne_df[col] = scalers[col].transform(ne_df[[col]])
        if printed:print(f'{col}: become {ne_df[[col]].values}')
    #drop useless columns
    coli2 = ['CREATED_DATE',
         'BIRTH_DATE', 'OPENING_DATE', 'C_DATETIME',
         'O_DATETIME1',
         'B_DATETIME']
    ne_df = ne_df.drop(columns=coli2)
  except Exception as e:
    # If there is an error, assign the default datetime value
    print(f"Error 2: {e}")
  return ne_df

##########################################################
from datetime import datetime

def prep_data_get_columns(data_i):
  dataset1 = pd.DataFrame([data_i])
  # print(dataset1.columns)
  coli = ['USER_ID', 'C_YEAR', 'C_MONTH', 'C_DAY', 'C_HOUR', 'C_MINUTE',
       'C_SECOND', 'O_YEAR', 'O_MONTH', 'O_DAY', 'O_HOUR', 'O_MINUTE',
       'O_SECOND', 'L_YEAR', 'L_MONTH', 'L_DAY', 'L_HOUR', 'L_MINUTE',
       'L_SECOND', 'MIN_YEAR', 'MIN_MONTH', 'MIN_DAY', 'MIN_HOUR',
       'MIN_MINUTE', 'MIN_SECOND', 'B_YEAR', 'B_MONTH', 'B_DAY', 'AMOUNT_GBP',
       'avg_transaction_amount', 'transaction_amount_std', 'TYPE', 'STATE',
       'CURRENCY', 'COUNTRY']
  ##################################
  #get trasction date in date time format
  try:
    date_time_str = f"{dataset1['C_YEAR'].iloc[0]}-{dataset1['C_MONTH'].iloc[0]}-{dataset1['C_DAY'].iloc[0]}\
    {dataset1['C_HOUR'].iloc[0]}:{dataset1['C_MINUTE'].iloc[0]}:{dataset1['C_SECOND'].iloc[0]}"
    # print(date_time_str)
    # Convert to datetime object
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    dataset1['CREATED_DATE'] = date_time_obj
  except Exception as e:
    # If there is an error, assign the default datetime value
    print(f"Error occurred: {e}. Assigning default value.")
    default_date_time = datetime.strptime("2019-05-06 01:19:21", '%Y-%m-%d %H:%M:%S')
    dataset1['CREATED_DATE'] = default_date_time

  ##########################
  #get oping date in date time format
  try:
    date_time_str = f"{dataset1['O_YEAR'].iloc[0]}-{dataset1['O_MONTH'].iloc[0]}-{dataset1['O_DAY'].iloc[0]}\
      {dataset1['O_HOUR'].iloc[0]}:{dataset1['O_MINUTE'].iloc[0]}:{dataset1['O_SECOND'].iloc[0]}"
    # Convert to datetime object
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    dataset1['OPENING_DATE'] = date_time_obj

  except Exception as e:
    # If there is an error, assign the default datetime value
    print(f"Error occurred: {e}. Assigning default value.")
    default_date_time = datetime.strptime("2019-04-19 23:10:19", '%Y-%m-%d %H:%M:%S')
    dataset1['OPENING_DATE'] = default_date_time
  #################
  #get birthday in date time format
  try:
    date_time_str = f"{dataset1['B_YEAR'].iloc[0]}-{dataset1['B_MONTH'].iloc[0]}-{dataset1['B_DAY'].iloc[0]}"
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d')
    dataset1['BIRTH_DATE'] = date_time_obj
  except Exception as e:
    # If there is an error, assign the default datetime value
    print(f"Error occurred: {e}. Assigning default value.")
    default_date_time = datetime.strptime("2002-02-16", '%Y-%m-%d')
    dataset1['BIRTH_DATE'] = default_date_time

  ################
  #GET SINCE LAST TRASCTION TIME IN SECONDS
  try:
    # Create the datetime string from the dataset
    date_time_str = f"{dataset1['L_YEAR'].iloc[0]}-{dataset1['L_MONTH'].iloc[0]}-{dataset1['L_DAY'].iloc[0]} " \
                    f"{dataset1['L_HOUR'].iloc[0]}:{dataset1['L_MINUTE'].iloc[0]}:{dataset1['L_SECOND'].iloc[0]}"

    # Convert to datetime object
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

    # Assign the converted datetime object to the 'since_last_tras' column
    dataset1['last_tras'] = date_time_obj

  except Exception as e:
    # If there is an error, assign the default datetime value
    print(f"Error occurred: {e}. Assigning default value.")
    default_date_time = datetime.strptime("2019-05-06 08:19:20", '%Y-%m-%d %H:%M:%S')
    dataset1['last_tras'] = default_date_time
  #get time between tracations
  dataset1['since_last_tras'] = (dataset1['CREATED_DATE']-dataset1['last_tras']).dt.total_seconds()
  if dataset1['since_last_tras'].iloc[0] < 0:
    print(f"time between tranction is invalid: Assigning default value.")
    dataset1['since_last_tras'] = 1

  #########################
  #get date of first trasaction in seconds
  try:
    date_time_str = f"{dataset1['MIN_YEAR'].iloc[0]}-{dataset1['MIN_MONTH'].iloc[0]}-{dataset1['MIN_DAY'].iloc[0]}\
    {dataset1['MIN_HOUR'].iloc[0]}:{dataset1['MIN_MINUTE'].iloc[0]}:{dataset1['MIN_SECOND'].iloc[0]}"
    # Convert to datetime object
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    timestamp_seconds = date_time_obj.timestamp()

    dataset1['min'] = timestamp_seconds
  except Exception as e:
    # If there is an error, assign the default datetime value
    print(f"Error occurred: {e}. Assigning default value.")
    default_date_time = datetime.strptime("2019-04-19 23:10:19", '%Y-%m-%d %H:%M:%S')
    timestamp_seconds = default_date_time.timestamp()
    dataset1['min'] = timestamp_seconds

  #######################
  ##turn numbers into numbers
  dataset1['AMOUNT_GBP'] = dataset1['AMOUNT_GBP'].astype(float)
  if dataset1['AMOUNT_GBP'].iloc[0] < 0:
    print(f"AMOUNT_GBP is invalid: Assigning default value.")
    dataset1['AMOUNT_GBP'] = 0.01
  dataset1['avg_transaction_amount'] = dataset1['avg_transaction_amount'].astype(float)
  if dataset1['avg_transaction_amount'].iloc[0] < 0:
    print(f"avg_transaction_amount is invalid: Assigning default value.")
    dataset1['avg_transaction_amount'] = 0.01
  dataset1['transaction_amount_std'] = dataset1['transaction_amount_std'].astype(float)
  if dataset1['transaction_amount_std'].iloc[0] < 0:
    print(f"transaction_amount_std is invalid: Assigning default value.")
    dataset1['transaction_amount_std'] = 0.01

  dataset1 = dataset1.drop(columns=['C_YEAR', 'C_MONTH', 'C_DAY', 'C_HOUR', 'C_MINUTE',
       'C_SECOND', 'O_YEAR', 'O_MONTH', 'O_DAY', 'O_HOUR', 'O_MINUTE',
       'O_SECOND', 'L_YEAR', 'L_MONTH', 'L_DAY', 'L_HOUR', 'L_MINUTE',
       'L_SECOND', 'MIN_YEAR', 'MIN_MONTH', 'MIN_DAY', 'MIN_HOUR',
       'MIN_MINUTE', 'MIN_SECOND', 'B_YEAR', 'B_MONTH', 'B_DAY','last_tras'])
  # print(dataset1.columns)

  return dataset1
##########################################################
