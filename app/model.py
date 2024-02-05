import os
import pickle
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score

from payment_request_analysis import PaymentRequestAnalysis

RANDOM_STATE = 42

#  Загрузка сохранённой обученной модели
with open('trained_prediction_model.pkl', 'rb') as f:
	model = pickle.load(f)

payment_analysis.load_datasets('data')
payment_analysis.data_preparing()
payment_analysis.data_merging()

def data_scaling(df):
    '''
    - производит масштабирование указанных признаков
    - на выходе: масштабированный датафрейм
    '''
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(['purpose', 'contact_result'], axis=1))

    df_scaled = pd.DataFrame(scaled_features, columns=df.drop(['purpose', 'contact_result'], axis=1).columns)
    df_scaled['purpose'] = df['purpose'].astype(str)
    df_scaled['contact_result'] = df['contact_result'].astype(str)

    return df_scaled

df_scaled = data_scaling(payment_analysis.df)

# Получение предсказаний на масштабированном датафрейме
predictions = model.predict(df_scaled)

# Вывод предсказаний
print(predictions)