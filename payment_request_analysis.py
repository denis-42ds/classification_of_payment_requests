import os
import torch
# import pickle
import random
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, precision_score

RANDOM_STATE = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PaymentRequestAnalysis:
	'''
 Осуществляет анализ данных сервиса с использованием методов машинного обучения. 
 Включает функции загрузки набора данных, предварительной обработки признаков, 
 разделения данных, масштабирования признаков, обучения модели и оценки ее производительности.
 '''
	def __init__(self):
		self.orders = None
		self.teachers_info = None
		self.suitable_teachers = None
		self.prefered_teachers_order_id = None
		self.df = None
		self.X_train = None
		self.X_valid = None
		self.X_test = None
		self.y_train = None
		self.y_valid = None
		self.y_test = None
		self.model = CatBoostClassifier(random_state=RANDOM_STATE, text_features=['purpose'], eval_metric='AUC')
		self.accuracy_before_scaling = None
		self.accuracy_after_scaling = None
		self.scaling_effect_percentage = None

	def load_datasets(self):
		'''
  загрузка датасетов с данными
  '''
		orders_path = os.path.join(folder_name, 'orders.feather')
		teachers_info_path = os.path.join(folder_name, 'teachers_info.feather')
		suitable_teachers_path = os.path.join(folder_name, 'suitable_teachers.feather')
		prefered_teachers_order_id_path = os.path.join(folder_name, 'prefered_teachers_order_id.feather')
		self.orders = pd.read_feather(orders_path)
		self.teachers_info = pd.read_feather(teachers_info_path)
		self.suitable_teachers = pd.read_feather(suitable_teachers_path)
		self.prefered_teachers_order_id = pd.read_feather(prefered_teachers_order_id_path)
		
	def data_preparing(self):
		'''
  произзводит первичную обработку датасетов по необходимости:
  - очистка данных от дубликатов строк
  - создание новых признаков
  - изменение типов данных
  '''
		self.orders = self.orders.drop_duplicates().reset_index(drop=True)
		self.orders = self.orders.drop(self.orders[self.orders['status_id'] > 16].index)
		self.orders['fact_of_payment'] = self.orders['status_id'].apply(lambda x: 1 if x == 6 or x == 13 or x == 5 or x == 15 else 0).astype('int8')
		self.orders[['order_date', 'start_date', 'first_lesson_date', 'payment_date']] = self.orders[['order_date', 'start_date', 'first_lesson_date', 'payment_date']].apply(pd.to_datetime)
		self.orders['order_month'] = self.orders['order_date'].dt.month.astype('int8')
		self.orders = self.orders.drop(self.orders[self.orders['subject_id'] > 36].index)
		self.orders['subject_id'] = self.orders['subject_id'].fillna(0)
		self.orders['teacher_sex'] = self.orders['teacher_sex'].astype('int8')
		self.orders['purpose'] = self.orders['purpose'].fillna('data not entered')
		self.orders['lesson_price_cat'] = self.orders['lesson_price'].apply(lambda x: 1 if x < 500 else 2 if 500 <= x < 1500 else 3 if 1500 <= x <= 3000 else 4).astype('int8')
		self.orders['home_metro_id'] = self.orders['home_metro_id'].fillna(-1)
		self.orders = self.orders.drop(self.orders[self.orders['additional_status_id'] == 1].index)
		self.orders['amount_to_pay'] = self.orders['amount_to_pay'].apply(lambda x: pd.to_numeric(x.replace(',', '.'), errors='coerce')).fillna(0).astype('float32')
		self.orders['planned_lesson_number'] = self.orders['planned_lesson_number'].astype('int8')
		self.orders['pupil_category_new_id'] = self.orders['pupil_category_new_id'].fillna(-1)
		self.orders['pupil_category_new_id'] = self.orders['pupil_category_new_id'].astype('int8')
		self.orders['lessons_per_week'] = self.orders['lessons_per_week'].astype('int8')
		self.orders['teacher_experience_from'] = self.orders['teacher_experience_from'].astype('int8')
		self.orders['teacher_experience_to'] = self.orders['teacher_experience_to'].astype('int8')
		self.orders['teacher_age_from'] = self.orders['teacher_age_from'].astype('int8')
		self.orders['teacher_age_to'] = self.orders['teacher_age_to'].astype('int8')
		self.orders['source_id'] = self.orders['source_id'].astype('int8')
		
		self.orders['order_group'] = self.orders['original_order_id'].fillna(self.orders['id'])
		self.orders['fact_of_payment'] = self.orders.groupby('order_group')['fact_of_payment'].transform(lambda x: 1 if x.any() else x)
		self.orders = self.orders[self.orders['original_order_id'].isnull()]
		self.orders = self.orders.drop('order_group', axis=1)

		self.orders = (self.orders.drop(['order_date',
										 'lesson_price',
										 'lesson_place',
										 'add_info',
										 'start_date',
										 'status_id',
										 'comments',
										 'prefered_teacher_id',
										 'first_lesson_date',
										 'creator_id',
										 'teacher_age_from',
										 'teacher_age_to',
										 'original_order_id',
										 'client_id',
										 'additional_status_id',
										 'max_metro_distance',
										 'estimated_fee',
										 'payment_date',
										 'is_display_to_teachers'],
										axis=1))
		
		
		self.teachers_info = self.teachers_info.drop_duplicates().reset_index(drop=True)
		
		self.suitable_teachers = self.suitable_teachers.drop_duplicates().reset_index(drop=True)
		
		self.prefered_teachers_order_id = self.prefered_teachers_order_id.drop_duplicates().reset_index(drop=True)

	def data_merging(self):
		'''
  - производит объединение таблиц,
  - очистку при необходимости
  - создание дополнительных признаков при необходимости
  '''
		self.df = (self.orders
				   .merge(self.suitable_teachers[['teacher_id', 'order_id']],
						  how='left',
						  left_on=['id', 'working_teacher_id'],
						  right_on=['order_id', 'teacher_id'])
				   .drop_duplicates()
				   .reset_index(drop=True)
				   .merge(self.prefered_teachers_order_id,
						  how='left',
						  left_on=['id', 'working_teacher_id'],
						  right_on=['order_id', 'teacher_id'])
				   .drop_duplicates()
				   .reset_index(drop=True)
				  )
		self.df['right_teacher'] = (np.where((self.df['working_teacher_id'] == self.df['teacher_id_x']) & 
											 (self.df['working_teacher_id'] == self.df['teacher_id_y']), 1, 0)
									.astype('int8')
								   )
		self.df = self.df.drop(['working_teacher_id', 'teacher_id_x', 'order_id_x', 'order_id_y', 'teacher_id_y'], axis=1)

	# def data_encoding(self):
	# 	'''
 #  - преобразует текстовые признаки в эмбеддинги,
 #  - получает из них среднее значение,
 #  - добавляет к датафрейму новый признак
 #  '''
	# 	minilm = SentenceTransformer('all-MiniLM-L6-v2', device=device)
	# 	embeddings = minilm.encode(self.df['purpose'].values, device=device)
	# 	self.df['purpose_emb'] = np.mean(embeddings, axis=1)
		
	def data_preparation(self):
		'''
  - производит отделение целевого признака,
    масштабирование данных,
    разделение на три выборки
  - на выходе: три выборки,
    печать размерностей этих выборок
	'''
		order_id = self.df['id']
		y = self.df['fact_of_payment']
		X = self.df.drop(['id', 'fact_of_payment'], axis=1)

		scaler = StandardScaler()
		X_es = (
			pd.DataFrame(scaler.fit_transform(X.drop(['purpose'], axis=1)),
						 columns=X.drop(['purpose'], axis=1).columns,
						 index=X.drop(['purpose'], axis=1).index)
			.merge(X['purpose'],
				   how='left',
				   left_index=True,
				   right_index=True,
				   sort=False)
		)
		X_train, self.X_test, y_train, self.y_test = (train_test_split(X_es,
																	   y,
																	   test_size=0.1,
																	   random_state=RANDOM_STATE,
																	   stratify=y)
													 )
		self.X_train, self.X_valid, self.y_train, self.y_valid = (train_test_split(X_train,
																				   y_train,
																				   test_size=0.2,
																				   random_state=RANDOM_STATE,
																				   stratify=y_train)
																 )
	
		
		
		
		
			
