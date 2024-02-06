import os
import pickle
import random
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score

RANDOM_STATE = 42

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
		self.model = CatBoostClassifier(random_state=RANDOM_STATE,
										text_features=['purpose', 'contact_result'],
										eval_metric='AUC')
		
	def load_datasets(self, folder_name):
		'''
  загрузка датасетов с данными
  '''
		orders_path = os.path.join(folder_name, 'orders_test.feather')  # внесено изменение. верный файл orders.feather
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
		# обработка датафрейма orders
		self.orders.drop_duplicates(inplace=True)
		self.orders.reset_index(drop=True, inplace=True)
		self.orders = self.orders.drop(self.orders[self.orders['status_id'] > 16].index)
		self.orders['fact_of_payment'] = (self.orders['status_id'].isin([6, 13, 5, 15])).astype('int8')
		self.orders[['order_date', 'start_date', 'first_lesson_date', 'payment_date']] = (
			self.orders[['order_date', 'start_date', 'first_lesson_date', 'payment_date']]
			.apply(pd.to_datetime)
		)
		self.orders['order_month'] = self.orders['order_date'].dt.month.astype('int8')
		self.orders = self.orders.drop(self.orders[self.orders['subject_id'] > 36].index)
		self.orders['subject_id'] = self.orders['subject_id'].fillna(0)
		self.orders['teacher_sex'] = self.orders['teacher_sex'].astype('int8')
		self.orders['purpose'] = self.orders['purpose'].fillna('data not entered')
		self.orders['lesson_price_cat'] = (self.orders['lesson_price']
										   .apply(lambda x: 1 if x < 500 
												  else 2 if 500 <= x < 1500 
												  else 3 if 1500 <= x <= 3000 
												  else 4).astype('int8')
										  )
		self.orders['home_metro_id'] = self.orders['home_metro_id'].fillna(-1)
		self.orders = self.orders.drop(self.orders[self.orders['additional_status_id'] == 1].index)
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
		self.orders['fact_of_payment'] = (self.orders.groupby('order_group')['fact_of_payment']
										  .transform(lambda x: 1 if x.any() else x)
										 )
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
										 'is_display_to_teachers',
										 'working_teacher_id',
										 'amount_to_pay'],
										axis=1))
		
		# обработка датафрейма teachers_info
		self.teachers_info = self.teachers_info.drop_duplicates().reset_index(drop=True)
		teachers_columns_duplicated = self.teachers_info.T.duplicated(keep=False)
		teacher_columns_duplicated = (
			list(set(teachers_columns_duplicated[teachers_columns_duplicated].index) - 
				 set(pd.Series(teachers_columns_duplicated[teachers_columns_duplicated].index)
					 .str.split('.').str[0].unique()))
		)
		self.teachers_info = self.teachers_info.drop(columns=teacher_columns_duplicated)
		
		self.teachers_info['photo_path'] = self.teachers_info['photo_path'].astype(str)
		self.teachers_info.fillna({'photo_path': 'No Photo'}, inplace=True)
		self.teachers_info['has_photo'] = (
			self.teachers_info['photo_path']
			.apply(lambda x: 1 if pd.notnull(x) else 0)
			.astype(np.int8)
		)
		self.teachers_info.drop(['date_update',
								 'reg_date',
								 'birth_date',
								 'teaching_start_date',
								 'user_id',
								 'external_comments',
								 'lesson_duration',
								 'lesson_cost',
								 'status_relevant_date',
								 'status_school_id',
								 'status_college_id',
								 'status_relevant_date',
								 'information',
								 'photo_path',
								 'comments',
								 'rules_confirmed_date',
								 'last_visited',
								 'is_pupils_needed',
								 'pupil_needed_date',
								 'amount_to_pay',
								 'remote_comments',
								 'passport_id',
								 'is_individual',
								 'partner_id',
								 'relevance_date',
								 'status_institution_id',
								 'free_time_relevance_date',
								 'rating_for_users_yesterday'],
								inplace=True,
								axis=1)

		# обработка датафрейма suitable_teachers
		self.suitable_teachers.drop_duplicates(inplace=True)
		self.suitable_teachers.reset_index(drop=True, inplace=True)
		self.suitable_teachers = (
			self.suitable_teachers[(self.suitable_teachers['enable_assign'] == 1) | 
			(self.suitable_teachers['enable_auto_assign'] == 1)]
		)

		# обработка датафрейма prefered_teachers_order_id
		self.prefered_teachers_order_id.drop_duplicates(inplace=True)
		self.prefered_teachers_order_id.reset_index(drop=True, inplace=True)

	def data_merging(self):
		'''
  - производит объединение таблиц,
  - очистку при необходимости
  - создание дополнительных признаков при необходимости
  '''
		df_teachers = (
			self.suitable_teachers[['teacher_id', 'order_id', 'contact_result']]
			.merge(self.teachers_info,
				   how='left',
				   left_on=['teacher_id'],
				   right_on=['id'])
			.drop_duplicates()
			.reset_index(drop=True)
			.drop('id', axis=1)
		)
		self.df = (
			self.orders
			.merge(df_teachers,
				   how='left',
				   left_on=['id'],
				   right_on=['order_id'])
			.drop_duplicates()
			.reset_index(drop=True)
			.drop('order_id', axis=1)
		)
		self.df = self.df.dropna(subset=['teacher_id']).reset_index(drop=True)
		self.df['contact_result'] = self.df['contact_result'].astype(str)
		self.df['contact_result'] = self.df['contact_result'].fillna('не заполнено')

	def data_scaling(self):
		'''производит масштабирование числовых признаков'''
		scaler = StandardScaler()
		ids = self.df['id']
		columns_to_scale = self.df.drop(['id', 'purpose', 'contact_result'], axis=1).columns
		scaled_features = scaler.fit_transform(self.df[columns_to_scale])
		df_scaled = pd.DataFrame(scaled_features, columns=columns_to_scale)
		df_scaled[['purpose', 'contact_result']] = self.df[['purpose', 'contact_result']].astype(str)
		df_scaled = df_scaled.drop(['teacher_id', 'fact_of_payment'], axis=1)
				
		return ids, df_scaled


	def data_preparation(self):
		'''
  - производит отделение целевого признака,
    масштабирование данных,
    разделение на три выборки
  - на выходе: три выборки,
    печать размерностей этих выборок
	'''
		ids = self.df[['id', 'teacher_id']]
		y = self.df['fact_of_payment']
		X = self.df.drop(['id', 'teacher_id', 'fact_of_payment'], axis=1)

		scaler = StandardScaler()
		X_es = (
			pd.DataFrame(scaler.fit_transform(X.drop(['purpose', 'contact_result'], axis=1)),
						 columns=X.drop(['purpose', 'contact_result'], axis=1).columns,
						 index=X.drop(['purpose', 'contact_result'], axis=1).index)
			.merge(X[['purpose', 'contact_result']],
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
	
		
	def ml_model_training(self):
		'''
  производит обучение модели и расчёт метрик
  '''
		self.model.fit(self.X_train,
					   self.y_train,
					   eval_set=(self.X_valid, self.y_valid),
					   verbose=100)

		y_pred_proba = self.model.predict_proba(self.X_valid)[:, 1]
		y_pred = self.model.predict(self.X_valid.values)

		roc_auc_value = roc_auc_score(self.y_valid, y_pred_proba)
		precision = precision_score(self.y_valid, y_pred)
		f1 = f1_score(self.y_valid, y_pred)
		
	def evaluate_model(self):
		'''производит проверку модели на отложенной выборке'''
		y_proba_test = self.model.predict_proba(self.X_test)[:, 1]
		roc_auc_test = roc_auc_score(self.y_test, y_proba_test)
		y_pred_test = self.model.predict(self.X_test.values)
		precision_test = precision_score(self.y_test, y_pred_test)
		f1_test = f1_score(self.y_test, y_pred_test)

		features_importance = pd.DataFrame(data = {'feature': self.X_train.columns,
												   'percent': np.round(self.model.feature_importances_, decimals=1)})
		print(features_importance.sort_values('percent', ascending=False).reset_index(drop=True)[:5])
		print(f"ROC-AUC на тестовой выборке: {round(roc_auc_test, 2)}")
		print(f"Precision на тестовой выборке: {round(precision_test, 2)}")
		print(f"F1 на тестовой выборке: {round(f1_test, 2)}")
		
		with open('model_ctbst', 'wb') as f:
			pickle.dump(self.model, f)

	def performing_all_calculations(self, folder_name='data'):
		'''производит выполнение всех функций'''
		self.load_datasets(folder_name)
		self.data_preparing()
		self.data_merging()
		self.data_preparation()
		self.ml_model_training()
		self.evaluate_model()
		
		# Сохранение обученной модели в файл
		with open('trained_prediction_model.pkl', 'wb') as f:
			pickle.dump(model, f)