import pickle
import pandas as pd
from payment_request_analysis import PaymentRequestAnalysis

class PredictionModel:
    def __init__(self, directory):
        self.directory = directory
    
    def predict(self):
        # Загрузка сохранённой обученной модели
        with open('trained_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Создание экземпляра класса PaymentRequestAnalysis
        payment_analysis = PaymentRequestAnalysis()

        # Подготовка данных
        payment_analysis.load_datasets(self.directory)
        payment_analysis.data_preparing()
        payment_analysis.data_merging()
        ids, df_scaled = payment_analysis.data_scaling()

        # Получение предсказаний
        predictions = model.predict_proba(df_scaled)
        solution = pd.DataFrame(predictions, index=ids, columns=['non_payment_proba', 'payment_proba'])
        solution.to_csv('solution.csv', index_label='id')

# Пример использования класса PredictionModel
if __name__ == '__main__':
    model = PredictionModel('data')
    model.predict()