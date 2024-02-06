from flask import Flask, request, jsonify
from model import PredictionModel

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Получение JSON данных из запроса
    json_data = request.get_json()

    # Проверка наличия ключа 'suitableTeachers' в JSON данных
    if 'suitableTeachers' not in json_data:
        return jsonify({'error': 'Invalid JSON data'})

    # Создание экземпляра класса PredictionModel
    model = PredictionModel('data')

    # Выполнение предсказаний
    model.predict()

    # Возвращение файла solution.csv в JSON формате
    with open('solution.csv', 'r') as f:
        solution_data = f.read()

    return jsonify({'solution': solution_data})

if __name__ == '__main__':
    app.run()
