import random
import logging
import json
import os
import sys
from flask import Flask, request, jsonify

# добавляем корень проекта в путь, чтобы найти src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.predict import predict

app = Flask(__name__)

# логирование в JSON-формат
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# проверка работоспособности сервиса
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

# основной эндпоинт предсказания
# параметр version=v1|v2, если не указан — случайно 50/50 (A/B)
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # force=True позволяет принять JSON даже без Content-Type заголовка
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'нет данных в теле запроса'}), 400

    # определяем версию модели
    version = request.args.get('version')
    if version not in ('v1', 'v2'):
        # случайное распределение 50/50 для A/B
        version = random.choice(['v1', 'v2'])

    # убираем целевую переменную если случайно передали
    features = {k: v for k, v in data.items() if k != 'default.payment.next.month'}

    try:
        prediction, probability = predict(features, version=version)
    except Exception as e:
        logger.error(f'ошибка предсказания: {e}')
        return jsonify({'error': str(e)}), 500

    result = {
        'prediction': prediction,
        'probability': round(probability, 4),
        'model_version': version
    }

    logger.info(json.dumps({'version': version, 'prediction': prediction, 'probability': round(probability, 4)}))
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
