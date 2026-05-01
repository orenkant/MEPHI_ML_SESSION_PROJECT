import sys
import os
import pytest

# добавляем корень проекта в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# пример корректного запроса с 23 признаками
SAMPLE_INPUT = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
}

def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'ok'

def test_predict_v1(client):
    resp = client.post('/predict?version=v1', json=SAMPLE_INPUT)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data
    assert 'probability' in data
    assert data['model_version'] == 'v1'
    assert data['prediction'] in (0, 1)
    assert 0.0 <= data['probability'] <= 1.0

def test_predict_v2(client):
    resp = client.post('/predict?version=v2', json=SAMPLE_INPUT)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['model_version'] == 'v2'

def test_predict_no_body(client):
    resp = client.post('/predict?version=v1')
    assert resp.status_code == 400

def test_predict_random_version(client):
    # без параметра version — случайный выбор, должно работать
    resp = client.post('/predict', json=SAMPLE_INPUT)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['model_version'] in ('v1', 'v2')
