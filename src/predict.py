import joblib
import pandas as pd

# фиксированный порядок колонок — должен совпадать с обучением
FEATURE_COLUMNS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# загружаем модель по версии
def load_model(version='v1'):
    path = f'models/model_{version}.pkl'
    model = joblib.load(path)
    return model

# предсказание для одного объекта
def predict(features: dict, version='v1'):
    model = load_model(version)
    # собираем DataFrame с нужным порядком колонок
    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return prediction, probability
