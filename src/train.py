import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# загружаем данные
df = pd.read_csv('data/UCI_Credit_Card.csv')

# убираем колонку ID — она не нужна для предсказания
df = df.drop(columns=['ID'])

# целевая переменная
TARGET = 'default.payment.next.month'

X = df.drop(columns=[TARGET])
y = df[TARGET]

# разбиваем на обучение и тест (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# v1 — логистическая регрессия со стандартизацией
# class_weight='balanced' — классы несбалансированы (дефолтов ~22%), без этого recall будет низким
pipeline_v1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

pipeline_v1.fit(X_train, y_train)
y_pred_v1 = pipeline_v1.predict(X_test)
print('=== Model v1: LogisticRegression ===')
print(classification_report(y_test, y_pred_v1))

# v2 — случайный лес без скейлинга, тоже с балансировкой классов
pipeline_v2 = Pipeline([
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
])

pipeline_v2.fit(X_train, y_train)
y_pred_v2 = pipeline_v2.predict(X_test)
print('=== Model v2: RandomForestClassifier ===')
print(classification_report(y_test, y_pred_v2))

# сохраняем модели
joblib.dump(pipeline_v1, 'models/model_v1.pkl')
joblib.dump(pipeline_v2, 'models/model_v2.pkl')

# сохраняем тестовую выборку для A/B анализа
X_test_saved = X_test.copy()
X_test_saved[TARGET] = y_test.values
X_test_saved.to_csv('data/test_data.csv', index=False)

print('Модели сохранены: models/model_v1.pkl, models/model_v2.pkl')
print('Тестовые данные сохранены: data/test_data.csv')
