import pandas as pd
import joblib
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest
from sklearn.metrics import f1_score, precision_score, recall_score

# загружаем тестовые данные
df = pd.read_csv('data/test_data.csv')

TARGET = 'default.payment.next.month'
X_test = df.drop(columns=[TARGET])
y_test = df[TARGET].values

# загружаем обе модели
model_v1 = joblib.load('models/model_v1.pkl')
model_v2 = joblib.load('models/model_v2.pkl')

# получаем предсказания
y_pred_v1 = model_v1.predict(X_test)
y_pred_v2 = model_v2.predict(X_test)

# считаем метрики для каждой модели
f1_v1 = f1_score(y_test, y_pred_v1)
f1_v2 = f1_score(y_test, y_pred_v2)
precision_v1 = precision_score(y_test, y_pred_v1)
precision_v2 = precision_score(y_test, y_pred_v2)
recall_v1 = recall_score(y_test, y_pred_v1)
recall_v2 = recall_score(y_test, y_pred_v2)

print('=== Метрики моделей ===')
print(f'v1 LogisticRegression: F1={f1_v1:.4f}, Precision={precision_v1:.4f}, Recall={recall_v1:.4f}')
print(f'v2 RandomForest:       F1={f1_v2:.4f}, Precision={precision_v2:.4f}, Recall={recall_v2:.4f}')

# имитируем разбиение на группы A и B (случайное 50/50)
np.random.seed(42)
n = len(y_test)
group_mask = np.random.choice([True, False], size=n)

# группа A — v1 (контроль), группа B — v2 (тест)
y_true_a = y_test[group_mask]
y_pred_a = y_pred_v1[group_mask]

y_true_b = y_test[~group_mask]
y_pred_b = y_pred_v2[~group_mask]

# считаем количество правильно пойманных дефолтов (TP) и общее число дефолтов в группе
# доля пойманных дефолтов = recall по классу 1, это и есть наша основная метрика A/B
tp_a = int(np.sum((y_pred_a == 1) & (y_true_a == 1)))
tp_b = int(np.sum((y_pred_b == 1) & (y_true_b == 1)))
n_a = int(np.sum(y_true_a == 1))
n_b = int(np.sum(y_true_b == 1))

print()
print('=== A/B тест: доля пойманных дефолтов ===')
print(f'Группа A (v1): поймано {tp_a} из {n_a} дефолтов, recall={tp_a/n_a:.4f}')
print(f'Группа B (v2): поймано {tp_b} из {n_b} дефолтов, recall={tp_b/n_b:.4f}')

# z-тест для пропорций из курса
alpha = 0.05
_, p_value = proportions_ztest(
    count=[tp_a, tp_b],
    nobs=[n_a, n_b],
    alternative='two-sided'
)
print()
print(f'Z-тест (двусторонний), alpha={alpha}')
print(f'p-value: {round(p_value, 4)}')
if p_value <= alpha:
    print('Отвергаем нулевую гипотезу — разница статистически значима')
else:
    print('Нет оснований отвергнуть нулевую гипотезу — разница не значима')

# доверительный интервал для разности пропорций из курса
def diff_proportion_conf_interval(x_p, n, gamma=0.95):
    alpha_ci = 1 - gamma
    diff = x_p[1] - x_p[0]
    z_crit = -norm.ppf(alpha_ci / 2)
    eps = z_crit * (x_p[0] * (1 - x_p[0]) / n[0] + x_p[1] * (1 - x_p[1]) / n[1]) ** 0.5
    lower_bound = diff - eps
    upper_bound = diff + eps
    return lower_bound, upper_bound

xp_a = tp_a / n_a
xp_b = tp_b / n_b

lower, upper = diff_proportion_conf_interval(
    x_p=[xp_a, xp_b],
    n=[n_a, n_b]
)
print()
print(f'95% ДИ разности recall (B - A): ({round(lower*100, 2)}%, {round(upper*100, 2)}%)')
if lower > 0:
    print('ДИ не содержит 0 и лежит правее — v2 лучше v1')
elif upper < 0:
    print('ДИ не содержит 0 и лежит левее — v1 лучше v2')
else:
    print('ДИ содержит 0 — нет уверенной разницы между моделями')

# бизнес-метрика: сколько денег можно сэкономить
# предположим средний убыток на дефолт = 50 000 NT$
avg_loss_per_default = 50000
saved_a = tp_a * avg_loss_per_default
saved_b = tp_b * avg_loss_per_default
print()
print('=== Бизнес-метрика: предотвращённые потери (NT$) ===')
print(f'Группа A (v1): {saved_a:,} NT$')
print(f'Группа B (v2): {saved_b:,} NT$')
print(f'Разница: {saved_b - saved_a:,} NT$ в пользу {"v2" if saved_b > saved_a else "v1"}')
