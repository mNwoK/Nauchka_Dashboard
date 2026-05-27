import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score

# Берем нашу дату
df = pd.read_csv('teachers_dz8.csv') # В СЛУЧАЕ ПРОЕКТА НУЖНО ML_YEARLY_ENRICHED

# train = 2022, test = 2023
X_train = df[df['year'] == 2022].drop(columns=['is_out_next_year', 'year']).copy()
y_train = df[df['year'] == 2022]['is_out_next_year'].copy()

X_test = df[df['year'] == 2023].drop(columns=['is_out_next_year', 'year']).copy()
y_test = df[df['year'] == 2023]['is_out_next_year'].copy()

# Убираем признаки из 2023 года, чтобы не было утечки
bad_cols = ['pkpp_prog_types_2023', 'pkpp_total_courses', 'pkpp_unique_types']

for col in bad_cols:
    X_train = X_train.drop(columns=[col], errors='ignore')
    X_test = X_test.drop(columns=[col], errors='ignore')

# Обработка пропусков
median_attrition = X_train['attrition_rate_2022'].median()

X_train['attrition_rate_2022'] = X_train['attrition_rate_2022'].fillna(median_attrition)
X_test['attrition_rate_2022'] = X_test['attrition_rate_2022'].fillna(median_attrition)

fill_zero_cols = [
    'has_higher_edu',
    'has_prestigious_university',
    'studied_in_moscow_any'
]

for col in fill_zero_cols:
    X_train[col] = X_train[col].fillna(0).astype(int)
    X_test[col] = X_test[col].fillna(0).astype(int)

X_train['is_critical'] = X_train['is_critical'].astype(int)
X_test['is_critical'] = X_test['is_critical'].astype(int)

cat_features = [
    'class_manager',
    'main_position_other',
    'main_position_head',
    'sex',
    'is_critical',
    'studied_in_moscow_any',
    'has_higher_edu',
    'has_prestigious_university'
]

cat_features = [col for col in cat_features if col in X_train.columns]

model_cb = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    random_seed=42,
    verbose=False
)

model_cb.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

probabilities = model_cb.predict_proba(X_test)[:, 1]

# Порог 0.1 берем как в нашей логике: лучше находить больше потенциально увольняющихся
threshold = 0.1
y_pred = (probabilities > threshold).astype(int)

print('Метрики CatBoost, порог = 0.1')
print(classification_report(y_test, y_pred, zero_division=0))
print('ROC-AUC:', round(roc_auc_score(y_test, probabilities), 4))
print('F3-score:', round(fbeta_score(y_test, y_pred, beta=3), 4))