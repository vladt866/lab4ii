import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                            precision_score, recall_score, confusion_matrix)
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("C:\\Users\\tvlad\\PycharmProjects\\Pythonlab1\\mean_mode_norm_onehot_with_deleting_train.csv")

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop('Transported', axis=1)
y = df['Transported']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация моделей
rf = RandomForestClassifier(random_state=42, max_depth=3)
gb = GradientBoostingClassifier(random_state=42, max_depth=3)

# Обучение моделей
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Предсказания
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

# Функция для оценки моделей
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Gradient Boosting", y_test, gb_pred)

# Перекрестная проверка с разными метриками (добавлены F1 и Recall)
print("\nCross-validation results:")
for scoring in ['accuracy', 'f1', 'precision', 'recall']:
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring=scoring)
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring=scoring)
    print(f"\nMetric: {scoring}")
    print(f"Random Forest mean {scoring}: {rf_cv.mean():.3f}")
    print(f"Gradient Boosting mean {scoring}: {gb_cv.mean():.3f}")

# Визуализация сравнения моделей
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_scores = [
    accuracy_score(y_test, rf_pred),
    precision_score(y_test, rf_pred),
    recall_score(y_test, rf_pred),
    f1_score(y_test, rf_pred)
]
gb_scores = [
    accuracy_score(y_test, gb_pred),
    precision_score(y_test, gb_pred),
    recall_score(y_test, gb_pred),
    f1_score(y_test, gb_pred)
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar([i - width/2 for i in x], rf_scores, width, label='Random Forest', color='skyblue')
plt.bar([i + width/2 for i in x], gb_scores, width, label='Gradient Boosting', color='salmon')

plt.ylabel('Score')
plt.title('Comparison of Classification Models')
plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавляем значения на столбцы
for i in x:
    plt.text(i - width/2, rf_scores[i] + 0.02, f'{rf_scores[i]:.3f}',
             ha='center', va='bottom')
    plt.text(i + width/2, gb_scores[i] + 0.02, f'{gb_scores[i]:.3f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()