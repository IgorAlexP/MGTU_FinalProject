import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn

# Создание подключения к базе данных SQLite
engine = create_engine('sqlite:///mydatabase.db')

# Загрузка исходных данных из файла Excel
raw_data = pd.read_excel('C:\\Users\\clic\\OneDrive\\Рабочий стол\\Study\\MGTU_final_project\\data2.xlsx')

# Добавляем префикс к именам столбцов
prefix = 'dwh_'
raw_data.columns = [prefix + col for col in raw_data.columns]

# Сохраняем данные в стейджинговой таблице
staging_table_name = 'staging_table'
raw_data.to_sql(staging_table_name, engine, if_exists='replace', index=False)

# Читаем данные из таблицы
result_data = pd.read_sql(f'SELECT * FROM {staging_table_name}', con=engine)

# Добавляем разбиение на обучающий и тестовый наборы
train_data, test_data = train_test_split(result_data, test_size=0.2, random_state=42)

# Выбираем первые 10 строк для обучения
train_data_subset = train_data.head(10)

# Исключаем столбцы с типом 'object'
numerical_columns = train_data_subset.select_dtypes(include=['number']).columns
X_train = train_data_subset[numerical_columns].drop(columns='dwh_product_id')  # Исключаем целевую переменную
y_train = train_data_subset['dwh_product_id']

# Заполняем пропущенные значения
imputer = SimpleImputer(strategy='mean')  # Можно выбрать другие стратегии
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Инициализация модели (в данном случае - RandomForestClassifier)
model = RandomForestClassifier(random_state=42)

# Обучение модели
model.fit(X_train_imputed, y_train)

# Предсказание на тестовых данных (для получения метрик)
X_test = test_data[numerical_columns].drop(columns='dwh_product_id')  # Исключаем целевую переменную
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
y_pred = model.predict(X_test_imputed)

# Рассчет метрик
accuracy = accuracy_score(test_data['dwh_product_id'], y_pred)

# Вывод метрик
print(f'Accuracy: {accuracy}')

# Сохранение модели в MLFlow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "random_forest_model")
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() is not None else None

# Загрузка тестовых данных из базы данных
test_data_db = pd.read_sql(f'SELECT * FROM {staging_table_name}', con=engine)

# Восстановление модели из MLflow
if run_id:
    model_path = f"runs:/{run_id}/random_forest_model"
    model_loaded = mlflow.sklearn.load_model(model_path)

    # Предсказание на тестовых данных
    X_test_db = test_data_db[numerical_columns].drop(columns='dwh_product_id')  # Исключаем целевую переменную
    X_test_db_imputed = pd.DataFrame(imputer.transform(X_test_db), columns=X_test_db.columns)
    y_pred_db = model_loaded.predict(X_test_db_imputed)

    # Создание новой таблицы для записи результатов предсказаний
    predictions_table_name = 'predictions'
    predictions = pd.DataFrame({'dwh_product_id': test_data_db['dwh_product_id'], 'predicted_dwh_product_id': y_pred_db})
    predictions.to_sql(predictions_table_name, engine, if_exists='replace', index=False)

    # Вывод успешного завершения
    print("Предсказания успешно записаны в базу данных.")
else:
    print("Ошибка: run_id не определен.")



# Визуализация распределения целевой переменной в тестовых данных
plt.figure(figsize=(10, 6))
sns.countplot(x='dwh_product_id', data=test_data)
plt.title('Распределение целевой переменной в тестовых данных')
plt.xlabel('Product ID')
plt.ylabel('Count')
plt.show()

# Визуализация фактических и предсказанных значений
plt.figure(figsize=(10, 6))
sns.scatterplot(x='dwh_product_id', y='predicted_dwh_product_id', data=predictions)
plt.title('Фактические vs. Предсказанные значения')
plt.xlabel('Фактический Product ID')
plt.ylabel('Предсказанный Product ID')
plt.show()

# Визуализация важности признаков (если применимо, например, для модели RandomForest)
if isinstance(model_loaded, RandomForestClassifier):
    feature_importances = model_loaded.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Признак': feature_names, 'Важность': feature_importances})
    importance_df = importance_df.sort_values(by='Важность', ascending=False)

    # Визуализация важности признаков
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Важность', y='Признак', data=importance_df)
    plt.title('Важность признаков')
    plt.show()
