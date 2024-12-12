import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Завантаження даних і опису.
def load_data_and_description(sample_path, description_path):

    data = pd.read_excel(sample_path)
    description = pd.read_excel(description_path)
    return data, description


# Валідація даних на основі опису.
def validate_data(data, description):

    if 'Expected_Type' not in description.columns:
        print("Попередження: Колонка 'Expected_Type' відсутня у файлі data_description.xlsx. Перевірте структуру.")
        return

    for column in description['Field_in_data']:
        if column in data.columns:
            expected_type = description.loc[description['Field_in_data'] == column, 'Expected_Type'].values[0]
            if expected_type == 'numeric' and not np.issubdtype(data[column].dtype, np.number):
                print(f"Попередження: Колонка {column} повинна бути числовою, але містить інші типи даних.")
        else:
            print(f"Попередження: Колонка {column} відсутня у даних.")


# Обробка даних: вибір колонок, заповнення пропусків, створення цільової змінної.
def preprocess_data(data, relevant_columns, target_column):

    data = data[relevant_columns + [target_column]].dropna()
    X = data[relevant_columns]
    y = (data[target_column] > 3).astype(int)  # Бінарна цільова змінна
    return X, y, data


# Створення та навчання нейронної мережі.
def create_neural_network(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Вихід з бінарною активацією

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model


# Оцінка моделі: точність та класифікаційний звіт.
def evaluate_model(model, X_test, y_test):

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Точність моделі: {accuracy}")
    print(f"Класифікаційний звіт:\n{report}")
    return y_pred


# Кластеризація даних за допомогою K-Means.
def perform_clustering(X, num_clusters=2):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    distances = kmeans.transform(X).min(axis=1)
    return clusters, distances, kmeans


# Виявлення шахрайства за допомогою аналізу відстаней до центроїдів.
def detect_fraud(data, distances, threshold_multiplier=2):

    threshold = distances.mean() + threshold_multiplier * distances.std()
    data['distance_to_centroid'] = distances
    data['potential_fraud'] = distances > threshold
    fraud_cases = data[data['potential_fraud']]
    print(f"Виявлено {len(fraud_cases)} потенційних шахрайських випадків.")
    if not fraud_cases.empty:
        print("Деталі потенційних шахрайських випадків:")
        print(fraud_cases[['loan_amount', 'loan_days', 'distance_to_centroid']])
    return fraud_cases


# Графік потенційних шахрайських випадків.
def plot_fraud_cases(data, fraud_cases):

    plt.figure(figsize=(8, 6))
    plt.scatter(data['loan_amount'], data['loan_days'], c=data['potential_fraud'], cmap='coolwarm', s=50)
    plt.title("Потенційні шахрайські випадки")
    plt.xlabel("Сума кредиту")
    plt.ylabel("Термін кредиту")
    plt.colorbar(label="Потенційне шахрайство")
    plt.show()


# Графік кластеризації
def plot_clustering(X, clusters, kmeans):

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red')
    plt.title("Графік кластеризації K-Means")
    plt.xlabel("Особливість 1")
    plt.ylabel("Особливість 2")
    plt.colorbar(label="Кластери")
    plt.show()


# Підрахунок кількості кредитів, які будуть і не будуть повернуті
def plot_credit_repayment(y_pred):

    repayment_counts = pd.Series(y_pred.flatten()).value_counts()
    repayment_labels = ['Не повернуто', 'Повернуто']

    plt.figure(figsize=(8, 6))
    plt.bar(repayment_labels, repayment_counts, color=['red', 'green'])
    plt.title("Кількість кредитів, які будуть і не будуть повернуті")
    plt.xlabel("Статус повернення кредиту")
    plt.ylabel("Кількість кредитів")
    plt.show()


# Збереження результатів у файл.
def save_results(data, file_name):

    data.to_csv(file_name, index=False)
    print(f"Результати збережено у файл: {file_name}")


def main():
    # Шляхи до файлів
    sample_data_path = 'sample_data.xlsx'
    data_description_path = 'data_description.xlsx'

    # Завантаження даних і опису
    data, description = load_data_and_description(sample_data_path, data_description_path)

    # Перевірка даних на основі опису
    validate_data(data, description)

    # Вибір колонок
    relevant_columns = [
        'loan_amount', 'loan_days', 'gender_id', 'children_count_id',
        'education_id', 'Marital status', 'prolongation_total_days', 'wizard_type_id'
    ]
    target_column = 'step'

    # Обробка даних
    X, y, processed_data = preprocess_data(data, relevant_columns, target_column)

    # Нормалізація даних
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Розділення даних
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Створення та навчання моделі
    model = create_neural_network(X_train, y_train, X_train.shape[1])

    # Оцінка моделі
    evaluate_model(model, X_test, y_test)

    # Кластеризація
    clusters, distances, kmeans = perform_clustering(X_scaled)
    processed_data['cluster'] = clusters

    # Виявлення шахрайства
    fraud_cases = detect_fraud(processed_data, distances)

    # Виклик функції після оцінки моделі
    y_pred = evaluate_model(model, X_test, y_test)

    # Побудова графіків
    plot_fraud_cases(processed_data, fraud_cases)
    plot_credit_repayment(y_pred)
    plot_clustering(X_scaled, clusters, kmeans)

    # Збереження результатів
    save_results(processed_data, 'processed_data_with_predictions.csv')


if __name__ == "__main__":
    main()
