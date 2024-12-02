import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.svm import SVR
import logging
import os

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Конфігураційні параметри
INPUT_FILE = "data_with_anomalies.csv"
DEGREE = 2  # Ступінь полінома
IQR_MULTIPLIER = 1.5  # Для очищення аномалій

# Перевірка наявності файлу
if not os.path.exists(INPUT_FILE):
    logging.error(f"Файл {INPUT_FILE} не знайдено. Переконайтеся, що він існує.")
    exit()

# Завантаження даних
logging.info(f"Завантаження даних з файлу {INPUT_FILE}.")
data = pd.read_csv(INPUT_FILE)

# Базовий аналіз даних
logging.info("Аналіз даних...")
logging.info(f"\n{data.describe()}")

# Побудова гістограм
plt.figure(figsize=(10, 6))
plt.hist(data['process_data'], bins=50, alpha=0.5, label='Process Data')
plt.hist(data['experimental_data'], bins=50, alpha=0.7, label='Experimental Data')
plt.title("Histogram of Process and Experimental Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('histogram.png')

# Очищення даних
logging.info("Видалення аномалій з використанням IQR.")
Q1 = data['experimental_data'].quantile(0.25)
Q3 = data['experimental_data'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - IQR_MULTIPLIER * IQR
upper_bound = Q3 + IQR_MULTIPLIER * IQR

data_cleaned = data[(data['experimental_data'] >= lower_bound) & (data['experimental_data'] <= upper_bound)]

logging.info(f"Кількість рядків до очищення: {len(data)}")
logging.info(f"Кількість рядків після очищення: {len(data_cleaned)}")

# Побудова гістограми очищених даних
plt.figure(figsize=(10, 6))
plt.hist(data['experimental_data'], bins=50, alpha=0.5, label='Original Experimental Data')
plt.hist(data_cleaned['experimental_data'], bins=50, alpha=0.7, label='Cleaned Experimental Data')
plt.title("Histogram of Experimental Data (Before and After Cleaning)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('histogram_cleaned.png')

# Основні статистичні метрики
mean_value = data_cleaned['experimental_data'].mean()
std_dev = data_cleaned['experimental_data'].std()
median_value = data_cleaned['experimental_data'].median()

logging.info(f"Середнє значення: {mean_value:.2f}")
logging.info(f"Стандартне відхилення: {std_dev:.2f}")
logging.info(f"Медіана: {median_value:.2f}")

# Побудова боксплота
plt.figure(figsize=(10, 5))
plt.boxplot(data_cleaned['experimental_data'], vert=False)
plt.title("Box Plot of Cleaned Experimental Data")
plt.xlabel("Value")
plt.savefig('boxplot_cleaned.png')

# Поліноміальна регресія
logging.info("Поліноміальна регресія.")
x = data_cleaned['x']
y = data_cleaned['experimental_data']

coefficients = np.polyfit(x, y, DEGREE)
polynomial = np.poly1d(coefficients)
logging.info(f"Коефіцієнти полінома: {coefficients}")

# Побудова графіка регресії
x_plot = np.linspace(min(x), max(x), 100)
y_plot = polynomial(x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Cleaned Data', color='orange', s=10)
plt.plot(x_plot, y_plot, label=f'Polynomial Regression (degree={DEGREE})', color='blue')
plt.title("Polynomial Regression Fit")
plt.xlabel("x")
plt.ylabel("Experimental Data")
plt.legend()
plt.savefig('polynomial_regression_fit.png')

# Оцінка моделі
y_pred = polynomial(x)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

logging.info(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}")


# Функція для очищення даних від аномалій за допомогою DBSCAN
def detect_anomalies_dbscan(data):
    """ Виявлення аномалій за допомогою DBSCAN (кластеризація) """
    # Нормалізація даних
    data_normalized = (data - np.mean(data)) / np.std(data)
    # Конвертація у формат NumPy
    data_normalized = data_normalized.values.reshape(-1, 1)
    # Кластеризація DBSCAN
    db = DBSCAN(eps=0.5, min_samples=5).fit(data_normalized)
    labels = db.labels_
    # Виявлення індексів аномалій
    anomalies = np.where(labels == -1)[0]
    return anomalies

# Очищення даних від аномалій за допомогою DBSCAN
logging.info("Видалення аномалій з використанням DBSCAN.")
anomalies = detect_anomalies_dbscan(data_cleaned['experimental_data'])
data_cleaned_no_anomalies = data_cleaned.drop(anomalies)

logging.info(f"Кількість рядків після видалення аномалій DBSCAN: {len(data_cleaned_no_anomalies)}")


# Згладжування даних з використанням альфа-бета фільтра
def alpha_beta_filter(data, alpha=0.1, beta=0.1):
    """ Альфа-бета фільтр для згладжування даних """
    filtered_data = []
    estimated_value = data[0]

    for t in range(1, len(data)):
        predicted_value = estimated_value
        estimated_value = alpha * data[t] + (1 - alpha) * (predicted_value)
        filtered_data.append(estimated_value)
    return np.array(filtered_data)


smoothed_data = alpha_beta_filter(data_cleaned_no_anomalies['experimental_data'])

# Побудова графіку згладжених даних
plt.figure(figsize=(10, 6))
plt.plot(smoothed_data, label='Smoothed Data', color='green')
plt.title("Smoothed Experimental Data using Alpha-Beta Filter")
plt.xlabel("Index")
plt.ylabel("Smoothed Data")
plt.legend()
plt.savefig('smoothed_data.png')

# Навчання моделі (SVM) для нелінійної регресії
logging.info("Навчання моделі SVM.")
X = data_cleaned_no_anomalies['x'].values.reshape(-1, 1)
y = data_cleaned_no_anomalies['experimental_data']
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svm_model.fit(X, y)

# Прогнозування з використанням моделі SVM
y_pred_svm = svm_model.predict(X)

# Оцінка моделі SVM
mse_svm = mean_squared_error(y, y_pred_svm)
mae_svm = mean_absolute_error(y, y_pred_svm)
r2_svm = r2_score(y, y_pred_svm)

logging.info(f"SVM Model - MSE: {mse_svm:.2f}, MAE: {mae_svm:.2f}, R^2: {r2_svm:.2f}")

logging.info("Скрипт виконано успішно.")
