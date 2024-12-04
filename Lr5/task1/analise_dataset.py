import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Завантаження даних
data = pd.read_csv('Bitcoin_Historical_Data.csv')

# Перегляд основної інформації про датасет
print("Інформація про датасет:")
print(data.info())
print("\nПерші 5 рядків:")
print(data.head())

# Перетворення стовпця 'Date' на тип datetime (якщо є в датасеті)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    print("\nСтовпець 'Date' перетворено на тип datetime.")

# Масштабування даних (стандартизація)
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)

# Перевірка масштабованих даних
print("\nПерші 5 рядків масштабованих даних:")
print(data_scaled.head())

# Підготовка для визначення оптимальної кількості кластерів (метод ліктя)
cluster_range = range(1, 11)
inertia = []

# Обчислення інерції для кожної кількості кластерів
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Побудова графіка для методу "лікоть"
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Метод ліктя для вибору оптимальної кількості кластерів')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.grid(True)
plt.savefig('elbow_method_plot.png')  # Збереження графіка в файл
plt.close()
print("Графік збережено як 'elbow_method_plot.png'")

# Вибір оптимальної кількості кластерів за графіком (наприклад, 3 або 4)
optimal_k = 3  # Можна змінити залежно від результатів графіка

# Виконання кластеризації K-means з оптимальним k
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
data_scaled['Cluster'] = kmeans_optimal.fit_predict(data_scaled)

# Виведення центроїдів кластерів
print("\nЦентри кластерів:")
print(kmeans_optimal.cluster_centers_)

# Перегляд результатів кластеризації
print("\nПерші 5 рядків з кластеризацією:")
print(data_scaled.head())

# Збереження результатів кластеризації у файл
data_scaled.to_csv('clustered_data.csv', index=False)
print("Результати кластеризації збережено у файл 'clustered_data.csv'")

# Візуалізація результатів кластеризації (для 2-х перших ознак)
plt.figure(figsize=(8, 6))
plt.scatter(data_scaled.iloc[:, 0], data_scaled.iloc[:, 1], c=data_scaled['Cluster'], cmap='viridis')
plt.title(f'Кластеризація за допомогою K-means, k={optimal_k}')
plt.xlabel(data_scaled.columns[0])
plt.ylabel(data_scaled.columns[1])
plt.colorbar(label='Cluster')
plt.savefig('clustered_data_plot.png')  # Збереження графіка
plt.close()
print("Графік кластеризації збережено як 'clustered_data_plot.png'")

