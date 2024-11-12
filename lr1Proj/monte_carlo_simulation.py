import numpy as np
import matplotlib.pyplot as plt

# Параметри рівномірного розподілу
low = -1  # Нижня межа
high = 1  # Верхня межа
num_samples = 1000  # Кількість зразків

# Генерація похибки вимірювання
measurement_error = np.random.uniform(low, high, num_samples)

# Функція для створення постійної величини досліджуваного процесу
def constant_process(size, value=10):
    return np.full(size, value)

# Генерація даних досліджуваного процесу
process_data = constant_process(num_samples, value=10)

# Адитивна модель експериментальних даних
experimental_data = process_data + measurement_error

# Налаштування Монте-Карло симуляції
num_simulations = 1000
results = []

for _ in range(num_simulations):
    # Повторна генерація похибки
    measurement_error = np.random.uniform(low, high, num_samples)
    # Повторне обчислення експериментальних даних
    experimental_data = process_data + measurement_error
    # Збереження середнього значення експериментальних даних
    results.append(np.mean(experimental_data))

# Результати Монте-Карло
monte_carlo_mean = np.mean(results)
monte_carlo_std_dev = np.std(results)

# Обчислення статистичних показників
variance = np.var(experimental_data)  # Дисперсія
std_dev = np.std(experimental_data)   # Стандартне відхилення
mean = np.mean(experimental_data)     # Математичне сподівання

# Побудова гістограми експериментальних даних
plt.hist(experimental_data, bins=50)
plt.title("Гістограма експериментальних даних")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.show()
