import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


# Параметри рівномірного розподілу
low = -1  # Нижній межа
high = 1  # Верхня межа
num_samples = 1000  # Кількість зразків

# Генерація похибки вимірювання
measurement_error = np.random.uniform(low, high, num_samples)


# Постійна величина для досліджуваного процесу
def constant_process(size, value=10):
    return np.full(size, value)


# Генерація даних для досліджуваного процесу
process_data = constant_process(num_samples, value=10)

# Адитивна модель експериментальних даних
experimental_data = process_data + measurement_error

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

# Дисперсія
variance = np.var(experimental_data)

# Стандартне відхилення
std_dev = np.std(experimental_data)

# Математичне сподівання
mean = np.mean(experimental_data)

# Побудова гістограми експериментальних даних
plt.hist(experimental_data, bins=50)
plt.title("Histogram of Experimental Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Основні параметри
num_samples = 1000
a, b, c = 1, 0.5, 2
x = np.linspace(-10, 10, num_samples)
process_data = a * x ** 2 + b * x + c

# Комбінації значень середнього та стандартного відхилення для гістограм
params = [
    (0, 1), (0, 5), (0, 10),
    (5, 1), (5, 5), (5, 10),
    (10, 1), (10, 5), (10, 10)
]

# Налаштування для кількох підграфіків
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Гістограми розподілу даних при зміні середнього значення похибки та стандартного відхилення")

for idx, (mean, std_dev) in enumerate(params):
    row, col = divmod(idx, 3)

    # Генеруємо похибку з параметрами mean і std_dev
    measurement_error = np.random.normal(mean, std_dev, num_samples)
    experimental_data = process_data + measurement_error

    # Побудова гістограми
    axs[row, col].hist(experimental_data, bins=50, color='skyblue', edgecolor='black', alpha=0.6)
    axs[row, col].set_title(f"Mean Error={mean}, Std Dev Error={std_dev}")
    axs[row, col].set_xlabel("Значення")
    axs[row, col].set_ylabel("Частота")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
