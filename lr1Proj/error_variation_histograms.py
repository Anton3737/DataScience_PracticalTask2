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

# Налаштування для кількох підграфіків (subplots)
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Гістограми розподілу даних при зміні середнього та стандартного відхилення похибки")

for idx, (mean, std_dev) in enumerate(params):
    row, col = divmod(idx, 3)

    # Генерація похибки з параметрами mean і std_dev
    measurement_error = np.random.normal(mean, std_dev, num_samples)
    experimental_data = process_data + measurement_error

    # Побудова гістограми
    axs[row, col].hist(experimental_data, bins=50, color='skyblue', edgecolor='black', alpha=0.6)
    axs[row, col].set_title(f"Середнє={mean}, Стандартне відхилення={std_dev}")
    axs[row, col].set_xlabel("Значення")
    axs[row, col].set_ylabel("Частота")

# Оформлення заголовків і відступів
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
