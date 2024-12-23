import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe

# Генерація даних із квадратичним трендом
np.random.seed(152)
x = np.linspace(0, 10, 50)
y = 3 * x**2 - 2 * x + 5 + np.random.normal(0, 10, size=len(x))

# Побудова квадратичного тренду
coeffs = np.polyfit(x, y, deg=2)  # Коєфіцієнти тренду
trend = np.polyval(coeffs, x)     # Значення тренду

# Усунення тренду (залишки)
residuals = y - trend

# Візуалізація даних
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Дані з трендом", color="blue")
plt.plot(x, trend, label="Квадратичний тренд", color="red")
plt.scatter(x, residuals, label="Залишки", color="green")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.legend()
plt.title("Аналіз тренду та залишків")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Статистичні характеристики залишків
stats = describe(residuals)
print("Опис залишків:")
print(f"Середнє: {stats.mean}")
print(f"Дисперсія: {stats.variance}")
print(f"Скошеність (Skewness): {stats.skewness}")
print(f"Ексцес (Kurtosis): {stats.kurtosis}")
