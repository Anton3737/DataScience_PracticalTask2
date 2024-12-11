import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Завантаження очищеного датасету
data = pd.read_excel("cleaned_data.xlsx")

# 2. Попередній аналіз даних
print("Перші кілька рядків датасету:")
print(data.head())

# Оцінка основних статистичних показників
print("\nОсновні статистичні показники:")
print(data.describe())

# Перевірка на наявність пропущених значень
print("\nПропущені значення:")
print(data.isnull().sum())

# 3. Визначення показників ефективності
# Показники: Загальний обсяг продажів (Кількість реалізацій * Ціна реалізації)
data["Продажі"] = data["КільКість реалізацій"] * data["Ціна реализації"]

# Прибуток: (Ціна реалізації - Собівартість одиниці) * Кількість реалізацій
data["Прибуток"] = (data["Ціна реализації"] - data["Собівартість одиниці"]) * data["КільКість реалізацій"]

# 4. Визначення математичної моделі даних
# Створимо регресійну модель для прогнозування прибутку на основі кількості реалізацій та ціни реалізації
X = data[["КільКість реалізацій", "Ціна реализації"]]
y = data["Прибуток"]

# Модель лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Перевірка коефіцієнтів моделі
print("\nКоефіцієнти лінійної регресії:")
print(f"Коефіцієнт для Кількості реалізацій: {model.coef_[0]}")
print(f"Коефіцієнт для Ціни реалізації: {model.coef_[1]}")
print(f"Перехоплення: {model.intercept_}")

# 5. Прогнозування динаміки зміни прибутку за регіонами
# Групуємо дані по регіонах і обчислюємо загальний прибуток за кожним регіоном
region_profit = data.groupby("Регіон")["Прибуток"].sum().reset_index()

# Створення таблиці прогнозів
predictions = model.predict(X)

# Додаємо прогнозовані значення до датасету
data["Прогнозований прибуток"] = predictions

# Групуємо за регіонами, щоб порівняти реальний та прогнозований прибуток
region_profit_forecast = data.groupby("Регіон")[["Прибуток", "Прогнозований прибуток"]].sum().reset_index()

# Виведення результатів
print("\nПрибуток та прогнозований прибуток по регіонах:")
print(region_profit_forecast)

# Побудова графіку для порівняння реального та прогнозованого прибутку за регіонами
plt.figure(figsize=(10, 6))
plt.bar(region_profit_forecast["Регіон"], region_profit_forecast["Прибуток"], label="Реальний прибуток", alpha=0.7)
plt.bar(region_profit_forecast["Регіон"], region_profit_forecast["Прогнозований прибуток"], label="Прогнозований прибуток", alpha=0.7)
plt.xlabel("Регіон")
plt.ylabel("Прибуток")
plt.title("Реальний та прогнозований прибуток по регіонах")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Збереження результатів у новий файл
output_file_path = "forecasted_data.xlsx"
region_profit_forecast.to_excel(output_file_path, index=False)

print(f"\nРезультати прогнозування збережено у файл: {output_file_path}")
