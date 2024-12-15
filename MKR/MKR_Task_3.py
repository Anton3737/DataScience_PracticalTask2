import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from scipy import stats

# Завантаження даних
data = pd.read_csv('taxi_trip_pricing.csv')

# Огляд пропущених значень
print("Перевірка на пропущені значення перед очищенням:")
print(data.isnull().sum())

# Аналіз типів даних у колонках
print("\nТипи даних колонок:")
print(data.dtypes)

# Окремо обробляємо числові та категоріальні стовпці
# Числові стовпці
num_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Для числових стовпців заповнюємо середнім значенням
imputer_num = SimpleImputer(strategy='mean')
data[num_cols] = imputer_num.fit_transform(data[num_cols])

# Категоріальні стовпці
cat_cols = data.select_dtypes(include=['object']).columns

# Для категоріальних стовпців заповнюємо найбільш частим значенням
imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

# Перевірка після заповнення пропусків
print("\nПеревірка на пропущені значення після заповнення:")
print(data.isnull().sum())

# Перевіряємо статистику для числових стовпців після заповнення
print("\nОписова статистика числових стовпців після заповнення:")
print(data[num_cols].describe())

# Перевіряємо кількість унікальних значень для категоріальних стовпців
print("\nКількість унікальних значень у категоріальних стовпцях після заповнення:")
print(data[cat_cols].nunique())

# Збереження очищеного датасету
data.to_csv('cleaned_taxi_trip_pricing.csv', index=False)

print("\nОчищений датасет збережено.")

# Додавання нових колонок: 'hour' для години доби та 'day_of_week' для дня тижня
print("\nНазви стовпців в даному датасеті:")
print(data.columns)

# Якщо є стовпець Time_of_Day, перетворимо його в категорії
if 'Time_of_Day' in data.columns:
    time_of_day_map = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    data['time_of_day'] = data['Time_of_Day'].map(time_of_day_map)
else:
    print("Стовпець 'Time_of_Day' не знайдено!")

# Додавання дня тижня
if 'Day_of_Week' in data.columns:
    data['day_of_week'] = pd.to_datetime(data['Day_of_Week'], format='%Y-%m-%d', errors='coerce').dt.day_name()
else:
    print("Стовпець 'Day_of_Week' не знайдено!")

# Перетворення 'day_of_week' на Weekday і Weekend
data['day_of_week'] = data['day_of_week'].apply(lambda x: 'Weekday' if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 'Weekend')

# Графік залежності ціни від часу доби
plt.figure(figsize=(10, 6))
sns.boxplot(x='time_of_day', y='Trip_Price', data=data)
plt.title('Залежність вартості поїздки від часу доби')
plt.xlabel('Час доби')
plt.ylabel('Вартість поїздки (USD)')
plt.xticks([0, 1, 2, 3], ['Morning', 'Afternoon', 'Evening', 'Night'])
plt.show()

# Графік залежності ціни від дня тижня (Weekday vs Weekend)
plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='Trip_Price', data=data)
plt.title('Залежність вартості поїздки від дня тижня')
plt.xlabel('Тип дня')
plt.ylabel('Вартість поїздки (USD)')
plt.xticks(rotation=0)
plt.show()

# Графік залежності ціни від погодних умов
plt.figure(figsize=(10, 6))
sns.boxplot(x='Weather', y='Trip_Price', data=data)
plt.title('Залежність вартості поїздки від погодних умов')
plt.xlabel('Погодні умови')
plt.ylabel('Вартість поїздки (USD)')
plt.show()

# Тепер будуємо модель квадратичного закону з нормальним законом похибки

# Вибір числових колонок для моделі
X = data[['Trip_Distance_km', 'Trip_Duration_Minutes']]  # Незалежні змінні
y = data['Trip_Price']  # Залежна змінна

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення поліноміальної ознаки (квадратичний термін)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Побудова моделі лінійної регресії на поліноміальних ознаках
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = model.predict(X_poly_test)

# Оцінка якості моделі
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Виведення коефіцієнтів моделі та оцінки
print("Коефіцієнти моделі:", model.coef_)
print("Перехоплення:", model.intercept_)
print("Середня квадратична похибка (MSE):", mse)
print("R^2:", r2)

# Перевірка нормальності похибки
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Розподіл похибок")
plt.show()

# Графік порівняння реальних значень і прогнозованих
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Реальні значення')
plt.ylabel('Прогнозовані значення')
plt.title('Порівняння реальних і прогнозованих значень')
plt.show()
