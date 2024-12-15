import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Завантажуємо дані з Excel файлу
df = pd.read_excel('exchange_rates.xlsx')

# Перетворюємо стовпці дати на формат datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

# Фільтруємо дані для трьох валют: USD, EUR, GBP
currencies = ['Долар США', 'Євро', 'Фунт стерлінгів']
filtered_data = df[df['Currency'].isin(currencies)]

# Перевірка кількості рядків після фільтрації
print(filtered_data.groupby('Currency').size())

# Створюємо графік для кожної валюти
plt.figure(figsize=(14, 7))

# Для кожної валюти будуємо графік та прогноз
for currency in currencies:
    currency_data = filtered_data[filtered_data['Currency'] == currency]

    # Перевірка на порожні дані
    if currency_data.empty:
        print(f"Немає даних для {currency}")
        continue

    # Сортуємо за датою
    currency_data = currency_data.sort_values(by='Date')

    # Прогноз на основі лінійної регресії
    X = np.array((currency_data['Date'] - currency_data['Date'].min()).dt.days).reshape(-1,
                                                                                        1)  # перетворюємо дату в кількість днів
    y = currency_data['Rate'].values  # курси валют

    # Створення моделі лінійної регресії
    model = LinearRegression()
    model.fit(X, y)

    # Остання дата з наявних даних
    last_day = currency_data['Date'].max()

    # Прогноз на наступні 3 місяці (90 днів), але починаючи з останнього дня
    future_dates = [last_day + pd.Timedelta(days=i) for i in range(1, 91)]  # Додаємо 90 днів до останньої дати
    future_days = np.array(
        [(last_day + pd.Timedelta(days=i) - currency_data['Date'].min()).days for i in range(1, 91)]).reshape(-1,
                                                                                                              1)  # 90 днів для прогнозу
    future_predictions = model.predict(future_days)

    # Виведення прогнозу на поточний та наступний тиждень
    print(f"Прогноз для валюти {currency}:")
    # Поточний тиждень (7 днів з останньої дати)
    for i in range(7):
        print(f"  {future_dates[i].date()}: {future_predictions[i]:.2f}")

    # Наступний тиждень (ще 7 днів після поточного тижня)
    for i in range(7, 14):
        print(f"  {future_dates[i].date()}: {future_predictions[i]:.2f}")

    # Візуалізація
    plt.plot(currency_data['Date'], currency_data['Rate'], label=f'{currency} - Фактичні дані', linestyle='-',
             marker='o')
    plt.plot(future_dates, future_predictions, label=f'{currency} - Прогноз', linestyle='--')

# Налаштовуємо графік
plt.title('Прогноз курсу валют на наступні 3 місяці (USD, EUR, GBP)')
plt.xlabel('Дата')
plt.ylabel('Курс валют')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Показуємо графік
plt.show()
