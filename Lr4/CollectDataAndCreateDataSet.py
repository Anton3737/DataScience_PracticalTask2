import requests
import pandas as pd
from datetime import datetime, timedelta


# Функція для отримання курсу валют за певну дату
def get_exchange_rate(date):
    url = f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?date={date}&json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        rates = []

        for item in data:
            rates.append({
                'Currency': item['txt'],
                'Rate': item['rate'],
                'Date': item['exchangedate']
            })

        return rates
    else:
        print(f"Не вдалося отримати дані для {date}")
        return []


# Діапазон дат з 01.01.2024 до поточної дати
start_date = datetime(2024, 1, 1)
end_date = datetime.now()

# Список для збереження даних
all_data = []

# Проходимо по кожному дню в діапазоні
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    print(f"Збираємо дані за {date_str}...")

    rates = get_exchange_rate(date_str)
    all_data.extend(rates)

    # Переходимо до наступного дня
    current_date += timedelta(days=1)

# Створення DataFrame
df = pd.DataFrame(all_data)

# Записуємо в Excel файл
excel_filename = "exchange_rates.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Дані успішно записані в файл {excel_filename}")
