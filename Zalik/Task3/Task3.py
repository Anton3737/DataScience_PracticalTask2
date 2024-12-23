import pandas as pd
import matplotlib.pyplot as plt

# Завантажуємо дані з CSV файлу
data = pd.read_csv("delServices.csv")

# Перетворюємо стовпець "Popularity (%)" на числовий тип
data['Popularity (%)'] = pd.to_numeric(data['Popularity (%)'], errors='coerce')


# Функція для розрахунку індексу ефективності
def calculate_effectiveness(row):
    # Розрахунок середнього часу доставки
    avg_delivery_time = (row['Delivery Time min'] + row['Delivery Time max']) / 2

    # Синтез показників ефективності (0 або 1)
    mobile_app = 1 if row['Mobile App'] == 'Yes' else 0
    api = 1 if row['API'] == 'Yes' else 0
    loyalty_program = 1 if row['Loyalty Program'] == 'Yes' else 0
    international_shipping = 1 if row['International Shipping'] == 'Yes' else 0
    fulfillment = 1 if row['Fulfillment'] == 'Yes' else 0
    time_slot_delivery = 1 if row['Time Slot Delivery'] == 'Yes' else 0
    large_cargo_shipping = 1 if row['Large Cargo Shipping'] == 'Yes' else 0
    user_rating = row['User Rating (1-10)']
    popularity = row['Popularity (%)'] / 100  # Перетворюємо у відсотки

    # Зважений показник ефективності
    effectiveness_score = (avg_delivery_time * 0.1 +
                           mobile_app * 0.1 +
                           api * 0.1 +
                           loyalty_program * 0.1 +
                           international_shipping * 0.1 +
                           fulfillment * 0.1 +
                           time_slot_delivery * 0.1 +
                           large_cargo_shipping * 0.1 +
                           user_rating * 0.2 +
                           popularity * 0.2)

    return effectiveness_score


# Додаємо колонку з розрахованим показником ефективності
data['Effectiveness Score'] = data.apply(calculate_effectiveness, axis=1)

# Обробка NaN значень: заміняємо їх на 0
data['Effectiveness Score'] = data['Effectiveness Score'].fillna(0)

# Сортуємо за індексом ефективності
sorted_data = data.sort_values(by='Effectiveness Score', ascending=False)

# Виведемо кілька графіків для порівняння критеріїв

# 1. Графік для порівняння часу доставки
plt.figure(figsize=(10, 6))
plt.bar(sorted_data['Delivery Service'], sorted_data['Delivery Time max'], color='skyblue', label='Макс. час доставки')
plt.bar(sorted_data['Delivery Service'], sorted_data['Delivery Time min'], color='orange', alpha=0.7,
        label='Мін. час доставки')
plt.xlabel('Служба доставки')
plt.ylabel('Час доставки (днів)')
plt.title('Порівняння часу доставки')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Графік для порівняння рейтингу користувачів
plt.figure(figsize=(10, 6))
plt.bar(sorted_data['Delivery Service'], sorted_data['User Rating (1-10)'], color='green')
plt.xlabel('Служба доставки')
plt.ylabel('Рейтинг користувача')
plt.title('Порівняння рейтингу користувачів')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Графік для порівняння популярності
plt.figure(figsize=(10, 6))
plt.bar(sorted_data['Delivery Service'], sorted_data['Popularity (%)'], color='purple')
plt.xlabel('Служба доставки')
plt.ylabel('Популярність (%)')
plt.title('Порівняння популярності')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. Графік для порівняння ефективності
plt.figure(figsize=(10, 6))
plt.bar(sorted_data['Delivery Service'], sorted_data['Effectiveness Score'], color='red')
plt.xlabel('Служба доставки')
plt.ylabel('Індекс ефективності')
plt.title('Порівняння індексів ефективності')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Виводимо результат
print(sorted_data[['Delivery Service', 'Effectiveness Score']])