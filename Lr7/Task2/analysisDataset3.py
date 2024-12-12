import pandas as pd
from matplotlib import pyplot as plt

# Зчитуємо файл XLS
file_path = 'Data_Set_3.xls'
data = pd.read_excel(file_path)

# Перевірка формату даних у колонках 'Unit Cost' та 'Total' і перетворення формул на значення
# Якщо в колонках є формули, їх потрібно замінити на значення
data['Total'] = data['Total'].apply(pd.to_numeric, errors='coerce')  # Перевести в числовий формат
data['Unit Cost'] = data['Unit Cost'].apply(pd.to_numeric, errors='coerce')  # Те саме для Unit Cost

# Збереження в CSV
csv_file_path = 'Data_Set_3.csv'
data.to_csv(csv_file_path, index=False)

print(f"Файл було успішно збережено як {csv_file_path}")

# Читання CSV
data = pd.read_csv(csv_file_path)


# ---------------------------------------

# Підготовка даних
data.columns = [col.strip() for col in data.columns]  # Видалення зайвих пробілів у назвах колонок
data['Total'] = pd.to_numeric(data['Total'], errors='coerce')  # Переведення "Total" у числовий формат
data.dropna(subset=['Total'], inplace=True)  # Видалення рядків з порожнім "Total"


# ---------------------------------------

# Аналіз даних
# Загальний обсяг продажів та прибуток по регіонах
sales_by_region = data.groupby('Region')['Total'].sum()

# Товар з найбільшою кількістю продажів у кожному регіоні
top_items_by_region = data.groupby(['Region', 'Item'])['Units'].sum().reset_index()
top_items_by_region = top_items_by_region.loc[top_items_by_region.groupby('Region')['Units'].idxmax()]

# Кількість замовлень, виконаних кожним продавцем
orders_by_rep = data['Rep'].value_counts()

# Візуалізація
# Графік продажів по регіонах
sales_by_region.plot(kind='bar', title='Загальний обсяг продажів по регіонах', ylabel='Продажі ($)')
plt.show()

# Графік кількості замовлень на типи товарів
items_count = data['Item'].value_counts()
items_count.plot(kind='pie', autopct='%1.1f%%', title='Розподіл замовлень за типами товарів')
plt.show()


# ---------------------------------------


# Обчислення загальної кількості проданих одиниць та суми продажів по кожному продавцю
sales_data = data.groupby('Rep').agg({'Units': 'sum', 'Total': 'sum'}).reset_index()

# Створення графіків
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Графік кількості проданих одиниць
ax[0].bar(sales_data['Rep'], sales_data['Units'], color='skyblue')
ax[0].set_title('Кількість проданих одиниць по продавцям')
ax[0].set_xlabel('Продавець')
ax[0].set_ylabel('Кількість одиниць')
ax[0].tick_params(axis='x', rotation=45)

# Графік суми продажів
ax[1].bar(sales_data['Rep'], sales_data['Total'], color='lightcoral')
ax[1].set_title('Сума продажів по продавцям')
ax[1].set_xlabel('Продавець')
ax[1].set_ylabel('Сума продажів')
ax[1].tick_params(axis='x', rotation=45)

# Показати графіки
plt.tight_layout()
plt.show()

# ---------------------------------------


# Перетворення колонки OrderDate у формат дати
data['OrderDate'] = pd.to_datetime(data['OrderDate'])

# Агрегуємо дані по датах: підсумовуємо кількість товару, проданого в кожну дату
sales_by_date = data.groupby('OrderDate')['Units'].sum().reset_index()

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(sales_by_date['OrderDate'], sales_by_date['Units'], marker='o', linestyle='-', color='b')

# Налаштування графіка
plt.title('Обсяг замовлень товару за часом')
plt.xlabel('Дата замовлення')
plt.ylabel('Кількість товару')
plt.xticks(rotation=45)
plt.grid(True)

# Показуємо графік
plt.tight_layout()
plt.show()
