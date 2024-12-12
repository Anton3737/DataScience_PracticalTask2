import pandas as pd

# Завантаження даних
data = pd.read_excel("Pr_1.xls")

# Виведення початкових даних для перевірки
print("Перші рядки датасету:")
print(data.head())

# Визначаємо, які колонки нам потрібні для обробки
# Основні дані: A - F (перші 6 колонок)
# Мапа Код магазину -> Регіон: J та K
main_columns = ["Код магазину", "Дата", "Місяц", "КільКість реалізацій", "Собівартість одиниці", "Ціна реализації"]
map_columns = ["Код магазину.1", "Регіон"]

# Розділення основного датасету і мапи
main_data = data[main_columns]
region_map = data[map_columns].drop_duplicates()

# Перевіряємо наявність пропущених значень у мапі регіонів
if region_map.isnull().any().any():
    print("У мапі регіонів є пропущені значення! Очищення...")
    region_map = region_map.dropna()

# Створюємо словник для швидкого пошуку Регіону за Кодом магазину
region_dict = region_map.set_index("Код магазину.1")["Регіон"].to_dict()

main_data = main_data.copy()
main_data.loc[:, "Регіон"] = main_data["Код магазину"].map(region_dict)

# Перевіряємо, чи всі значення колонок "Регіон" були заповнені
missing_regions = main_data[main_data["Регіон"].isnull()]
if not missing_regions.empty:
    print("Деякі записи не мають відповідного регіону. Їх кількість:", len(missing_regions))
    print("Перші кілька таких записів:")
    print(missing_regions.head())

# Оновлюємо колонку "Місяц" на основі колонки "Дата"
def extract_month(date):
    month_names = {
        1: "Січень", 2: "Лютий", 3: "Березень", 4: "Квітень", 5: "Травень", 6: "Червень",
        7: "Липень", 8: "Серпень", 9: "Вересень", 10: "Жовтень", 11: "Листопад", 12: "Грудень"
    }
    return month_names[pd.to_datetime(date).month]

main_data["Місяц"] = main_data["Дата"].apply(extract_month)

# Зберігаємо очищений датасет у новий файл
output_file_path = "cleaned_data.scv"
main_data.to_csv(output_file_path, index=False)

print(f"Очищений датасет збережено у файл: {output_file_path}")
