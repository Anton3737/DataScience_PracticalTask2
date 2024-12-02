import pandas as pd

def save_data_to_csv(data, filename="generated_data.csv"):
    """Зберігаємо дані у CSV-файл."""
    data.to_csv(filename, index=False)
    print("Дані збережено у файл", filename)
