import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Імпорт даних
def load_data(filename):
    df = pd.read_csv(filename)
    print("Дані успішно завантажені!")
    return df


# 2. Нормалізація критеріїв
def normalize_data(df):
    criteria_max = ["Точність часу", "Рейтинг популярності"]
    criteria_min = [
        "Ціна",
        "Тривалість роботи від батареї",
        "Вага",
        "Водостійкість",
        "Вартість ремонту",
        "Зручність використання",
        "Екологічний вплив",
        "Час доставки"
    ]

    normalized_df = df.copy()
    for col in criteria_max:
        normalized_df[col] = df[col] / df[col].max()
    for col in criteria_min:
        normalized_df[col] = df[col].min() / df[col]
    return normalized_df


# 3. Розрахунок ефективності
def calculate_efficiency(df):
    df["Ефективність"] = df.iloc[:, 1:].mean(axis=1)
    return df


# 4. Побудова графіків
def plot_data(original_df, normalized_df, results_df):
    # Гістограма розподілу значень критеріїв
    original_df.iloc[:, 1:].hist(bins=10, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Розподіл значень критеріїв", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Стовпчастий графік ефективності
    plt.figure(figsize=(10, 6))
    plt.bar(results_df["Продукт"], results_df["Ефективність"], color='seagreen')
    plt.title("Порівняння ефективності продуктів", fontsize=16)
    plt.xlabel("Продукт")
    plt.ylabel("Ефективність")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Кореляційна матриця
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_df.iloc[:, 1:].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Кореляційна матриця критеріїв", fontsize=16)
    plt.tight_layout()
    plt.show()


# 5. Збереження результатів
def save_results(df, filename):
    df.to_csv(filename, index=False)
    print(f"Результати збережено у файл {filename}")


# Основний скрипт
if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "results.csv"

    # Читання даних
    data = load_data(input_file)

    # Нормалізація
    normalized_data = normalize_data(data)

    # Розрахунок ефективності
    results = calculate_efficiency(normalized_data)

    # Збереження результатів
    save_results(results, output_file)

    # Побудова графіків
    plot_data(data, normalized_data, results)

    print("Процес завершено. Перевірте згенеровані файли та візуалізації.")
